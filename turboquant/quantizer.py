"""
turboquant/quantizer.py
-----------------------
Core implementation of TurboQuant (arXiv:2504.19874).

Two quantizers:
  TurboQuantMSE   – minimises mean-squared reconstruction error
  TurboQuantProd  – unbiased inner-product estimator (MSE + QJL residual)

Both are data-oblivious (online) — no training required.

Numerical note
--------------
The coordinate marginal PDF (Lemma 1) is a scaled Beta distribution
whose normalisation constant involves Γ(d/2).  For d ≥ 30 this overflows
double precision.  The paper itself states that in high dimensions the
distribution converges to N(0, 1/d), so we switch to that approximation
for d ≥ _GAUSSIAN_THRESHOLD (= 30).  This is exact in the limit and
accurate to < 1 % for d ≥ 30.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm as _sp_norm
from dataclasses import dataclass
import warnings

# Switch to Gaussian N(0,1/d) approximation above this dimension
_GAUSSIAN_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Coordinate distribution helpers
# ---------------------------------------------------------------------------

def _pdf(x: float, d: int) -> float:
    """PDF of one coordinate of a uniform random point on S^{d-1}."""
    if d >= _GAUSSIAN_THRESHOLD:
        return float(_sp_norm.pdf(x, 0.0, 1.0 / np.sqrt(d)))
    if abs(x) >= 1.0:
        return 0.0
    log_coef = gammaln(d / 2) - (0.5 * np.log(np.pi) + gammaln((d - 1) / 2))
    return float(np.exp(log_coef + ((d - 3) / 2) * np.log(max(1 - x * x, 0.0))))


def _integration_limits(d: int) -> tuple[float, float]:
    """
    Integration limits for the coordinate PDF.
    For the Gaussian approx, use ±6σ; for the exact Beta, use ±1.
    """
    if d >= _GAUSSIAN_THRESHOLD:
        sigma = 1.0 / np.sqrt(d)
        return -6 * sigma, 6 * sigma
    return -1.0, 1.0


# ---------------------------------------------------------------------------
# Lloyd-Max solver — 1-D k-means on the coordinate distribution
# ---------------------------------------------------------------------------

def _lloyd_max_centroids(n_levels: int, d: int,
                         n_iter: int = 400, tol: float = 1e-12) -> np.ndarray:
    """
    Solve the 1-D k-means (Lloyd-Max) problem for the coordinate distribution.

    Returns sorted centroids c_1 < … < c_{n_levels}.

    Warm-started at the optimal quantiles of N(0, 1/d).
    """
    sigma = 1.0 / np.sqrt(d)
    # Quantile warm-start (near-optimal for Gaussian)
    probs = (np.arange(1, n_levels + 1) - 0.5) / n_levels
    centroids = _sp_norm.ppf(probs, 0.0, sigma)

    lo_global, hi_global = _integration_limits(d)

    for _ in range(n_iter):
        old = centroids.copy()
        bounds = np.concatenate([[lo_global],
                                  (centroids[:-1] + centroids[1:]) / 2,
                                  [hi_global]])

        new_c = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = bounds[i], bounds[i + 1]
            f = lambda x: _pdf(x, d)
            num, _ = quad(lambda x: x * f(x), lo, hi,
                          limit=120, epsabs=1e-13, epsrel=1e-13)
            den, _ = quad(f, lo, hi,
                          limit=120, epsabs=1e-13, epsrel=1e-13)
            new_c[i] = num / den if den > 1e-20 else (lo + hi) / 2.0

        centroids = np.sort(new_c)
        if np.max(np.abs(centroids - old)) < tol:
            break

    return centroids


# ---------------------------------------------------------------------------
# Codebook cache
# ---------------------------------------------------------------------------

class _CodebookCache:
    """Lazy cache: (b, d) → centroids ndarray."""

    def __init__(self):
        self._store: dict = {}

    def get(self, b: int, d: int) -> np.ndarray:
        key = (b, d)
        if key not in self._store:
            self._store[key] = _lloyd_max_centroids(2 ** b, d)
        return self._store[key]


_CACHE = _CodebookCache()


# ---------------------------------------------------------------------------
# TurboQuantMSE
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    MSE-optimal TurboQuant (Algorithm 1).

    Parameters
    ----------
    d    : int – vector dimension
    b    : int – bits per coordinate  (1 ≤ b ≤ 8)
    seed : int – RNG seed for the rotation matrix
    """

    def __init__(self, d: int, b: int, seed: int = 42):
        if not (1 <= b <= 8):
            raise ValueError(f"b must be in [1, 8], got {b}")
        self.d = d
        self.b = b

        # Random rotation matrix via QR decomposition
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((d, d))
        self.Pi, _ = np.linalg.qr(G)          # (d, d) orthogonal

        # Lloyd-Max codebook
        self.centroids = _CACHE.get(b, d)      # (2^b,)

    # ------------------------------------------------------------------
    def quant(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize unit-norm vectors.

        Parameters
        ----------
        X : (n, d) float64, unit-norm rows

        Returns
        -------
        idx : (n, d) int16 — centroid indices (0-based)

        Implementation note
        -------------------
        Centroids are sorted, so we use np.searchsorted (binary search,
        O(n·d·log 2^b) = O(n·d·b)) instead of broadcasting over all 2^b
        centroids (O(n·d·2^b)).  This is ~8× faster at b=8 and avoids
        the large temporary (n, d, 2^b) array that caused OOM on large n.
        """
        Y = X @ self.Pi.T                          # (n, d) rotated

        # Voronoi boundaries: midpoints between consecutive centroids
        bounds = (self.centroids[:-1] + self.centroids[1:]) / 2.0  # (2^b - 1,)

        # searchsorted gives the insertion point = nearest centroid index
        flat = Y.ravel()
        raw  = np.searchsorted(bounds, flat).astype(np.int16)
        return raw.reshape(Y.shape)

    def dequant(self, idx: np.ndarray) -> np.ndarray:
        """
        Reconstruct vectors from centroid indices.

        Parameters
        ----------
        idx : (n, d) int16

        Returns
        -------
        X_hat : (n, d) float64
        """
        return self.centroids[idx] @ self.Pi           # (n, d)

    def quant_dequant(self, X: np.ndarray) -> np.ndarray:
        """Convenience: quantize then immediately dequantize."""
        return self.dequant(self.quant(X))

    def mse(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error E[‖x − x̃‖²] over a batch."""
        X_hat = self.quant_dequant(X)
        return float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))

    def inner_product_error(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Per-pair signed IP error:  ⟨y, Q⁻¹(Q(x))⟩ − ⟨y, x⟩

        X, Y : (n, d) unit-norm
        Returns (n,) float64.
        """
        X_hat = self.quant_dequant(X)
        return np.sum(X_hat * Y, axis=1) - np.sum(X * Y, axis=1)


# ---------------------------------------------------------------------------
# TurboQuantProd
# ---------------------------------------------------------------------------

@dataclass
class ProdQuantized:
    """Output of TurboQuantProd.quant()."""
    idx:     np.ndarray   # (n, d) int16  — MSE indices at b-1 bits
    qjl:     np.ndarray   # (n, d) int8   — sign(S·r) ∈ {-1, +1}
    r_norms: np.ndarray   # (n,)   float64 — ‖residual‖₂


class TurboQuantProd:
    """
    Inner-product-optimal TurboQuant (Algorithm 2).

    Two-stage pipeline:
      1. TurboQuantMSE at (b-1) bits  → minimise ‖x − x̃_mse‖
      2. QJL (sign sketch) on residual → unbiased IP correction

    Total bit budget: b bits per coordinate.

    Parameters
    ----------
    d    : int – vector dimension
    b    : int – total bits per coordinate (b ≥ 2)
    seed : int – RNG seed
    """

    def __init__(self, d: int, b: int, seed: int = 42):
        if b < 2:
            raise ValueError("TurboQuantProd requires b ≥ 2")
        self.d = d
        self.b = b
        self._mse = TurboQuantMSE(d, b - 1, seed=seed)

        # QJL projection  S ∈ R^{d×d}, i.i.d. N(0,1)
        rng = np.random.default_rng(seed + 999)
        self.S = rng.standard_normal((d, d))           # (d, d)

    def quant(self, X: np.ndarray) -> ProdQuantized:
        """
        Quantize a batch of unit-norm vectors.

        Parameters
        ----------
        X : (n, d) float64, unit-norm rows

        Returns
        -------
        ProdQuantized with fields idx, qjl, r_norms
        """
        idx   = self._mse.quant(X)
        X_mse = self._mse.dequant(idx)
        R     = X - X_mse                             # residual (n, d)

        r_norms = np.linalg.norm(R, axis=1)           # (n,)

        # Normalise residual before QJL to get direction
        safe = np.where(r_norms > 1e-15, r_norms, 1.0)
        R_unit = R / safe[:, None]

        # QJL: sign(S · r̂)
        proj = R_unit @ self.S.T                       # (n, d)
        qjl  = np.sign(proj).astype(np.int8)
        qjl[qjl == 0] = 1                             # resolve sign(0)

        return ProdQuantized(idx=idx, qjl=qjl, r_norms=r_norms)

    def dequant(self, q: ProdQuantized) -> np.ndarray:
        """
        Reconstruct from ProdQuantized.

        Returns
        -------
        X_hat : (n, d) float64
        """
        X_mse = self._mse.dequant(q.idx)              # (n, d)

        # QJL dequant: √(π/2)/d · Sᵀ · z, then scale by ‖r‖
        scale = np.sqrt(np.pi / 2.0) / self.d
        X_qjl = (q.qjl.astype(np.float64) @ self.S) * scale   # (n, d)
        X_qjl *= q.r_norms[:, None]

        return X_mse + X_qjl

    def quant_dequant(self, X: np.ndarray) -> np.ndarray:
        return self.dequant(self.quant(X))

    def inner_product_error(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Per-pair signed IP error:  ⟨y, Q⁻¹(Q(x))⟩ − ⟨y, x⟩
        Should be zero-mean (unbiased) for any fixed X, Y.

        X, Y : (n, d) unit-norm
        Returns (n,) float64.
        """
        X_hat = self.quant_dequant(X)
        return np.sum(X_hat * Y, axis=1) - np.sum(X * Y, axis=1)
