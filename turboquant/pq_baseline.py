"""
turboquant/pq_baseline.py
-------------------------
Simple Product Quantization (PQ) baseline using k-means codebooks.
Used for indexing-time and recall comparison with TurboQuant.

PQ splits each d-dimensional vector into M sub-vectors of dimension d/M,
then independently k-means-quantizes each sub-space.
"""

from __future__ import annotations

import numpy as np
import time
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class PQQuantized:
    codes: np.ndarray   # (n, M) int – centroid indices per sub-space


class ProductQuantizer:
    """
    Standard Product Quantization.

    Parameters
    ----------
    d        : int – vector dimension (must be divisible by M)
    b        : int – bits per sub-space (codebook size = 2^b per sub-space)
    M        : int – number of sub-spaces (default: d//4, capped at 32)
    seed     : int – RNG seed
    """

    def __init__(self, d: int, b: int, M: int | None = None, seed: int = 42):
        self.d = d
        self.b = b
        if M is not None:
            self.M = M
        else:
            cap = min(d // 4, 32)
            self.M = max(m for m in range(1, cap + 1) if d % m == 0)
        assert d % self.M == 0, f"d={d} must be divisible by M={self.M}"
        self.sub_d = d // self.M
        self.k = 2 ** b          # codebook size per sub-space
        self.seed = seed
        self.codebooks: list[np.ndarray] | None = None  # M × (k, sub_d)
        self._train_time: float = 0.0

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "ProductQuantizer":
        """
        Train k-means codebooks on X (n, d).
        This is the expensive offline step – not needed by TurboQuant.
        """
        n = X.shape[0]
        self.codebooks = []
        t0 = time.perf_counter()
        for m in range(self.M):
            sub = X[:, m * self.sub_d: (m + 1) * self.sub_d]
            km = KMeans(n_clusters=self.k, random_state=self.seed,
                        n_init=4, max_iter=100)
            km.fit(sub)
            self.codebooks.append(km.cluster_centers_)   # (k, sub_d)
        self._train_time = time.perf_counter() - t0
        return self

    def quant(self, X: np.ndarray) -> PQQuantized:
        assert self.codebooks is not None, "Call fit() first"
        n = X.shape[0]
        codes = np.zeros((n, self.M), dtype=np.int32)
        for m in range(self.M):
            sub = X[:, m * self.sub_d: (m + 1) * self.sub_d]   # (n, sub_d)
            cb  = self.codebooks[m]                              # (k, sub_d)
            # Nearest centroid: (n, k) distances
            dists = np.sum((sub[:, None, :] - cb[None, :, :]) ** 2, axis=-1)
            codes[:, m] = np.argmin(dists, axis=1)
        return PQQuantized(codes=codes)

    def dequant(self, q: PQQuantized) -> np.ndarray:
        assert self.codebooks is not None
        n = q.codes.shape[0]
        X_hat = np.zeros((n, self.d))
        for m in range(self.M):
            X_hat[:, m * self.sub_d: (m + 1) * self.sub_d] = \
                self.codebooks[m][q.codes[:, m]]
        return X_hat

    def quant_dequant(self, X: np.ndarray) -> np.ndarray:
        return self.dequant(self.quant(X))

    @property
    def train_time(self) -> float:
        return self._train_time
