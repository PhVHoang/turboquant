"""
benchmarks/bench_distortion.py
-------------------------------
Reproduces Fig. 3 from the paper:
  - MSE and inner-product error vs bit-width b ∈ {1, 2, 3, 4, 5}
  - Overlaid with upper bound √(3π)/2 · 4^{-b} and lower bound 4^{-b}
  - Both TurboQuantMSE and TurboQuantProd shown on the IP plot

Usage:
    python -m benchmarks.bench_distortion
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turboquant import TurboQuantMSE, TurboQuantProd
from turboquant.bounds import (mse_upper_bound, mse_lower_bound,
                                prod_upper_bound, prod_lower_bound)

# ── Config ────────────────────────────────────────────────────────────────
D         = 1024        # dimension (paper uses 1536; 512 is fast & representative)
N         = 20_000      # number of vectors
BITWIDTHS = [1, 2, 3, 4, 5]
SEED      = 42
OUTDIR    = "plots"
# ──────────────────────────────────────────────────────────────────────────


def generate_data(n: int, d: int, seed: int) -> tuple:
    """Return unit-norm X (n, d) and unit-norm Y (n, d)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n, d)).astype(np.float64)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # sanity check
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9)
    return X, Y


def run(verbose: bool = True) -> dict:
    os.makedirs(OUTDIR, exist_ok=True)
    X, Y = generate_data(N, D, SEED)

    results = {
        "bitwidths": BITWIDTHS,
        "mse_mse":   [],   # TurboQuantMSE MSE
        "ip_mse":    [],   # TurboQuantMSE inner-product error (variance)
        "ip_prod":   [],   # TurboQuantProd inner-product error (variance)
    }

    for b in BITWIDTHS:
        if verbose:
            print(f"  bit-width b={b} ...", end=" ", flush=True)

        # ── MSE quantizer ─────────────────────────────────────────────
        qmse = TurboQuantMSE(D, b, seed=SEED)
        mse_val = qmse.mse(X)
        ip_err_mse = qmse.inner_product_error(X, Y)
        ip_var_mse = float(np.mean(ip_err_mse ** 2))

        results["mse_mse"].append(mse_val)
        results["ip_mse"].append(ip_var_mse)

        # ── Prod quantizer ────────────────────────────────────────────
        if b >= 2:
            qprod = TurboQuantProd(D, b, seed=SEED)
            ip_err_prod = qprod.inner_product_error(X, Y)
            ip_var_prod = float(np.mean(ip_err_prod ** 2))
        else:
            ip_var_prod = float("nan")   # prod needs b >= 2
        results["ip_prod"].append(ip_var_prod)

        if verbose:
            print(f"MSE={mse_val:.5f}  IP_mse={ip_var_mse:.6f}  "
                  f"IP_prod={ip_var_prod:.6f}")

    _plot(results, D)
    return results


def _plot(results: dict, d: int):
    bs  = np.array(results["bitwidths"], dtype=float)
    bs_fine = np.linspace(1, 5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("TurboQuant: distortion vs theoretical bounds", fontsize=13)

    # ── Left: MSE ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(bs, results["mse_mse"], "b-o", lw=2, ms=7,
                label="TurboQuant$_\\mathrm{mse}$")
    ax.semilogy(bs_fine, [mse_upper_bound(b) for b in bs_fine],
                "r--", lw=1.5, label=r"Upper: $\frac{\sqrt{3\pi}}{2}\cdot4^{-b}$")
    ax.semilogy(bs_fine, [mse_lower_bound(b) for b in bs_fine],
                "g--", lw=1.5, label=r"Lower: $4^{-b}$")
    ax.set_xlabel("Bit-width (b)", fontsize=11)
    ax.set_ylabel("Mean squared error ($D_\\mathrm{mse}$)", fontsize=11)
    ax.set_title("MSE distortion", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks([1, 2, 3, 4, 5])

    # ── Right: Inner-product error ─────────────────────────────────────
    ax = axes[1]
    ax.semilogy(bs, results["ip_mse"], "b-o", lw=2, ms=7,
                label="TurboQuant$_\\mathrm{mse}$")

    prod_vals = [v for v in results["ip_prod"] if not np.isnan(v)]
    prod_bs   = [b for b, v in zip(bs, results["ip_prod"]) if not np.isnan(v)]
    ax.semilogy(prod_bs, prod_vals, "m-^", lw=2, ms=7,
                label="TurboQuant$_\\mathrm{prod}$")

    ax.semilogy(bs_fine, [prod_upper_bound(b, y_norm_sq=1.0, d=d) for b in bs_fine],
                "r--", lw=1.5,
                label=r"Upper: $\frac{\sqrt{3\pi}}{2}\cdot\frac{\|y\|^2}{d}\cdot4^{-b}$")
    ax.semilogy(bs_fine, [prod_lower_bound(b, y_norm_sq=1.0, d=d) for b in bs_fine],
                "g--", lw=1.5,
                label=r"Lower: $\frac{\|y\|^2}{d}\cdot4^{-b}$")

    ax.set_xlabel("Bit-width (b)", fontsize=11)
    ax.set_ylabel("Inner-product error ($D_\\mathrm{prod}$)", fontsize=11)
    ax.set_title("Inner-product distortion", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    path = os.path.join("plots", "fig3_distortion_bounds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {path}")
    plt.close()


if __name__ == "__main__":
    print("=== Benchmark: Distortion vs Theoretical Bounds ===")
    run()
