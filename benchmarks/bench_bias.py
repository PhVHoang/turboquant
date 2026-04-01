"""
benchmarks/bench_bias.py
------------------------
Reproduces Figs. 1 & 2 from the paper:

Fig 1: Error distribution histograms at b=1..4
  - TurboQuantProd : zero-mean (unbiased)
  - TurboQuantMSE  : non-zero mean (biased) at low b

Fig 2: IP error variance vs average inner product at b=2
  - TurboQuantProd: constant variance across IP buckets (unbiased)
  - TurboQuantMSE : bias grows with IP magnitude

Usage:
    python -m benchmarks.bench_bias
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turboquant import TurboQuantMSE, TurboQuantProd

# ── Config ────────────────────────────────────────────────────────────────
D     = 512
N     = 30_000
SEED  = 42
OUTDIR = "plots"
# ──────────────────────────────────────────────────────────────────────────


def _unit_vectors(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def run_fig1(verbose: bool = True):
    """Error distribution histograms for b=1..4."""
    os.makedirs(OUTDIR, exist_ok=True)
    X = _unit_vectors(N, D, SEED)
    Y = _unit_vectors(N, D, SEED + 1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharey=False)
    fig.suptitle("Fig 1: IP error distribution – "
                 "TurboQuant$_\\mathrm{prod}$ (top) vs "
                 "TurboQuant$_\\mathrm{mse}$ (bottom)", fontsize=12)

    for col, b in enumerate([1, 2, 3, 4]):
        # Prod
        if b >= 2:
            qprod = TurboQuantProd(D, b, seed=SEED)
            err_prod = qprod.inner_product_error(X, Y)
        else:
            # b=1: prod not defined; show MSE errors centred for reference
            qmse_tmp = TurboQuantMSE(D, 1, seed=SEED)
            err_prod = qmse_tmp.inner_product_error(X, Y) - \
                       np.mean(qmse_tmp.inner_product_error(X, Y))

        # MSE
        qmse = TurboQuantMSE(D, b, seed=SEED)
        err_mse = qmse.inner_product_error(X, Y)

        for row, (err, label) in enumerate([(err_prod, "prod"), (err_mse, "mse")]):
            ax = axes[row][col]
            ax.hist(err, bins=60, color="steelblue" if label == "prod" else "coral",
                    alpha=0.75, density=True)
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.axvline(np.mean(err), color="red", lw=1.5,
                       label=f"mean={np.mean(err):.4f}")
            ax.set_title(f"b={b}" if row == 0 else "", fontsize=10)
            ax.set_xlabel("IP error", fontsize=8)
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(
                    "TurboQuant$_\\mathrm{prod}$" if label == "prod"
                    else "TurboQuant$_\\mathrm{mse}$", fontsize=8)
        if verbose:
            print(f"  b={b}: prod_mean={np.mean(err_prod):.5f}  "
                  f"mse_mean={np.mean(err_mse):.5f}")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig1_bias_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def run_fig2(verbose: bool = True):
    """IP error variance vs average inner-product at b=2."""
    os.makedirs(OUTDIR, exist_ok=True)

    # Create Y vectors with controlled average inner product
    # by mixing a fixed direction with noise
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(D)
    base /= np.linalg.norm(base)

    target_ips = [0.01, 0.05, 0.10, 0.17]
    n_per_bucket = 8_000
    b = 2

    qmse  = TurboQuantMSE(D, b, seed=SEED)
    qprod = TurboQuantProd(D, b, seed=SEED)

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharey=False)
    fig.suptitle("Fig 2: IP error vs average inner product at b=2\n"
                 "TurboQuant$_\\mathrm{prod}$ (top, constant variance) vs "
                 "TurboQuant$_\\mathrm{mse}$ (bottom, bias grows)", fontsize=11)

    for col, target in enumerate(target_ips):
        # Build X vectors with average IP ≈ target with base direction
        # X = α·base + noise,  normalised
        alpha = target  # rough approximation
        X = alpha * base[None, :] + rng.standard_normal((n_per_bucket, D)) * 0.1
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y = np.tile(base, (n_per_bucket, 1))   # query = base direction

        avg_ip = float(np.mean(np.sum(X * Y, axis=1)))

        err_prod = qprod.inner_product_error(X, Y)
        err_mse  = qmse.inner_product_error(X, Y)

        for row, (err, label, color) in enumerate([
            (err_prod, "prod", "steelblue"),
            (err_mse,  "mse",  "coral"),
        ]):
            ax = axes[row][col]
            ax.hist(err, bins=60, color=color, alpha=0.75, density=True)
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.axvline(np.mean(err), color="red", lw=1.5,
                       label=f"mean={np.mean(err):.4f}")
            ax.set_title(f"Avg IP={avg_ip:.2f}" if row == 0 else "", fontsize=9)
            ax.set_xlabel("IP error", fontsize=8)
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(
                    "TurboQuant$_\\mathrm{prod}$" if label == "prod"
                    else "TurboQuant$_\\mathrm{mse}$", fontsize=8)

        if verbose:
            print(f"  avg_ip≈{avg_ip:.2f}: prod_mean={np.mean(err_prod):.5f}  "
                  f"mse_mean={np.mean(err_mse):.5f}")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig2_bias_vs_ip.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


if __name__ == "__main__":
    print("=== Benchmark: Bias comparison (Fig 1 & 2) ===")
    print("\n-- Fig 1: Error histograms --")
    run_fig1()
    print("\n-- Fig 2: Bias vs average IP --")
    run_fig2()
