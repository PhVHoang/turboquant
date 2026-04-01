"""
benchmarks/bench_timing.py
--------------------------
Reproduces Table 2 from the paper: quantization/indexing time comparison.

Measures wall-clock time to quantize 100,000 vectors of varying dimension d,
for TurboQuant vs Product Quantization (both 4-bit).

Paper numbers (NVIDIA A100):
  d=200  : PQ=37s, TurboQuant=0.0007s
  d=1536 : PQ=240s, TurboQuant=0.0013s

On M2 CPU, absolute numbers will differ, but the ratio should hold.

Usage:
    python -m benchmarks.bench_timing
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turboquant import TurboQuantMSE, ProductQuantizer

# ── Config ────────────────────────────────────────────────────────────────
DIMENSIONS    = [128, 200, 512, 1024]   # paper uses 200, 1536, 3072
                                       # 1536+ is slow on CPU for PQ
N_VECTORS     = 100_000
BITWIDTH      = 4
N_REPS_TQ     = 5     # repeat TurboQuant timing (fast)
N_REPS_PQ     = 1     # PQ is slow; one rep is enough
SEED          = 42
OUTDIR        = "plots"
# ──────────────────────────────────────────────────────────────────────────


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def time_turboquant(X: np.ndarray, d: int, b: int, n_reps: int) -> float:
    """Time TurboQuant quantization (excluding one-time setup)."""
    qt = TurboQuantMSE(d, b, seed=SEED)
    # warm-up
    _ = qt.quant(X[:100])
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = qt.quant(X)
        times.append(time.perf_counter() - t0)
    return float(np.min(times))


def time_pq(X: np.ndarray, d: int, b: int) -> tuple[float, float]:
    """Time PQ: returns (train_time, quant_time)."""
    cap = min(d // 4, 16)
    M = max(m for m in range(1, cap + 1) if d % m == 0)
    pq = ProductQuantizer(d, b, M=M, seed=SEED)

    t0 = time.perf_counter()
    pq.fit(X)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pq.quant(X)
    quant_time = time.perf_counter() - t0

    return train_time, quant_time


def run(verbose: bool = True) -> dict:
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []
    results = {"dims": [], "tq": [], "pq_train": [], "pq_quant": [], "pq_total": []}

    for d in DIMENSIONS:
        X = _unit(rng.standard_normal((N_VECTORS, d)))

        if verbose:
            print(f"  d={d} ...", end=" ", flush=True)

        tq_time = time_turboquant(X, d, BITWIDTH, N_REPS_TQ)
        pq_train, pq_quant = time_pq(X, d, BITWIDTH)
        pq_total = pq_train + pq_quant

        results["dims"].append(d)
        results["tq"].append(tq_time)
        results["pq_train"].append(pq_train)
        results["pq_quant"].append(pq_quant)
        results["pq_total"].append(pq_total)

        speedup = pq_total / tq_time
        rows.append([d, f"{tq_time*1000:.2f} ms",
                     f"{pq_train:.2f} s",
                     f"{pq_quant*1000:.2f} ms",
                     f"{pq_total:.2f} s",
                     f"{speedup:.0f}×"])

        if verbose:
            print(f"TurboQuant={tq_time*1000:.2f}ms  "
                  f"PQ(train)={pq_train:.2f}s  speedup={speedup:.0f}×")

    headers = ["d", "TurboQuant", "PQ train", "PQ quant", "PQ total", "Speedup"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    _plot(results)
    return results


def _plot(results: dict):
    dims    = results["dims"]
    tq      = [t * 1000 for t in results["tq"]]         # ms
    pq_tot  = [t * 1000 for t in results["pq_total"]]   # ms

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Indexing time: TurboQuant vs PQ  "
                 f"(4-bit, n={N_VECTORS:,})", fontsize=12)

    # Left: absolute times
    ax = axes[0]
    x = np.arange(len(dims))
    w = 0.35
    ax.bar(x - w/2, tq,     width=w, label="TurboQuant", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, pq_tot, width=w, label="PQ (train+quant)", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in dims])
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Absolute time (ms)", fontsize=11)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Right: log scale to see TurboQuant bar
    ax = axes[1]
    ax.bar(x - w/2, tq,     width=w, label="TurboQuant", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, pq_tot, width=w, label="PQ (train+quant)", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in dims])
    ax.set_ylabel("Time (ms, log scale)", fontsize=11)
    ax.set_title("Log scale", fontsize=11)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "table2_indexing_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {path}")
    plt.close()


if __name__ == "__main__":
    print("=== Benchmark: Indexing Time – TurboQuant vs PQ ===")
    print(f"  n={N_VECTORS:,} vectors, b={BITWIDTH}-bit, dims={DIMENSIONS}")
    run()
