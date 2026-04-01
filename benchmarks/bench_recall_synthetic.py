"""
benchmarks/bench_recall_synthetic.py
-------------------------------------
Recall@1@k benchmark on synthetic unit-norm vectors.
Runs instantly with no downloads — good for quick validation.
Uses the same methodology as bench_recall.py (GloVe version).

Usage:
    python -m benchmarks.bench_recall_synthetic
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turboquant import TurboQuantMSE, ProductQuantizer

# ── Config ────────────────────────────────────────────────────────────────
DIMS       = [64, 200, 512]        # multiple dims like Fig. 5 of paper
N_TRAIN    = 100_000
N_QUERY    = 500
TOP_K_LIST = [1, 2, 4, 8, 16, 32, 64]
BITWIDTHS  = [2, 4]
SEED       = 42
OUTDIR     = "plots"
# ──────────────────────────────────────────────────────────────────────────


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _recall_at_k(query, db, db_hat, k):
    true_top1    = np.argmax(query @ db.T, axis=1)
    approx_topk  = np.argpartition(-(query @ db_hat.T), k, axis=1)[:, :k]
    hits = sum(true_top1[i] in approx_topk[i] for i in range(len(true_top1)))
    return hits / len(true_top1)


def run(verbose: bool = True) -> dict:
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    all_results = {}   # dim -> {(method, b): [recall@k]}
    timing_rows = []

    for D in DIMS:
        if verbose:
            print(f"\n  === d={D} ===")

        db    = _unit(rng.standard_normal((N_TRAIN, D)))
        query = _unit(rng.standard_normal((N_QUERY, D)))
        results = {}

        for b in BITWIDTHS:
            # ── TurboQuant ────────────────────────────────────────────
            t0 = time.perf_counter()
            q  = TurboQuantMSE(D, b, seed=SEED)
            db_hat_tq = q.quant_dequant(db)
            tq_time = time.perf_counter() - t0

            rc_tq = [_recall_at_k(query, db, db_hat_tq, k) for k in TOP_K_LIST]
            results[("TurboQuant", b)] = rc_tq
            if verbose:
                print(f"  TurboQuant b={b}: time={tq_time*1000:.1f}ms  "
                      f"R@1={rc_tq[0]:.3f}  R@64={rc_tq[-1]:.3f}")

            # ── PQ ───────────────────────────────────────────────────
            cap = min(D // 4, 16)
            M   = max(m for m in range(1, cap + 1) if D % m == 0)
            t0  = time.perf_counter()
            pq  = ProductQuantizer(D, b, M=M, seed=SEED)
            pq.fit(db)
            db_hat_pq = pq.quant_dequant(db)
            pq_time = time.perf_counter() - t0

            rc_pq = [_recall_at_k(query, db, db_hat_pq, k) for k in TOP_K_LIST]
            results[("PQ", b)] = rc_pq
            speedup = pq_time / tq_time
            timing_rows.append([D, b, f"{tq_time*1000:.1f} ms",
                                 f"{pq_time:.2f} s", f"{speedup:.0f}×"])
            if verbose:
                print(f"  PQ         b={b}: time={pq_time:.2f}s  "
                      f"R@1={rc_pq[0]:.3f}  R@64={rc_pq[-1]:.3f}  "
                      f"speedup={speedup:.0f}×")

        all_results[D] = results

    _plot(all_results)
    _print_timing(timing_rows)
    return all_results


def _plot(all_results: dict):
    n_dims = len(DIMS)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4.5), sharey=True)
    if n_dims == 1:
        axes = [axes]

    styles = {
        ("TurboQuant", 2): ("b-o",  "TurboQuant 2-bit"),
        ("TurboQuant", 4): ("b--s", "TurboQuant 4-bit"),
        ("PQ",         2): ("r-o",  "PQ 2-bit"),
        ("PQ",         4): ("r--s", "PQ 4-bit"),
    }

    for ax, D in zip(axes, DIMS):
        results = all_results[D]
        for key, (fmt, label) in styles.items():
            if key in results:
                ax.plot(TOP_K_LIST, results[key], fmt, lw=2, ms=6, label=label)
        ax.set_title(f"d={D}", fontsize=11)
        ax.set_xlabel("Top-k", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TOP_K_LIST)
        ax.set_xticklabels(TOP_K_LIST)
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Recall@1@k", fontsize=11)
    fig.suptitle(f"Recall@1@k – synthetic unit-norm vectors "
                 f"(n={N_TRAIN:,}, q={N_QUERY})", fontsize=11)
    plt.tight_layout()

    path = os.path.join(OUTDIR, "fig5_recall_synthetic.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {path}")
    plt.close()


def _print_timing(rows):
    try:
        from tabulate import tabulate
        headers = ["d", "b", "TurboQuant", "PQ total", "Speedup"]
        print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        for r in rows:
            print("  ", r)


if __name__ == "__main__":
    print("=== Benchmark: Recall@k (synthetic vectors) ===")
    print(f"  dims={DIMS}  n_train={N_TRAIN:,}  n_query={N_QUERY}")
    run()
