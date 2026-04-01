"""
run_all.py
----------
Master script: runs all four benchmarks in sequence and prints a summary.

Usage:
    python run_all.py               # all benchmarks
    python run_all.py --skip-recall # skip GloVe download
    python run_all.py --only timing # only run timing benchmark
"""

import argparse
import sys
import os
import time

os.makedirs("plots", exist_ok=True)

BENCHMARKS = {
    "distortion": ("Distortion vs theoretical bounds (Fig. 3)",
                   "benchmarks.bench_distortion"),
    "bias":       ("Bias comparison MSE vs prod (Fig. 1-2)",
                   "benchmarks.bench_bias"),
    "recall":     ("Recall@k – GloVe d=200 (Fig. 5, downloads ~800 MB)",
                   "benchmarks.bench_recall"),
    "recall-syn": ("Recall@k – synthetic vectors (no download)",
                   "benchmarks.bench_recall_synthetic"),
    "timing":     ("Indexing time vs PQ (Table 2)",
                   "benchmarks.bench_timing"),
}


def main():
    parser = argparse.ArgumentParser(description="Run TurboQuant benchmarks")
    parser.add_argument("--only",       help="Run only this benchmark", default=None)
    parser.add_argument("--skip-recall",action="store_true",
                        help="Skip the GloVe download/recall benchmark")
    args = parser.parse_args()

    to_run = list(BENCHMARKS.keys())
    if args.only:
        if args.only not in BENCHMARKS:
            print(f"Unknown benchmark '{args.only}'. "
                  f"Choose from: {list(BENCHMARKS)}")
            sys.exit(1)
        to_run = [args.only]
    if args.skip_recall and "recall" in to_run:
        to_run.remove("recall")

    print("=" * 60)
    print("  TurboQuant benchmark suite")
    print("  Paper: arXiv:2504.19874")
    print("=" * 60)
    print()

    summary = []
    for name in to_run:
        title, mod_path = BENCHMARKS[name]
        print(f"{'─'*60}")
        print(f"  [{name.upper()}] {title}")
        print(f"{'─'*60}")
        t0 = time.perf_counter()
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            mod.run(verbose=True)
            elapsed = time.perf_counter() - t0
            summary.append((name, "✓", f"{elapsed:.1f}s"))
            print(f"\n  Completed in {elapsed:.1f}s\n")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            summary.append((name, "✗", str(e)))
            print(f"\n  FAILED: {e}\n")

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, status, info in summary:
        print(f"  {status} {name:12s}  {info}")
    print()
    print("  Plots saved to: ./plots/")
    print("    fig1_bias_histograms.png")
    print("    fig2_bias_vs_ip.png")
    print("    fig3_distortion_bounds.png")
    print("    fig5_recall_glove.png       (if recall ran)")
    print("    fig5_recall_synthetic.png  (if recall-syn ran)")
    print("    table2_indexing_time.png")


if __name__ == "__main__":
    main()
