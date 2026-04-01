"""
benchmarks/bench_recall.py
--------------------------
Reproduces Fig. 5 from the paper: Recall@1@k for nearest-neighbour search.

Downloads GloVe 6B 200-dim vectors (~800 MB, one-time).
Compares TurboQuant vs Product Quantization at 2-bit and 4-bit.

Usage:
    python -m benchmarks.bench_recall
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turboquant import TurboQuantMSE, TurboQuantProd, ProductQuantizer

# ── Config ────────────────────────────────────────────────────────────────
GLOVE_URL  = ("https://nlp.stanford.edu/data/glove.6B.zip")
GLOVE_DIR  = os.path.expanduser("~/.cache/turboquant")
GLOVE_TXT  = os.path.join(GLOVE_DIR, "glove.6B.200d.txt")
GLOVE_NPY  = os.path.join(GLOVE_DIR, "glove_200d.npy")

N_TRAIN    = 100_000
N_QUERY    =   1_000
TOP_K_LIST = [1, 2, 4, 8, 16, 32, 64]
BITWIDTHS  = [2, 4]
SEED       = 42
OUTDIR     = "plots"
# ──────────────────────────────────────────────────────────────────────────


def _load_or_download_glove() -> np.ndarray:
    """Download and cache GloVe 6B 200-dim vectors as a numpy array."""
    os.makedirs(GLOVE_DIR, exist_ok=True)

    if os.path.exists(GLOVE_NPY):
        print("  Loading cached GloVe embeddings...")
        return np.load(GLOVE_NPY)

    if not os.path.exists(GLOVE_TXT):
        zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
        if not os.path.exists(zip_path):
            print("  Downloading GloVe 6B (~822 MB) – one-time only...")
            print(f"  URL: {GLOVE_URL}")
            print("  This may take a few minutes on a slow connection.")

            def _progress(block, block_size, total):
                done = block * block_size
                pct  = min(100, done * 100 // total) if total > 0 else 0
                print(f"\r  Progress: {pct:3d}%  ({done//1_000_000} MB)", end="")

            urllib.request.urlretrieve(GLOVE_URL, zip_path, _progress)
            print()

        print("  Extracting glove.6B.200d.txt ...")
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("glove.6B.200d.txt", GLOVE_DIR)

    print("  Parsing GloVe text file (first time only) ...")
    vectors = []
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            vectors.append([float(x) for x in parts[1:]])
    arr = np.array(vectors, dtype=np.float32)
    np.save(GLOVE_NPY, arr)
    print(f"  Cached {arr.shape[0]:,} vectors of dim {arr.shape[1]}")
    return arr


def _unit(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-15)


def _recall_at_k(
    query: np.ndarray,        # (n_q, d)
    db:    np.ndarray,        # (n_db, d) – original
    db_hat: np.ndarray,       # (n_db, d) – quantized
    k: int
) -> float:
    """
    Recall@1@k: fraction of queries where the true top-1 neighbour
    (by exact IP) appears in the top-k approximated neighbours.
    """
    # True top-1 by exact inner product
    true_scores = query @ db.T              # (n_q, n_db)
    true_top1   = np.argmax(true_scores, axis=1)   # (n_q,)

    # Approximate top-k by quantized inner product
    approx_scores = query @ db_hat.T       # (n_q, n_db)
    approx_topk   = np.argpartition(-approx_scores, k, axis=1)[:, :k]

    hits = sum(
        true_top1[i] in approx_topk[i]
        for i in range(len(true_top1))
    )
    return hits / len(true_top1)


def run(verbose: bool = True) -> dict:
    os.makedirs(OUTDIR, exist_ok=True)

    vecs = _load_or_download_glove()

    # Subsample and normalise
    rng  = np.random.default_rng(SEED)
    idx  = rng.permutation(len(vecs))
    db_raw    = _unit(vecs[idx[:N_TRAIN]].astype(np.float64))
    query_raw = _unit(vecs[idx[N_TRAIN:N_TRAIN + N_QUERY]].astype(np.float64))
    D = db_raw.shape[1]

    if verbose:
        print(f"  Dataset: GloVe d={D}  db={N_TRAIN:,}  query={N_QUERY:,}")

    results = {}  # key: (method, b) -> list of recall values

    for b in BITWIDTHS:
        if verbose:
            print(f"\n  --- b={b} bits ---")

        # ── TurboQuant ────────────────────────────────────────────────
        t0 = time.perf_counter()
        qtq = TurboQuantMSE(D, b, seed=SEED)
        db_hat_tq = qtq.quant_dequant(db_raw)
        tq_time = time.perf_counter() - t0
        if verbose:
            print(f"  TurboQuant index time: {tq_time:.4f}s")

        recalls_tq = []
        for k in TOP_K_LIST:
            r = _recall_at_k(query_raw, db_raw, db_hat_tq, k)
            recalls_tq.append(r)
        results[("TurboQuant", b)] = recalls_tq
        if verbose:
            print(f"  TurboQuant Recall@1@1 = {recalls_tq[0]:.4f}  "
                  f"@64 = {recalls_tq[-1]:.4f}")

        # ── Product Quantization ──────────────────────────────────────
        t0 = time.perf_counter()
        cap = min(D // 4, 16)
        M = max(m for m in range(1, cap + 1) if D % m == 0)
        pq = ProductQuantizer(D, b, M=M, seed=SEED)
        pq.fit(db_raw)
        db_hat_pq = pq.quant_dequant(db_raw)
        pq_time = time.perf_counter() - t0
        if verbose:
            print(f"  PQ index time: {pq_time:.4f}s  (M={M})")

        recalls_pq = []
        for k in TOP_K_LIST:
            r = _recall_at_k(query_raw, db_raw, db_hat_pq, k)
            recalls_pq.append(r)
        results[("PQ", b)] = recalls_pq
        if verbose:
            print(f"  PQ         Recall@1@1 = {recalls_pq[0]:.4f}  "
                  f"@64 = {recalls_pq[-1]:.4f}")

    _plot(results)
    return results


def _plot(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title(f"Recall@1@k – GloVe d=200  "
                 f"(db={N_TRAIN:,}, query={N_QUERY:,})", fontsize=11)

    styles = {
        ("TurboQuant", 2): ("b-o",  "TurboQuant 2-bit"),
        ("TurboQuant", 4): ("b--s", "TurboQuant 4-bit"),
        ("PQ",         2): ("r-o",  "PQ 2-bit"),
        ("PQ",         4): ("r--s", "PQ 4-bit"),
    }
    for key, (fmt, label) in styles.items():
        if key in results:
            ax.plot(TOP_K_LIST, results[key], fmt, lw=2, ms=6, label=label)

    ax.set_xlabel("Top-k", fontsize=11)
    ax.set_ylabel("Recall@1@k", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TOP_K_LIST)
    ax.set_xticklabels(TOP_K_LIST)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTDIR, "fig5_recall_glove.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {path}")
    plt.close()


if __name__ == "__main__":
    print("=== Benchmark: Recall@k Nearest-Neighbour Search (GloVe d=200) ===")
    run()
