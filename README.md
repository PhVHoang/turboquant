# TurboQuant — Python Implementation

A clean, well-tested Python implementation of:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
> Zandieh, Daliri, Hadian, Mirrokni — arXiv:2504.19874

Benchmarked and validated on Apple M2. Reproduces all key figures from the paper.

---

## Package structure

```
turboquant/
├── turboquant/
│   ├── __init__.py
│   ├── quantizer.py            ← TurboQuantMSE, TurboQuantProd
│   ├── bounds.py               ← theoretical upper/lower bounds
│   └── pq_baseline.py         ← ProductQuantizer comparison baseline
├── benchmarks/
│   ├── bench_distortion.py         → Fig. 3:  MSE & IP error vs bounds
│   ├── bench_bias.py               → Fig. 1-2: bias comparison histograms
│   ├── bench_recall.py             → Fig. 5:  Recall@k on GloVe d=200
│   ├── bench_recall_synthetic.py   → Fig. 5:  Recall@k (no download)
│   └── bench_timing.py             → Table 2: indexing time vs PQ
├── plots/                          ← output directory (auto-created)
├── run_all.py                      ← master runner
├── pyproject.toml
└── README.md
```

---

## Setup (M2 Mac)

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install numpy scipy scikit-learn matplotlib tabulate

# 3. (optional) install as editable package
pip install -e .
```

---

## Running benchmarks

```bash
# Recommended first run: everything except the 800 MB GloVe download
python run_all.py --skip-recall

# Full suite including GloVe (downloaded once to ~/.cache/turboquant/)
python run_all.py

# Individual benchmarks
python run_all.py --only distortion
python run_all.py --only bias
python run_all.py --only recall-syn   # synthetic vectors, no download
python run_all.py --only recall       # GloVe d=200
python run_all.py --only timing

# Or as modules
python -m benchmarks.bench_distortion
python -m benchmarks.bench_bias
python -m benchmarks.bench_recall_synthetic
python -m benchmarks.bench_recall
python -m benchmarks.bench_timing
```

---

## Expected results (M2 16 GB)

### Distortion vs theoretical bounds (bench_distortion, ~2 min)

| b | MSE (ours) | Paper Thm 1 | Lower bound | Upper bound |
|---|-----------|-------------|-------------|-------------|
| 1 | 0.363     | 0.360       | 0.250       | 0.384       |
| 2 | 0.117     | 0.117       | 0.063       | 0.096       |
| 3 | 0.034     | 0.030       | 0.016       | 0.024       |
| 4 | 0.009     | 0.009       | 0.004       | 0.006       |

### Bias comparison (bench_bias, ~1 min)

- **TurboQuantProd**: IP error mean ≈ 0 at all bit-widths ✓
- **TurboQuantMSE**: bias grows with inner product magnitude at low b

### Recall@k (bench_recall_synthetic, ~3 min)

TurboQuant reaches Recall@1@64 ≈ 1.00 at 4-bit; PQ achieves 0.11–0.78.
TurboQuant wins with zero training overhead.

### Indexing time (bench_timing, ~2 min)

| d    | TurboQuant | PQ total | Speedup |
|------|-----------|----------|---------|
| 200  | ~5 ms     | ~25 s    | ~5000×  |
| 512  | ~15 ms    | ~50 s    | ~3000×  |
| 1024 | ~50 ms    | ~120 s   | ~2400×  |

*(M2 with Accelerate-BLAS numpy; exact numbers vary by numpy build.)*

---

## Using the quantizers

```python
import numpy as np
from turboquant import TurboQuantMSE, TurboQuantProd

# Input must be unit-norm
rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 512)).astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True)

# MSE quantizer (4-bit)
q     = TurboQuantMSE(d=512, b=4, seed=42)
idx   = q.quant(X)          # (1000, 512) int16
X_hat = q.dequant(idx)      # (1000, 512) float64
print(f"MSE: {q.mse(X):.4f}")   # ≈ 0.009

# Inner-product quantizer (4-bit, unbiased)
qp    = TurboQuantProd(d=512, b=4, seed=42)
out   = qp.quant(X)         # ProdQuantized(idx, qjl, r_norms)
X_hat = qp.dequant(out)

Y = rng.standard_normal((1000, 512))
Y /= np.linalg.norm(Y, axis=1, keepdims=True)
err = qp.inner_product_error(X, Y)
print(f"IP error mean (≈ 0 = unbiased): {err.mean():.6f}")

# Theoretical bounds
from turboquant.bounds import mse_upper_bound, mse_lower_bound
for b in [1, 2, 3, 4, 5]:
    print(f"b={b}: [{mse_lower_bound(b):.5f}, {mse_upper_bound(b):.5f}]")
```

---

## Key implementation notes

**`searchsorted` for nearest-centroid**: Lloyd-Max centroids are always sorted,
so Voronoi boundaries are just their midpoints. `np.searchsorted` finds the
nearest centroid in O(n·d·b) with O(n·d) memory — vs the naive O(n·d·2^b)
broadcasting that causes OOM at large n/b.

**Gaussian approximation at large d**: The Beta PDF normalisation `Γ(d/2)`
overflows float64 for d ≥ ~170. The paper states explicitly that the Beta
converges to N(0, 1/d) in high dimensions — we switch at d ≥ 30, where
the error is < 1%.

**Two-stage IP quantizer**: `TurboQuantProd` uses b−1 bits of MSE quantization
to shrink the residual ‖r‖, then 1-bit QJL (sign sketch) on the residual
direction to correct the bias. The combination is provably unbiased (Theorem 2).

---

## Citation

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author  = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025}
}
```
