"""
turboquant/bounds.py
--------------------
Information-theoretic distortion bounds from the paper (Theorems 1, 2, 3).

Upper bounds (TurboQuant achieves):
  MSE:         Dmse  ≤ √(3π/2) · 4^{-b}
  Inner prod:  Dprod ≤ √(3π/2) · ‖y‖²/d · 4^{-b}

Lower bounds (any algorithm must exceed):
  MSE:         Dmse  ≥ 4^{-b}
  Inner prod:  Dprod ≥ ‖y‖²/d · 4^{-b}

Empirical distortion values from Table in Theorem 1 for b=1..4:
  Dmse ≈ 0.36, 0.117, 0.03, 0.009
"""

import numpy as np

# Constant factor between upper and lower bound
_C = np.sqrt(3 * np.pi / 2)   # ≈ 2.166... wait, paper says ≈2.7
# Actually √(3π/2) = √(4.712) ≈ 2.17, but they write √3π/2 = √3·π/2 ≈ 2.72
# Paper: √(3π)/2  (not √(3π/2))  → let's be precise
_UPPER_CONST = np.sqrt(3 * np.pi) / 2   # ≈ 2.72


def mse_upper_bound(b: float) -> float:
    """Upper bound on MSE: √(3π)/2 · 4^{-b}  (Theorem 1)."""
    return _UPPER_CONST * 4.0 ** (-b)


def mse_lower_bound(b: float) -> float:
    """Lower bound on MSE: 4^{-b}  (Theorem 3)."""
    return 4.0 ** (-b)


def prod_upper_bound(b: float, y_norm_sq: float = 1.0, d: int = 1) -> float:
    """Upper bound on inner-product distortion: √(3π)/2 · ‖y‖²/d · 4^{-b}."""
    return _UPPER_CONST * (y_norm_sq / d) * 4.0 ** (-b)


def prod_lower_bound(b: float, y_norm_sq: float = 1.0, d: int = 1) -> float:
    """Lower bound on inner-product distortion: ‖y‖²/d · 4^{-b}."""
    return (y_norm_sq / d) * 4.0 ** (-b)


# Fine-grained empirical values from Theorem 1 / Theorem 2 (paper)
MSE_EMPIRICAL = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
PROD_EMPIRICAL = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}  # divided by d
