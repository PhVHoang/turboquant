"""
TurboQuant – Python implementation of arXiv:2504.19874
======================================================

Modules
-------
turboquant.quantizer   : TurboQuantMSE, TurboQuantProd
turboquant.bounds      : theoretical upper/lower distortion bounds
turboquant.pq_baseline : ProductQuantizer (comparison baseline)
benchmarks.*           : runnable benchmark scripts
"""

from .quantizer import TurboQuantMSE, TurboQuantProd
from .bounds import (mse_upper_bound, mse_lower_bound,
                     prod_upper_bound, prod_lower_bound)
from .pq_baseline import ProductQuantizer

__all__ = [
    "TurboQuantMSE", "TurboQuantProd",
    "mse_upper_bound", "mse_lower_bound",
    "prod_upper_bound", "prod_lower_bound",
    "ProductQuantizer",
]
