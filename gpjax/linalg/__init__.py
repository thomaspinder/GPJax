"""Linear algebra module for GPJax."""

from gpjax.linalg.custom_operators import BlockDiag, Kronecker
from gpjax.linalg.utils import add_jitter, cholesky_factor, logdet

__all__ = [
    "BlockDiag",
    "Kronecker",
    "add_jitter",
    "cholesky_factor",
    "logdet",
]
