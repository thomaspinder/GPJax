"""Linear algebra module for GPJax."""

from gpjax.linalg.custom_operators import BlockDiag, Kronecker
from gpjax.linalg.utils import (
    add_jitter,
    cholesky_factor,
    logdet,
    logdet_from_factor,
    stabilised_cholesky,
)

__all__ = [
    "BlockDiag",
    "Kronecker",
    "add_jitter",
    "cholesky_factor",
    "logdet",
    "logdet_from_factor",
    "stabilised_cholesky",
]
