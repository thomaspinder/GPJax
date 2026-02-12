"""Model-level abstractions for specialized GP inference.

This module contains model classes that orchestrate GPJax components
(kernels, likelihoods, priors) for specialized inference algorithms
that decompose or structure the problem differently than standard GP inference.

OILMM (Orthogonal Instantaneous Linear Mixing Model):
    Achieves O(n³m) complexity for multi-output GPs by using an orthogonal
    mixing matrix that decouples inference into M independent single-output
    GP problems.
"""

from gpjax.models.oilmm import (
    OILMMModel,
    OILMMPosterior,
    OrthogonalMixingMatrix,
    create_oilmm,
    create_oilmm_from_data,
    create_oilmm_with_kernels,
    oilmm_mll,
)

__all__ = [
    "OILMMModel",
    "OILMMPosterior",
    "OrthogonalMixingMatrix",
    "create_oilmm",
    "create_oilmm_from_data",
    "create_oilmm_with_kernels",
    "oilmm_mll",
]
