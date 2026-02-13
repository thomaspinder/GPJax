"""Additive kernel module."""

from gpjax.kernels.additive.decompose import predict_first_order, rank_first_order
from gpjax.kernels.additive.oak import OrthogonalAdditiveKernel
from gpjax.kernels.additive.sobol import sobol_indices
from gpjax.kernels.additive.transforms import (
    SinhArcsinhTransform,
    fit_all_normalising_flows,
    fit_normalising_flow,
)

__all__ = [
    "OrthogonalAdditiveKernel",
    "SinhArcsinhTransform",
    "fit_all_normalising_flows",
    "fit_normalising_flow",
    "predict_first_order",
    "rank_first_order",
    "sobol_indices",
]
