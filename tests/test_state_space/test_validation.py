"""Tests for state-space validation helpers."""

import gpjax as gpx
from gpjax.state_space._validation import _validate_temporal_kernel
import jax.numpy as jnp
import pytest


def test_validate_temporal_kernel_accepts_scalar_kernel():
    kernel = gpx.kernels.Matern12(lengthscale=1.0, variance=1.0)
    _validate_temporal_kernel(kernel)


def test_validate_temporal_kernel_rejects_vector_lengthscale():
    kernel = gpx.kernels.Matern12(lengthscale=jnp.ones(3), n_dims=3)
    with pytest.raises(ValueError, match=r"scalar lengthscale|1-D"):
        _validate_temporal_kernel(kernel)


def test_validate_temporal_kernel_rejects_active_dims_list():
    kernel = gpx.kernels.Matern12(active_dims=[0, 1], lengthscale=1.0, n_dims=2)
    with pytest.raises(ValueError, match=r"active_dims|1-D"):
        _validate_temporal_kernel(kernel)


def test_validate_temporal_kernel_rejects_n_dims_greater_than_one():
    kernel = gpx.kernels.Matern12(lengthscale=jnp.ones(2), n_dims=2)
    with pytest.raises(ValueError, match=r"n_dims|1-D"):
        _validate_temporal_kernel(kernel)
