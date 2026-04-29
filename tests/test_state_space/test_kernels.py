"""Tests for state-space kernels and the to_sde dispatcher."""

import gpjax as gpx
from gpjax.state_space.kernels import TruncatedPeriodic, to_sde
from gpjax.state_space.sde import (
    Matern12SDE,
    Matern32SDE,
    Matern52SDE,
    SumSDE,
    TruncatedPeriodicSDE,
)
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special


def test_truncated_periodic_call_matches_bessel_formula():
    lengthscale, variance, period, K = 0.4, 1.0, 1.0, 6
    kernel = TruncatedPeriodic(
        lengthscale=lengthscale,
        variance=variance,
        period=period,
        truncation_order=K,
    )
    taus = jnp.array([0.0, 0.1, 0.25, 0.5, 0.9])
    bessel_argument = 1.0 / (4.0 * lengthscale**2)
    expected = (
        variance
        * np.exp(-bessel_argument)
        * (
            scipy.special.iv(0, bessel_argument)
            + 2.0
            * sum(
                scipy.special.iv(k, bessel_argument)
                * np.cos(2 * np.pi * k * np.asarray(taus) / period)
                for k in range(1, K + 1)
            )
        )
    )
    actual = np.array([kernel(jnp.array([float(t)]), jnp.array([0.0])) for t in taus])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_truncated_periodic_accepts_parent_config_space():
    """The kernel itself does no 1-D-temporal enforcement; that is `to_sde`'s job."""
    kernel = TruncatedPeriodic(
        lengthscale=jnp.ones(3),
        variance=1.0,
        period=1.0,
        truncation_order=4,
        n_dims=3,
    )
    assert kernel.truncation_order == 4


def test_to_sde_matern12_returns_matern12sde():
    kernel = gpx.kernels.Matern12(lengthscale=2.0, variance=3.0)
    sde = to_sde(kernel)
    assert isinstance(sde, Matern12SDE)
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 2.0)
    np.testing.assert_allclose(np.asarray(sde.variance), 3.0)


def test_to_sde_default_handler_raises_for_unsupported_kernel():
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    with pytest.raises(
        NotImplementedError, match=r"Matern12|Matern32|Matern52|TruncatedPeriodic"
    ):
        to_sde(kernel)


def test_to_sde_matern32_returns_matern32sde():
    kernel = gpx.kernels.Matern32(lengthscale=1.5, variance=2.0)
    sde = to_sde(kernel)
    assert isinstance(sde, Matern32SDE)
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 1.5)
    np.testing.assert_allclose(np.asarray(sde.variance), 2.0)


def test_to_sde_matern52_returns_matern52sde():
    kernel = gpx.kernels.Matern52(lengthscale=0.7, variance=1.2)
    sde = to_sde(kernel)
    assert isinstance(sde, Matern52SDE)
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 0.7)
    np.testing.assert_allclose(np.asarray(sde.variance), 1.2)


def test_to_sde_truncated_periodic_returns_truncated_periodic_sde():
    kernel = TruncatedPeriodic(
        lengthscale=0.5, variance=2.5, period=3.0, truncation_order=4
    )
    sde = to_sde(kernel)
    assert isinstance(sde, TruncatedPeriodicSDE)
    assert sde.state_dim == 1 + 2 * 4
    np.testing.assert_allclose(np.asarray(sde.lengthscale), 0.5)
    np.testing.assert_allclose(np.asarray(sde.variance), 2.5)
    np.testing.assert_allclose(np.asarray(sde.period), 3.0)


def test_to_sde_sum_kernel_returns_sumsde():
    kernel = gpx.kernels.Matern32(lengthscale=1.0, variance=1.0) + TruncatedPeriodic(
        lengthscale=0.5,
        variance=1.0,
        period=1.0,
        truncation_order=4,
    )
    sde = to_sde(kernel)
    assert isinstance(sde, SumSDE)
    assert len(sde.components) == 2
    assert isinstance(sde.components[0], Matern32SDE)
    assert isinstance(sde.components[1], TruncatedPeriodicSDE)
    expected_state_dim = 2 + (1 + 2 * 4)
    assert sde.state_dim == expected_state_dim


@pytest.mark.parametrize(
    "kernel_factory,error_match",
    [
        (
            lambda: gpx.kernels.Periodic(lengthscale=1.0, variance=1.0),
            "TruncatedPeriodic",
        ),
        (
            lambda: gpx.kernels.RBF(lengthscale=1.0, variance=1.0),
            r"Matern12|Matern32|Matern52|TruncatedPeriodic",
        ),
        (
            lambda: gpx.kernels.Linear(),
            r"Matern12|Matern32|Matern52|TruncatedPeriodic|state-space conversion not implemented",
        ),
    ],
)
def test_to_sde_rejects_unsupported_kernels(kernel_factory, error_match):
    kernel = kernel_factory()
    with pytest.raises(NotImplementedError, match=error_match):
        to_sde(kernel)


def test_to_sde_rejects_product_kernel_with_kronecker_blowup_message():
    kernel = gpx.kernels.Matern32(lengthscale=1.0, variance=1.0) * gpx.kernels.Matern32(
        lengthscale=0.5, variance=1.0
    )
    with pytest.raises(NotImplementedError, match=r"Kronecker|product|state space"):
        to_sde(kernel)
