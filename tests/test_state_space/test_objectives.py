"""Tests for state_space_mll objective."""

import gpjax as gpx
from gpjax.state_space.gps import StateSpacePrior
from gpjax.state_space.objectives import state_space_mll
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_state_space.test_filter import _build_matern12_dataset


@pytest.mark.parametrize("jitter", [0.0, 1e-6])
def test_state_space_mll_matches_conjugate_mll_on_matern12(jitter):
    """state_space_mll must match the dense conjugate_mll for Matern12."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n = 50
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    # State-space prior + Gaussian likelihood.
    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
        jitter=jitter,
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=obs_stddev)
    ss_posterior = ss_prior * likelihood

    ss_mll = state_space_mll(ss_posterior, train_data)

    # Dense reference.
    dense_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
        jitter=jitter,
    )
    dense_posterior = dense_prior * likelihood
    dense_mll = gpx.objectives.conjugate_mll(dense_posterior, train_data)

    np.testing.assert_allclose(float(ss_mll), float(dense_mll), atol=1e-6, rtol=1e-8)


def test_state_space_mll_returns_scalar():
    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=10, obs_stddev=0.3)
    posterior = ss_prior * likelihood
    X, y = _build_matern12_dataset(n=10, lengthscale=1.0, variance=1.0, obs_stddev=0.3)
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    mll = state_space_mll(posterior, train_data)
    assert mll.ndim == 0


def test_state_space_mll_residualises_constant_mean():
    """With a learnable Constant mean, the filter must consume y - mean_function(X)."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n = 30
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    # Add a known constant offset to y to test residualisation.
    constant_offset = 2.5
    y_offset = y + constant_offset
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y_offset.reshape(-1, 1))

    # State-space prior with Constant mean function fitted to the offset.
    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Constant(constant=jnp.array(constant_offset)),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood
    ss_mll = state_space_mll(posterior, train_data)

    # Dense reference.
    dense_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(constant=jnp.array(constant_offset)),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    dense_posterior = dense_prior * likelihood
    dense_mll = gpx.objectives.conjugate_mll(dense_posterior, train_data)

    np.testing.assert_allclose(float(ss_mll), float(dense_mll), atol=1e-6, rtol=1e-8)


def test_state_space_mll_observation_mask_equivalent_to_dropping():
    """Masking observations must give the same MLL as dropping them entirely."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n = 25
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    mask_indices = [3, 11, 17]
    is_observed = jnp.ones(n, dtype=bool)
    for i in mask_indices:
        is_observed = is_observed.at[i].set(False)

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood

    masked_mll = state_space_mll(posterior, train_data, observation_mask=is_observed)

    # Dense reference: drop the masked points entirely.
    keep_mask = np.array(is_observed)
    X_kept = X[keep_mask]
    y_kept = y[keep_mask]
    train_data_kept = gpx.Dataset(X=X_kept.reshape(-1, 1), y=y_kept.reshape(-1, 1))
    likelihood_kept = gpx.likelihoods.Gaussian(
        num_datapoints=int(keep_mask.sum()), obs_stddev=obs_stddev
    )
    posterior_kept = ss_prior * likelihood_kept
    dropped_mll = state_space_mll(posterior_kept, train_data_kept)

    np.testing.assert_allclose(float(masked_mll), float(dropped_mll), atol=1e-9)


@pytest.mark.parametrize(
    "kernel_factory",
    [
        lambda: gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
        lambda: gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        lambda: gpx.kernels.Matern52(lengthscale=1.0, variance=1.0),
    ],
)
def test_state_space_mll_gradient_through_kernel_hyperparameters_is_finite(
    kernel_factory,
):
    """jax.grad of state_space_mll w.r.t. kernel hyperparameters must be finite."""
    lengthscale_value = 1.5
    variance_value = 0.8
    obs_stddev = 0.2
    n = 20
    X, y = _build_matern12_dataset(
        n=n,
        lengthscale=lengthscale_value,
        variance=variance_value,
        obs_stddev=obs_stddev,
    )
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    base_kernel = kernel_factory()

    def loss_through_lengthscale(lengthscale):
        kernel_class = type(base_kernel)
        kernel = kernel_class(lengthscale=lengthscale, variance=variance_value)
        prior = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=kernel,
        )
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=obs_stddev)
        posterior = prior * likelihood
        return -state_space_mll(posterior, train_data)

    grad_value = jax.grad(loss_through_lengthscale)(jnp.asarray(1.5))
    assert jnp.isfinite(grad_value), f"Non-finite gradient: {grad_value}"
    assert grad_value != 0.0, (
        "Zero gradient is suspicious — should depend on lengthscale"
    )


def test_state_space_mll_gradient_through_obs_stddev_is_finite():
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    n = 20
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))

    def loss(obs_stddev_value):
        prior = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
        )
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=n, obs_stddev=obs_stddev_value
        )
        posterior = prior * likelihood
        return -state_space_mll(posterior, train_data)

    grad_value = jax.grad(loss)(jnp.asarray(0.2))
    assert jnp.isfinite(grad_value)
    assert grad_value != 0.0
