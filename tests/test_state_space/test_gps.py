"""Tests for StateSpacePrior and StateSpaceConjugatePosterior."""

import gpjax as gpx
from gpjax.distributions import GaussianDistribution
from gpjax.state_space.gps import StateSpaceConjugatePosterior, StateSpacePrior
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest


def test_state_space_prior_predict_diagonal_returns_gaussian_distribution():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=2.0),
    )
    Xtest = jnp.linspace(0.0, 5.0, 7).reshape(-1, 1)
    dist = prior.predict(Xtest)
    assert isinstance(dist, GaussianDistribution)
    assert isinstance(dist.scale, lx.DiagonalLinearOperator)
    # Stationary marginal variance for Matern32 is the kernel's `variance` parameter.
    expected_variance = 2.0 + prior.jitter
    np.testing.assert_allclose(
        np.asarray(lx.diagonal(dist.scale)), expected_variance, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(dist.mean), 0.0, atol=1e-12)


def test_state_space_prior_predict_dense_raises():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
    )
    Xtest = jnp.array([[0.0], [1.0]])
    with pytest.raises(NotImplementedError, match=r"diagonal|dense"):
        prior.predict(Xtest, return_covariance_type="dense")


def test_state_space_prior_jitter_is_added_to_marginal_variance():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        jitter=1e-3,
    )
    dist = prior.predict(jnp.array([[0.5]]))
    expected_variance = 1.0 + 1e-3
    np.testing.assert_allclose(
        np.asarray(lx.diagonal(dist.scale)), expected_variance, atol=1e-12
    )


def test_state_space_prior_predict_with_constant_mean_function():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Constant(),
        kernel=gpx.kernels.Matern52(lengthscale=2.0, variance=0.7),
    )
    Xtest = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    dist = prior.predict(Xtest)
    expected_mean = prior.mean_function(Xtest).squeeze(-1)
    np.testing.assert_allclose(
        np.asarray(dist.mean), np.asarray(expected_mean), atol=1e-12
    )


def test_state_space_conjugate_posterior_construction():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=10, obs_stddev=0.2)
    posterior = StateSpaceConjugatePosterior(prior=prior, likelihood=likelihood)
    assert posterior.prior is prior
    assert posterior.likelihood is likelihood


def test_state_space_conjugate_posterior_predict_dense_raises():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=5, obs_stddev=0.1)
    posterior = StateSpaceConjugatePosterior(prior=prior, likelihood=likelihood)

    train_X = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    train_y = jnp.zeros((5, 1))
    train_data = gpx.Dataset(X=train_X, y=train_y)
    Xtest = jnp.array([[0.5]])
    with pytest.raises(NotImplementedError, match=r"diagonal|dense"):
        posterior.predict(Xtest, train_data, return_covariance_type="dense")


def test_state_space_conjugate_posterior_predict_smoothed_phase10_stub():
    """Phase 10 will fill in the smoothed prediction; for now it raises."""
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=5, obs_stddev=0.1)
    posterior = StateSpaceConjugatePosterior(prior=prior, likelihood=likelihood)
    train_data = gpx.Dataset(X=jnp.zeros((5, 1)), y=jnp.zeros((5, 1)))
    with pytest.raises(NotImplementedError, match=r"Phase 10|smooth"):
        posterior.predict(jnp.array([[0.5]]), train_data)


def test_state_space_conjugate_posterior_predict_filter_phase10_stub():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=5, obs_stddev=0.1)
    posterior = StateSpaceConjugatePosterior(prior=prior, likelihood=likelihood)
    train_data = gpx.Dataset(X=jnp.zeros((5, 1)), y=jnp.zeros((5, 1)))
    with pytest.raises(NotImplementedError, match=r"Phase 10|filter"):
        posterior.predict_filter(jnp.array([[0.5]]), train_data)


def test_state_space_prior_times_gaussian_returns_state_space_posterior():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=5, obs_stddev=0.2)
    posterior = prior * likelihood
    assert isinstance(posterior, StateSpaceConjugatePosterior)
    assert posterior.prior is prior


def test_state_space_prior_times_bernoulli_raises_type_error():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Bernoulli(num_datapoints=5)
    with pytest.raises(TypeError, match=r"Gaussian|conjugate"):
        prior * likelihood


def test_state_space_prior_times_multioutput_gaussian_raises():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.MultiOutputGaussian(num_datapoints=5, num_outputs=2)
    with pytest.raises(TypeError, match=r"single|MultiOutput|scalar"):
        prior * likelihood


def test_state_space_prior_times_gaussian_with_array_obs_stddev_raises():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=3, obs_stddev=jnp.ones(3))
    with pytest.raises(ValueError, match=r"scalar|obs_stddev"):
        prior * likelihood
