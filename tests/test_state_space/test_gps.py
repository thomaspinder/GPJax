"""Tests for StateSpacePrior and StateSpaceConjugateModel."""

import gpjax as gpx
from gpjax.distributions import GaussianDistribution
from gpjax.state_space.gps import StateSpaceConjugateModel, StateSpacePrior
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


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
def test_state_space_prior_predict_dense_matches_kernel_gram(kernel_class):
    """The prior has no data to condition on, so the dense joint is exactly
    the kernel's own gram — the SDE is an exact representation, not an
    approximation."""
    lengthscale, variance = 1.2, 0.9
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel_class(lengthscale=lengthscale, variance=variance),
        jitter=1e-8,
    )
    Xtest = jnp.array([[0.0], [1.0], [3.5], [3.5], [10.0]])
    dist = prior.predict(Xtest, covariance="dense")
    assert isinstance(dist, GaussianDistribution)
    assert isinstance(dist.scale, lx.MatrixLinearOperator)

    dense_kernel = kernel_class(lengthscale=lengthscale, variance=variance)
    expected_cov = dense_kernel.gram(Xtest).as_matrix() + 1e-8 * jnp.eye(5)
    np.testing.assert_allclose(
        np.asarray(dist.covariance_matrix), np.asarray(expected_cov), atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(dist.mean), 0.0, atol=1e-12)


def test_state_space_prior_predict_dense_matches_diagonal_on_the_diagonal():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=2.0),
        jitter=1e-6,
    )
    Xtest = jnp.linspace(0.0, 5.0, 7).reshape(-1, 1)
    dense_dist = prior.predict(Xtest, covariance="dense")
    diagonal_dist = prior.predict(Xtest, covariance="diagonal")
    np.testing.assert_allclose(
        np.asarray(dense_dist.variance), np.asarray(diagonal_dist.variance), atol=1e-12
    )


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
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.2)
    posterior = StateSpaceConjugateModel(prior=prior, likelihood=likelihood)
    assert posterior.prior is prior
    assert posterior.likelihood is likelihood


def test_state_space_conjugate_posterior_predict_dense_matches_condition_call():
    """``predict`` with ``covariance="dense"`` is sugar over ``condition(D)(t)``."""
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
        jitter=1e-6,
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
    posterior = StateSpaceConjugateModel(prior=prior, likelihood=likelihood)

    train_X = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    train_y = jnp.zeros((5, 1))
    train_data = gpx.Dataset(X=train_X, y=train_y)
    Xtest = jnp.array([[0.5], [0.8]])

    sugar = posterior.predict(Xtest, train_data, covariance="dense")
    explicit = posterior.condition(train_data)(Xtest, covariance="dense")

    assert isinstance(sugar.scale, lx.MatrixLinearOperator)
    np.testing.assert_array_equal(
        np.asarray(sugar.covariance_matrix), np.asarray(explicit.covariance_matrix)
    )
    np.testing.assert_array_equal(np.asarray(sugar.mean), np.asarray(explicit.mean))


def test_state_space_conjugate_posterior_predict_filter_dense_still_raises():
    """The causal predictive keeps rejecting ``covariance="dense"`` — see
    ``StateSpacePosterior.filtered``'s docstring for why."""
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
    posterior = StateSpaceConjugateModel(prior=prior, likelihood=likelihood)

    train_X = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    train_y = jnp.zeros((5, 1))
    train_data = gpx.Dataset(X=train_X, y=train_y)
    Xtest = jnp.array([[0.5]])
    with pytest.raises(NotImplementedError, match=r"diagonal|dense"):
        posterior.predict_filter(Xtest, train_data, covariance="dense")


def test_state_space_prior_times_gaussian_returns_state_space_posterior():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.2)
    posterior = prior * likelihood
    assert isinstance(posterior, StateSpaceConjugateModel)
    assert posterior.prior is prior


def test_state_space_prior_times_bernoulli_raises_type_error():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Bernoulli()
    with pytest.raises(TypeError, match=r"Gaussian|conjugate"):
        prior * likelihood


def test_state_space_prior_times_multioutput_gaussian_raises():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.MultiOutputGaussian(num_outputs=2)
    with pytest.raises(TypeError, match=r"single|MultiOutput|scalar"):
        prior * likelihood


def test_state_space_prior_times_gaussian_with_array_obs_stddev_raises():
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=jnp.ones(3))
    with pytest.raises(ValueError, match=r"scalar|obs_stddev"):
        prior * likelihood


def test_state_space_predict_dense_returns_finite_psd_covariance():
    data = gpx.Dataset(
        X=jnp.linspace(0, 5, 10).reshape(-1, 1),
        y=jnp.sin(jnp.linspace(0, 5, 10)).reshape(-1, 1),
    )
    posterior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
        jitter=1e-6,
    ) * gpx.likelihoods.Gaussian(obs_stddev=0.1)

    Xtest = jnp.linspace(0, 5, 4).reshape(-1, 1)
    dist = posterior.predict(Xtest, data, covariance="dense")

    cov = np.asarray(dist.covariance_matrix)
    assert cov.shape == (4, 4)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0.0)
