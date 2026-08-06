"""Closed-form oracles for the conjugate GP quantities.

These tests pin the *values* of ``conjugate_mll``, ``ConjugatePosterior.predict``
and ``conjugate_loocv`` on a tiny fixed dataset, computed through an independent
linear-algebra path (direct ``jnp.linalg.solve``/``slogdet``, never
``GaussianDistribution``). Everything downstream in the test suite — the Kalman
MLL, ``collapsed_elbo`` — is validated against ``conjugate_mll``, so these
oracles are the ground truth the rest of the suite stands on.
"""

import gpjax as gpx
import jax.numpy as jnp

JITTER = 1e-6
OBS_STDDEV = 0.3
LENGTHSCALE = 0.7
VARIANCE = 1.3

X = jnp.array([[0.0], [0.45], [1.1]])
Y = jnp.array([[0.2], [-0.1], [0.6]])
XTEST = jnp.array([[0.2], [0.85]])


def _rbf(x1, x2):
    sq_dists = (x1[:, None, 0] - x2[None, :, 0]) ** 2
    return VARIANCE * jnp.exp(-0.5 * sq_dists / LENGTHSCALE**2)


def _posterior():
    kernel = gpx.kernels.RBF(lengthscale=LENGTHSCALE, variance=VARIANCE)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(), kernel=kernel, jitter=JITTER
    )
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=X.shape[0], obs_stddev=OBS_STDDEV
    )
    return prior * likelihood


def _sigma():
    n = X.shape[0]
    return _rbf(X, X) + (JITTER + OBS_STDDEV**2) * jnp.eye(n)


def test_conjugate_mll_matches_closed_form():
    posterior = _posterior()
    data = gpx.Dataset(X=X, y=Y)

    sigma = _sigma()
    n = X.shape[0]
    quad = Y[:, 0] @ jnp.linalg.solve(sigma, Y[:, 0])
    _, logdet = jnp.linalg.slogdet(sigma)
    expected = -0.5 * (quad + logdet + n * jnp.log(2.0 * jnp.pi))

    actual = gpx.objectives.conjugate_mll(posterior, data)
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_predict_matches_closed_form():
    posterior = _posterior()
    data = gpx.Dataset(X=X, y=Y)

    sigma = _sigma()
    kxt = _rbf(X, XTEST)
    ktt = _rbf(XTEST, XTEST)
    sigma_inv_y = jnp.linalg.solve(sigma, Y[:, 0])
    expected_mean = kxt.T @ sigma_inv_y
    expected_cov = ktt - kxt.T @ jnp.linalg.solve(sigma, kxt) + JITTER * jnp.eye(2)

    predictive = posterior.predict(XTEST, data)
    assert jnp.allclose(predictive.mean, expected_mean, atol=1e-8)
    assert jnp.allclose(predictive.covariance_matrix, expected_cov, atol=1e-8)


def test_loocv_matches_closed_form():
    posterior = _posterior()
    data = gpx.Dataset(X=X, y=Y)

    sigma = _sigma()
    sigma_inv = jnp.linalg.inv(sigma)
    resid = Y[:, 0]
    diag = jnp.diag(sigma_inv)
    loo_means = resid - sigma_inv @ resid / diag
    loo_vars = 1.0 / diag
    expected = jnp.sum(
        -0.5 * (jnp.log(2 * jnp.pi * loo_vars) + (resid - loo_means) ** 2 / loo_vars)
    )

    actual = gpx.objectives.conjugate_loocv(posterior, data)
    assert jnp.allclose(actual, expected, atol=1e-8)
