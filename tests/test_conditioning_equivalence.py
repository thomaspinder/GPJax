"""Equivalence tests across independently-derived conditioning code paths.

Five call sites in the library derive the conjugate conditioning algebra
independently (predict, MLL, LOOCV, collapsed ELBO, variational predicts).
These tests pin the mathematical identities that must hold across them, so
that consolidating the derivations cannot silently change behaviour.
"""

import jax.numpy as jnp
import pytest

import gpjax as gpx


def _make(jitter=1e-6):
    x = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = jnp.sin(3.0 * x)
    data = gpx.Dataset(X=x, y=y)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(),
        kernel=gpx.kernels.RBF(),
        jitter=jitter,
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=0.2)
    return prior * likelihood, data


def test_collapsed_elbo_equals_mll_when_z_is_x():
    posterior, data = _make()
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=data.X
    )
    elbo = gpx.objectives.collapsed_elbo(q, data)
    mll = gpx.objectives.conjugate_mll(posterior, data)
    # The identity is exact only in the jitter -> 0 limit: the family's jitter
    # enters Kzz while the model's enters Sigma, so a small O(jitter/noise)
    # discrepancy is expected even when both knobs are 1e-6.
    assert jnp.allclose(elbo, mll, atol=2e-4)


@pytest.mark.xfail(
    strict=True,
    reason="jitter has two owners (Prior.jitter vs the variational family's "
    "jitter); collapsed_elbo and conjugate_mll factorise different matrices "
    "at non-default jitter. Resolved by the v1.0 conditioning stack.",
)
def test_collapsed_elbo_equals_mll_when_z_is_x_nondefault_jitter():
    posterior, data = _make(jitter=1e-3)
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=data.X
    )
    elbo = gpx.objectives.collapsed_elbo(q, data)
    mll = gpx.objectives.conjugate_mll(posterior, data)
    assert jnp.allclose(elbo, mll, atol=2e-4)


def test_whitened_matches_unwhitened_at_matched_parameters():
    posterior, data = _make()
    z = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)
    q_white = gpx.variational_families.WhitenedVariationalGaussian(
        posterior=posterior, inducing_inputs=z
    )
    q_plain = gpx.variational_families.VariationalGaussian(
        posterior=posterior, inducing_inputs=z
    )
    # At default parameters, whitened q(u) = N(0, I) and unwhitened
    # q(u) = N(0, I) describe different measures UNLESS the predictive
    # reduces identically; we instead match parameters explicitly:
    # unwhitened (mu, sqrt) = (m(z) + Lz mu_w, Lz sqrt_w).
    import equinox as eqx

    kernel = posterior.prior.kernel
    kzz = kernel.gram(z).as_matrix() + 1e-6 * jnp.eye(z.shape[0])
    lz = jnp.linalg.cholesky(kzz)
    mu_white = jnp.array([[0.3], [-0.2], [0.1], [0.4], [-0.5]])
    sqrt_white = 0.1 * jnp.eye(5) + 0.05 * jnp.tril(jnp.ones((5, 5)), k=-1)

    mean_z = posterior.prior.mean_function(z)
    mu_plain = mean_z + lz @ mu_white
    sqrt_plain = lz @ sqrt_white

    q_white = eqx.tree_at(
        lambda q: (q.variational_mean, q.variational_root_covariance),
        q_white,
        (
            type(q_white.variational_mean)(mu_white)
            if hasattr(type(q_white.variational_mean), "unwrap")
            else mu_white,
            type(q_white.variational_root_covariance)(sqrt_white)
            if hasattr(type(q_white.variational_root_covariance), "unwrap")
            else sqrt_white,
        ),
    )
    q_plain = eqx.tree_at(
        lambda q: (q.variational_mean, q.variational_root_covariance),
        q_plain,
        (
            type(q_plain.variational_mean)(mu_plain)
            if hasattr(type(q_plain.variational_mean), "unwrap")
            else mu_plain,
            type(q_plain.variational_root_covariance)(sqrt_plain)
            if hasattr(type(q_plain.variational_root_covariance), "unwrap")
            else sqrt_plain,
        ),
    )

    xtest = jnp.linspace(-0.2, 1.2, 7).reshape(-1, 1)
    dist_white = q_white(xtest)
    dist_plain = q_plain(xtest)
    assert jnp.allclose(dist_white.mean, dist_plain.mean, atol=1e-5)
    assert jnp.allclose(
        dist_white.covariance_matrix, dist_plain.covariance_matrix, atol=1e-5
    )
