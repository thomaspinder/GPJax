"""Equivalence tests across independently-derived conditioning code paths.

Five call sites in the library derive the conjugate conditioning algebra
independently (predict, MLL, LOOCV, collapsed ELBO, variational predicts).
These tests pin the mathematical identities that must hold across them, so
that consolidating the derivations cannot silently change behaviour.
"""

import gpjax as gpx
import jax.numpy as jnp
import pytest


def _make(jitter=1e-6):
    x = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = jnp.sin(3.0 * x)
    data = gpx.Dataset(X=x, y=y)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(),
        kernel=gpx.kernels.RBF(),
        jitter=jitter,
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.2)
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
    reason="the variational family still carries its own jitter knob "
    "(q.jitter enters Kzz; the model's Prior.jitter enters Sigma), so "
    "collapsed_elbo and conjugate_mll factorise different matrices at "
    "non-default jitter. The model-side split is fixed (see "
    "test_predict_consistent_with_mll_nondefault_jitter); the family-side "
    "knob unifies in the variational universalisation PR, post-natgrads.",
)
def test_collapsed_elbo_equals_mll_when_z_is_x_nondefault_jitter():
    posterior, data = _make(jitter=1e-3)
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=posterior, inducing_inputs=data.X
    )
    elbo = gpx.objectives.collapsed_elbo(q, data)
    mll = gpx.objectives.conjugate_mll(posterior, data)
    assert jnp.allclose(elbo, mll, atol=2e-4)


def test_predict_consistent_with_mll_nondefault_jitter():
    """predict and the MLL must factorise the SAME matrix at any jitter.

    Before v1.0, ConjugatePosterior.predict applied ``self.jitter`` to the
    training gram while conjugate_mll applied ``prior.jitter`` — two
    independent knobs that could legally diverge. Both quantities are now
    views of one conditioned Posterior, so this closed-form check holds at
    a deliberately non-default jitter.
    """
    jitter = 1e-3
    model, data = _make(jitter=jitter)
    posterior = model.condition(data)

    kernel_gram = model.prior.kernel.gram(data.X).as_matrix()
    noise_variance = 0.2**2
    sigma = kernel_gram + (jitter + noise_variance) * jnp.eye(data.n)
    residual = data.y[:, 0] - model.prior.mean_function(data.X)[:, 0]

    quad = residual @ jnp.linalg.solve(sigma, residual)
    _, logdet = jnp.linalg.slogdet(sigma)
    expected_mll = -0.5 * (quad + logdet + data.n * jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(posterior.log_marginal_likelihood, expected_mll, atol=1e-8)

    cross = model.prior.kernel.cross_covariance(data.X, data.X)
    expected_mean = model.prior.mean_function(data.X)[:, 0] + cross.T @ (
        jnp.linalg.solve(sigma, residual)
    )
    predictive = posterior(data.X)
    assert jnp.allclose(predictive.mean, expected_mean, atol=1e-8)


def test_whitened_matches_unwhitened_at_matched_parameters():
    posterior, _ = _make()
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
