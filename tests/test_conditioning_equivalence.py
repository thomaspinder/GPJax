"""Equivalence tests across independently-derived conditioning code paths.

Five call sites in the library derive the conjugate conditioning algebra
independently (predict, MLL, LOOCV, collapsed ELBO, variational predicts).
These tests pin the mathematical identities that must hold across them, so
that consolidating the derivations cannot silently change behaviour.
"""

from itertools import pairwise

import gpjax as gpx
import jax.numpy as jnp

# Jitter values swept when pinning the scaling of the collapsed-bound / MLL
# gap. Five decades, stopping at 1e-8: the normalised gap is still flat to
# within 0.1% at 1e-12, so 1e-8 sits seven orders above the float64 noise
# floor of this fixture.
_JITTER_SWEEP = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)

# Ceiling on the leading coefficient of that gap, i.e. on gap / jitter. Two
# in-flight states of this branch measured the coefficient at 84.5 and at 186
# (see test_collapsed_elbo_mll_gap_scales_with_jitter), so a pin that hugs
# either single observation is brittle; 500 keeps ~2.7x headroom over the
# larger one while still bounding the gap at the 1e-6 default by 5e-4, which
# is 1.4e-4 relative to the objective value.
_GAP_COEFFICIENT_CEILING = 500.0

# Permitted spread of gap / jitter across the sweep. Measured spread is 1.02
# in this checkout and 1.69 in the other state referenced above, so 4.0 keeps
# ~2.4x headroom over the worse observation.
_GAP_COEFFICIENT_SPREAD = 4.0


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


def _collapsed_elbo_mll_gap(jitter: float) -> float:
    """Absolute gap between the collapsed bound and the exact MLL at ``jitter``.

    Inducing inputs are placed on the training inputs, so the two quantities
    are the same object up to where each side adds ``Prior.jitter``.

    Args:
        jitter: The value of ``Prior.jitter`` to build the model with.

    Returns:
        float: ``|collapsed_elbo - conjugate_mll|``.
    """
    model, data = _make(jitter=jitter)
    q = gpx.variational_families.CollapsedVariationalGaussian(
        model=model, inducing_inputs=data.X
    )
    elbo = gpx.objectives.collapsed_elbo(q, data)
    mll = gpx.objectives.conjugate_mll(model, data)
    return float(jnp.abs(elbo - mll))


def test_collapsed_elbo_equals_mll_when_z_is_x():
    """Point pin on the gap at the default jitter.

    The identity is exact only in the jitter -> 0 limit: ``Prior.jitter``
    enters Kzz on the bound's side and Sigma on the MLL's, so an
    O(n * jitter / (2 sigma^2)) discrepancy is expected even at 1e-6. The
    tolerance is stated as ``_GAP_COEFFICIENT_CEILING * jitter`` rather than
    as a bare atol so that it comes from the same measured coefficient as the
    scaling test below. The measured gap here is 8.45e-05 against a 5e-04
    budget (17%); the previous 2e-04 budget was 93% consumed in another state
    of this branch, where the coefficient measures 186 rather than 84.5.
    """
    jitter = 1e-6
    gap = _collapsed_elbo_mll_gap(jitter)
    assert gap < _GAP_COEFFICIENT_CEILING * jitter, f"gap {gap:.3e} at {jitter:.0e}"


def test_collapsed_elbo_mll_gap_scales_with_jitter():
    """The collapsed-bound / MLL gap must be first order in the jitter.

    Formerly a strict xfail at a non-default jitter: the family carried its
    own jitter knob, so the two sides factorised different matrices and the
    gap acquired a component that did not vanish with ``Prior.jitter``. With
    the knob unified the gap is the intrinsic O(jitter) slack of the Titsias
    bound, and *that* is the property worth pinning — a point tolerance on a
    quantity of order 6 cannot discriminate below a few percent at jitter
    1e-3, because the true discrepancy there really is ~1e-1.

    Measured on this fixture (jitter, gap, gap / jitter)::

        1e-08   8.449e-07   84.49
        1e-07   8.449e-06   84.49
        1e-06   8.448e-05   84.48
        1e-05   8.439e-04   84.39
        1e-04   8.415e-03   84.15
        1e-03   8.308e-02   83.08

    The normalised gap is flat to 1.02x across five decades here, and to
    1.69x (186 down to 110 over 1e-6 to 1e-3) in the other in-flight state of
    this branch, which is mildly sub-linear because higher-order terms in the
    jitter grow at the top of the range. The band is therefore taken from
    those measurements, not from the exact linearity theory predicts: 4.0 on
    the spread and 500 on the coefficient itself. That still detects an
    additive error of ~2.5e-06 on a quantity of order 3.5 — five orders
    tighter than the atol=2e-1 this test used to carry — because at the
    bottom of the sweep the genuine gap is only 8.4e-07.
    """
    gaps = [_collapsed_elbo_mll_gap(jitter) for jitter in _JITTER_SWEEP]
    table = "\n".join(
        f"  jitter={jitter:.0e}  gap={gap:.4e}  gap/jitter={gap / jitter:9.3f}"
        for jitter, gap in zip(_JITTER_SWEEP, gaps, strict=True)
    )

    # Shrinking the jitter must shrink the gap: a component that does not
    # vanish with the jitter shows up here as a plateau at the bottom.
    for smaller, larger in pairwise(gaps):
        assert smaller < larger, f"gap is not monotone in the jitter:\n{table}"

    # ... and it must shrink proportionally, not merely eventually.
    coefficients = [
        gap / jitter for jitter, gap in zip(_JITTER_SWEEP, gaps, strict=True)
    ]
    spread = max(coefficients) / min(coefficients)
    assert spread < _GAP_COEFFICIENT_SPREAD, (
        f"gap / jitter varies by {spread:.2f}x across the sweep, so the gap is "
        f"not O(jitter):\n{table}"
    )
    assert max(coefficients) < _GAP_COEFFICIENT_CEILING, (
        f"gap / jitter reaches {max(coefficients):.1f}, above the measured "
        f"ceiling {_GAP_COEFFICIENT_CEILING:.0f}:\n{table}"
    )


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
        model=posterior, inducing_inputs=z
    )
    q_plain = gpx.variational_families.VariationalGaussian(
        model=posterior, inducing_inputs=z
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
