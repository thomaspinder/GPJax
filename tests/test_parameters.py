import equinox as eqx
import gpjax as gpx
from gpjax.parameters import (
    NonNegativeReal,
    PositiveReal,
    Real,
    SigmoidBounded,
    collect_log_prior,
    val,
)
import jax
import jax.numpy as jnp
import numpyro.distributions as npd
import paramax
from paramax import AbstractUnwrappable
import pytest


def test_positive_real_unwraps_to_array():
    from gpjax.parameters import PositiveReal

    p = PositiveReal(jnp.array(2.0))
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert isinstance(val, jnp.ndarray)
    assert jnp.allclose(val, jnp.array(2.0), atol=1e-5)


def test_positive_real_preserves_positivity():
    from gpjax.parameters import PositiveReal

    p = PositiveReal(jnp.array(0.01))
    val = p.unwrap()
    assert val > 0


def test_non_negative_real_unwraps():
    from gpjax.parameters import NonNegativeReal

    p = NonNegativeReal(jnp.array(0.5))
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert jnp.allclose(val, jnp.array(0.5), atol=1e-5)


def test_real_unwraps_to_identity():
    from gpjax.parameters import Real

    p = Real(jnp.array(-3.0))
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert jnp.allclose(val, jnp.array(-3.0))


def test_sigmoid_bounded_unwraps_within_bounds():
    from gpjax.parameters import SigmoidBounded

    p = SigmoidBounded(jnp.array(0.5), low=0.0, high=1.0)
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert 0.0 <= val <= 1.0
    assert jnp.allclose(val, jnp.array(0.5), atol=1e-5)


def test_sigmoid_bounded_custom_bounds():
    from gpjax.parameters import SigmoidBounded

    p = SigmoidBounded(jnp.array(5.0), low=2.0, high=8.0)
    val = p.unwrap()
    assert 2.0 <= val <= 8.0
    assert jnp.allclose(val, jnp.array(5.0), atol=1e-5)


def test_lower_triangular_unwraps():
    from gpjax.parameters import LowerTriangular

    L = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    p = LowerTriangular(L)
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert jnp.allclose(val, L, atol=1e-5)


def test_paramax_unwrap_on_module():
    """unwrap() on an eqx.Module containing parameters produces plain arrays."""
    import equinox as eqx
    from gpjax.parameters import PositiveReal, Real

    class Dummy(eqx.Module):
        a: PositiveReal
        b: Real

    m = Dummy(a=PositiveReal(jnp.array(2.0)), b=Real(jnp.array(-1.0)))
    unwrapped = paramax.unwrap(m)
    assert isinstance(unwrapped.a, jnp.ndarray)
    assert isinstance(unwrapped.b, jnp.ndarray)
    assert jnp.allclose(unwrapped.a, jnp.array(2.0), atol=1e-5)
    assert jnp.allclose(unwrapped.b, jnp.array(-1.0))


def test_val_unwraps_nested_wrappers():
    """_val must resolve wrappers nested *inside* a parameter, not just the outer node.

    paramax.non_trainable wraps array leaves, so freezing a parameter yields e.g.
    PositiveReal(_unconstrained=NonTrainable(...)). A single .unwrap() call would
    feed the NonTrainable object straight into the softplus bijection of the
    PositiveReal.
    """
    from gpjax.parameters import (
        PositiveReal,
        val,
    )

    p = paramax.non_trainable(PositiveReal(jnp.array(2.0)))
    assert isinstance(p._unconstrained, paramax.NonTrainable)
    assert jnp.equal(val(p), jnp.array(2.0))


def test_lower_triangular_positive_diagonal():
    """LowerTriangular enforces positive diagonal via softplus."""
    from gpjax.parameters import LowerTriangular

    L = jnp.array([[2.0, 0.0], [0.5, 3.0]])
    p = LowerTriangular(L)
    val = p.unwrap()
    assert jnp.allclose(val, L, atol=1e-5)
    assert val[0, 0] > 0
    assert val[1, 1] > 0


@pytest.mark.parametrize(
    "factory, storage_field",
    [
        (
            lambda v: __import__(
                "gpjax.parameters", fromlist=["PositiveReal"]
            ).PositiveReal(v),
            "_unconstrained",
        ),
        (
            lambda v: __import__(
                "gpjax.parameters", fromlist=["NonNegativeReal"]
            ).NonNegativeReal(v),
            "_unconstrained",
        ),
        (lambda v: __import__("gpjax.parameters", fromlist=["Real"]).Real(v), "value"),
        (
            lambda v: __import__(
                "gpjax.parameters", fromlist=["SigmoidBounded"]
            ).SigmoidBounded(v, low=0.0, high=1.0),
            "_unconstrained",
        ),
    ],
)
def test_scalar_parameters_preserve_float32(factory, storage_field):
    """Regression: parameters should preserve float32 inputs (discussion #628)."""
    value = jnp.asarray(0.5, dtype=jnp.float32)
    p = factory(value)
    assert getattr(p, storage_field).dtype == jnp.float32
    assert p.unwrap().dtype == jnp.float32


def test_lower_triangular_preserves_float32():
    """Regression: LowerTriangular preserves float32 inputs (discussion #628).

    Relies on the local dtype-preserving variant of
    SoftplusLowerCholeskyTransform (temporary workaround for numpyro's
    untyped allocations in vec_to_tril_matrix).
    """
    from gpjax.parameters import LowerTriangular

    L = jnp.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
    p = LowerTriangular(L)
    assert p._flat.dtype == jnp.float32
    val = p.unwrap()
    assert val.dtype == jnp.float32
    assert jnp.allclose(val, L, atol=1e-5)


def test_coregionalization_matrix():
    """CoregionalizationMatrix produces a PSD matrix."""
    from gpjax.parameters import CoregionalizationMatrix
    import jax.random as jr

    cm = CoregionalizationMatrix(num_outputs=3, rank=2, key=jr.key(0))
    B = cm.B
    assert B.shape == (3, 3)
    # PSD check: all eigenvalues >= 0
    eigvals = jnp.linalg.eigvalsh(B)
    assert jnp.all(eigvals >= -1e-6)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

PARAMETER_TYPES = [
    (PositiveReal, 2.0),
    (NonNegativeReal, 2.0),
    (Real, -0.5),
    (SigmoidBounded, 0.5),
]


@pytest.mark.parametrize(("cls", "value"), PARAMETER_TYPES)
def test_prior_defaults_to_none(cls, value):
    assert cls(jnp.asarray(value)).prior is None


@pytest.mark.parametrize(("cls", "value"), PARAMETER_TYPES)
def test_prior_is_read_back_unchanged(cls, value):
    prior = npd.Normal(0.0, 1.0)
    assert cls(jnp.asarray(value), prior=prior).prior is prior


@pytest.mark.parametrize(("cls", "value"), PARAMETER_TYPES)
def test_every_parameter_type_contributes_a_log_prior(cls, value):
    prior = npd.Normal(0.0, 1.0)
    p = cls(jnp.asarray(value), prior=prior)
    assert jnp.equal(collect_log_prior(p), prior.log_prob(val(p)).sum())


def test_prior_hyperparameters_are_not_pytree_leaves():
    """The prior's own arrays must never reach the trainable partition."""
    prior = npd.LogNormal(jnp.array(1.2345), jnp.array(6.789))
    p = PositiveReal(jnp.array(1.0), prior=prior)

    leaves = jax.tree.leaves(p)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], p._unconstrained)

    params, _ = eqx.partition(p, eqx.is_array)
    assert not any(
        jnp.allclose(leaf, 1.2345) or jnp.allclose(leaf, 6.789)
        for leaf in jax.tree.leaves(params)
    )


def test_prior_survives_partition_and_combine():
    """``fit`` splits models this way, so the prior has to come back intact."""
    prior = npd.LogNormal(0.0, 1.0)
    p = PositiveReal(jnp.array(2.0), prior=prior)
    params, static = eqx.partition(p, eqx.is_array)
    assert eqx.combine(params, static).prior is prior


def test_prior_receives_no_gradient():
    """Differentiating the log-prior touches the parameter, never the prior."""
    prior = npd.LogNormal(jnp.array(0.0), jnp.array(1.0))
    p = PositiveReal(jnp.array(2.0), prior=prior)

    grads = jax.grad(collect_log_prior)(p)
    assert len(jax.tree.leaves(grads)) == 1
    assert grads.prior is prior


def test_collect_log_prior_uses_the_constrained_value():
    """A prior on a PositiveReal is evaluated at the positive value, not at the
    unconstrained value it is stored as."""
    prior = npd.LogNormal(0.0, 1.0)
    p = PositiveReal(jnp.array(2.0), prior=prior)

    assert jnp.equal(collect_log_prior(p), prior.log_prob(jnp.array(2.0)))
    assert not jnp.allclose(
        collect_log_prior(p), prior.log_prob(p._unconstrained), atol=1e-5
    )


def test_collect_log_prior_is_zero_without_priors():
    assert collect_log_prior(gpx.kernels.RBF()) == 0.0


def test_collect_log_prior_sums_over_the_model():
    lengthscale_prior = npd.LogNormal(0.0, 1.0)
    variance_prior = npd.LogNormal(jnp.log(2.0), 0.5)
    kernel = gpx.kernels.RBF(
        lengthscale=PositiveReal(1.5, prior=lengthscale_prior),
        variance=PositiveReal(0.5, prior=variance_prior),
    )
    expected = lengthscale_prior.log_prob(1.5) + variance_prior.log_prob(0.5)
    assert jnp.equal(collect_log_prior(kernel), expected)


def test_collect_log_prior_sums_a_batched_prior_per_dimension():
    """An ARD lengthscale takes a different prior in each dimension."""
    lengthscale = jnp.array([0.5, 1.0, 2.0])
    prior = npd.LogNormal(
        loc=jnp.array([0.0, 0.7, 1.5]), scale=jnp.array([1.0, 0.5, 0.25])
    )
    kernel = gpx.kernels.RBF(lengthscale=PositiveReal(lengthscale, prior=prior))
    assert jnp.equal(collect_log_prior(kernel), prior.log_prob(lengthscale).sum())


def test_collect_log_prior_is_jittable():
    prior = npd.LogNormal(0.0, 1.0)
    kernel = gpx.kernels.RBF(lengthscale=PositiveReal(2.0, prior=prior))
    assert jnp.equal(
        eqx.filter_jit(collect_log_prior)(kernel),
        collect_log_prior(kernel),
    )
