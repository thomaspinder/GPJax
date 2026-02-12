from flax import nnx
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    FillTriangularTransform,
    LowerTriangular,
    NonNegativeReal,
    Parameter,
    PositiveReal,
    Real,
    SigmoidBounded,
    transform,
)
from hypothesis import (
    given,
    strategies as st,
)
from hypothesis.extra.numpy import arrays
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest


def valid_shapes(min_dims=0, max_dims=2):
    return st.integers(min_dims, max_dims).flatmap(
        lambda d: st.lists(st.integers(1, 5), min_size=d, max_size=d).map(tuple)
    )


def real_arrays(shape_strategy=valid_shapes(), min_value=None, max_value=None):
    return arrays(
        dtype=np.float64,
        shape=shape_strategy,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ).map(jnp.array)


@given(value=real_arrays())
def test_real_parameter(value):
    # Should accept any real value
    p = Real(value)
    assert jnp.array_equal(p[...], value)
    assert jnp.array_equal(p[...], value)
    assert p.tag == "real"


@given(value=real_arrays(min_value=1e-6, max_value=1e6))
def test_positive_real_valid(value):
    p = PositiveReal(value)
    assert jnp.array_equal(p[...], value)
    assert p.tag == "positive"


@given(value=real_arrays(max_value=-1e-6))
def test_positive_real_invalid(value):
    with pytest.raises(ValueError):
        PositiveReal(value)


@given(value=real_arrays(min_value=0.0, max_value=1e6))
def test_non_negative_real_valid(value):
    p = NonNegativeReal(value)
    assert jnp.array_equal(p[...], value)
    assert p.tag == "non_negative"


@given(value=real_arrays(max_value=-1e-6))
def test_non_negative_real_invalid(value):
    with pytest.raises(ValueError):
        NonNegativeReal(value)


@given(value=real_arrays(min_value=0.0, max_value=1.0))
def test_sigmoid_bounded_valid(value):
    p = SigmoidBounded(value)
    assert jnp.array_equal(p[...], value)
    assert p.tag == "sigmoid"


@given(value=real_arrays(min_value=1.001, max_value=1e6))
def test_sigmoid_bounded_invalid_high(value):
    with pytest.raises(ValueError):
        SigmoidBounded(value)


@given(value=real_arrays(max_value=-0.001))
def test_sigmoid_bounded_invalid_low(value):
    with pytest.raises(ValueError):
        SigmoidBounded(value)


@given(
    param_class=st.sampled_from([NonNegativeReal, PositiveReal, Real, SigmoidBounded]),
    data=st.data(),
)
def test_transform_roundtrip(param_class, data):
    # Generate valid value for the parameter type
    if param_class == NonNegativeReal:
        val = data.draw(real_arrays(min_value=0.0, max_value=10.0))
    elif param_class == PositiveReal:
        val = data.draw(real_arrays(min_value=1e-3, max_value=10.0))
    elif param_class == Real:
        val = data.draw(real_arrays(min_value=-10.0, max_value=10.0))
    elif param_class == SigmoidBounded:
        val = data.draw(real_arrays(min_value=1e-3, max_value=1.0 - 1e-3))
    else:
        return  # Should not happen

    params = nnx.State({"p": param_class(val)})

    # Forward
    t_params = transform(params, DEFAULT_BIJECTION, inverse=False)

    # Inverse
    inv_params = transform(t_params, DEFAULT_BIJECTION, inverse=True)

    # Check both access patterns
    assert jnp.allclose(inv_params["p"][...], val, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(inv_params["p"][...], val, atol=1e-5, rtol=1e-5)


# Strategy for lower triangular matrices
def lower_triangular_matrices(n_min=1, n_max=5):
    return st.integers(n_min, n_max).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-10, max_value=10, width=64),
        ).map(lambda x: jnp.tril(jnp.array(x)))
    )


@given(value=lower_triangular_matrices())
def test_lower_triangular_valid(value):
    p = LowerTriangular(value)
    assert jnp.array_equal(p[...], value)
    assert p.tag == "lower_triangular"


@given(
    n=st.integers(2, 5),
    data=st.data(),
)
def test_lower_triangular_invalid(n, data):
    # Generate a square matrix
    mat = data.draw(
        arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-10, max_value=10, width=64),
        ).map(jnp.array)
    )
    # Ensure it's NOT lower triangular by setting an upper element
    row, col = np.triu_indices(n, 1)
    if len(row) > 0:
        # Pick a random upper triangular index
        idx = data.draw(st.integers(0, len(row) - 1))
        r, c = row[idx], col[idx]
        # Set to non-zero
        mat = mat.at[r, c].set(1.0)

        with pytest.raises(ValueError):
            LowerTriangular(mat)


@given(n=st.integers(1, 10))
def test_fill_triangular_shapes(n):
    k = n * (n + 1) // 2
    vec = jnp.zeros(k)
    ft = FillTriangularTransform()

    mat = ft(vec)
    assert mat.shape == (n, n)
    assert jnp.allclose(mat, jnp.tril(mat))


@given(n=st.integers(1, 5), data=st.data())
def test_fill_triangular_roundtrip_hypothesis(n, data):
    k = n * (n + 1) // 2
    vec = data.draw(
        arrays(
            dtype=np.float64,
            shape=(k,),
            elements=st.floats(min_value=-5.0, max_value=5.0, width=64),
        ).map(jnp.array)
    )

    ft = FillTriangularTransform()

    # Forward
    mat = ft(vec)
    assert mat.shape == (n, n)

    # Inverse
    vec_recon = ft.inv(mat)

    assert jnp.allclose(vec, vec_recon)


def test_fill_triangular_errors():
    ft = FillTriangularTransform()
    # A vector of length 2 is invalid: no integer n satisfies n(n+1)/2 = 2
    with pytest.raises(ValueError):
        ft(jnp.zeros(2))


@pytest.mark.parametrize(
    "param_cls, value",
    [
        (PositiveReal, jnp.array(1.0)),
        (PositiveReal, jnp.array([1.0, 2.0])),
        (Real, jnp.array(1.0)),
        (NonNegativeReal, jnp.array(1.0)),
    ],
)
def test_parameter_construction_under_grad(param_cls, value):
    """Regression test for #592: parameter construction must accept JAX tracers."""

    def f(x):
        return param_cls(x)[...].sum()

    grad = jax.grad(f)(value)
    assert grad.shape == value.shape


class TestParameterPriorStorage:
    """Regression tests: prior must be stored via NNX metadata, not instance attrs."""

    def test_parameter_with_prior_constructs(self):
        """Bug 1: Parameter.__init__ must not use direct setattr for numpyro_properties."""
        p = PositiveReal(1.0, prior=dist.LogNormal(0.0, 1.0))
        assert isinstance(p, Parameter)

    def test_parameter_without_prior_constructs(self):
        p = PositiveReal(1.0)
        assert isinstance(p, Parameter)

    def test_prior_accessible_after_construction(self):
        prior = dist.LogNormal(0.0, 1.0)
        p = PositiveReal(1.0, prior=prior)
        numpyro_props = getattr(p, "numpyro_properties", {})
        assert numpyro_props.get("prior") is prior

    def test_no_prior_gives_empty_numpyro_properties(self):
        p = PositiveReal(1.0)
        numpyro_props = getattr(p, "numpyro_properties", {})
        assert numpyro_props.get("prior") is None

    def test_tag_property_works(self):
        """Bug 2: Parameter.tag must not use removed .metadata property."""
        p = PositiveReal(1.0)
        assert p.tag == "positive"

    def test_tag_property_all_types(self):
        assert Real(0.0).tag == "real"
        assert PositiveReal(1.0).tag == "positive"
        assert NonNegativeReal(0.0).tag == "non_negative"
        assert SigmoidBounded(0.5).tag == "sigmoid"
        assert LowerTriangular(jnp.eye(2)).tag == "lower_triangular"

    def test_prior_survives_split_merge(self):
        """Prior metadata must survive nnx.split / nnx.merge cycle."""

        class M(nnx.Module):
            def __init__(self):
                self.ls = PositiveReal(1.0, prior=dist.LogNormal(0.0, 1.0))

        m = M()
        graphdef, state = nnx.split(m)
        m2 = nnx.merge(graphdef, state)
        numpyro_props = getattr(m2.ls, "numpyro_properties", {})
        assert isinstance(numpyro_props.get("prior"), dist.LogNormal)

    def test_prior_survives_replace(self):
        """Prior metadata must survive Variable.replace()."""
        prior = dist.LogNormal(0.0, 1.0)
        p = PositiveReal(1.0, prior=prior)
        p2 = p.replace(jnp.array(2.0))
        numpyro_props = getattr(p2, "numpyro_properties", {})
        assert numpyro_props.get("prior") is prior


class TestCoregionalizationMatrix:
    def test_init_shape(self):
        """W is [P, R] and kappa is [P]."""
        from gpjax.parameters import CoregionalizationMatrix

        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=2, key=key)
        assert coreg.W[...].shape == (3, 2)
        assert coreg.kappa[...].shape == (3,)

    def test_B_shape_and_symmetry(self):
        """B is [P, P] and symmetric."""
        from gpjax.parameters import CoregionalizationMatrix

        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=2, key=key)
        B = coreg.B
        assert B.shape == (3, 3)
        assert jnp.allclose(B, B.T)

    def test_B_positive_semi_definite(self):
        """All eigenvalues of B are non-negative."""
        from gpjax.parameters import CoregionalizationMatrix

        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=4, rank=2, key=key)
        eigvals = jnp.linalg.eigvalsh(coreg.B)
        assert jnp.all(eigvals >= 0.0)

    def test_B_positive_definite_with_kappa(self):
        """B has strictly positive eigenvalues because kappa > 0."""
        from gpjax.parameters import CoregionalizationMatrix

        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=1, key=key)
        eigvals = jnp.linalg.eigvalsh(coreg.B)
        assert jnp.all(eigvals > 0.0)

    def test_rank_one(self):
        """Rank-1 W produces rank-1 WW^T; B has rank >= 1 from kappa."""
        from gpjax.parameters import CoregionalizationMatrix

        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=1, key=key)
        assert coreg.W[...].shape == (3, 1)
        assert coreg.B.shape == (3, 3)

    def test_full_rank(self):
        """Rank P coregionalization matrix."""
        from gpjax.parameters import CoregionalizationMatrix

        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=3, key=key)
        assert coreg.W[...].shape == (3, 3)
