import jax.numpy as jnp
import paramax
from paramax import AbstractUnwrappable


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


def test_lower_triangular_positive_diagonal():
    """LowerTriangular enforces positive diagonal via softplus."""
    from gpjax.parameters import LowerTriangular

    L = jnp.array([[2.0, 0.0], [0.5, 3.0]])
    p = LowerTriangular(L)
    val = p.unwrap()
    assert jnp.allclose(val, L, atol=1e-5)
    assert val[0, 0] > 0
    assert val[1, 1] > 0


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
