from flax import nnx
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import (
    seed,
    trace,
)

from gpjax.numpyro_extras import register_parameters
from gpjax.parameters import (
    PositiveReal,
    Real,
)


class MockSubModule(nnx.Module):
    def __init__(self):
        self.c = Real(jnp.array(3.0))


class MockModel(nnx.Module):
    def __init__(self):
        self.a = PositiveReal(jnp.array(1.0))
        self.b = Real(jnp.array(2.0))
        self.submodule = MockSubModule()


def test_register_parameters_default_priors():
    model = MockModel()

    def model_fn():
        return register_parameters(model)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    # Check sites exist
    assert "a" in tr
    assert "b" in tr
    assert "submodule.c" in tr

    # Check distributions
    # a: PositiveReal -> LogNormal
    # LogNormal is a TransformedDistribution.
    assert isinstance(tr["a"]["fn"], dist.LogNormal)

    # b: Real -> Normal
    # If scalar, to_event(0) returns Normal. If vector, to_event(1) returns Independent.
    if isinstance(tr["b"]["fn"], dist.Independent):
        assert isinstance(tr["b"]["fn"].base_dist, dist.Normal)
    else:
        assert isinstance(tr["b"]["fn"], dist.Normal)

    # submodule.c: Real -> Normal
    if isinstance(tr["submodule.c"]["fn"], dist.Independent):
        assert isinstance(tr["submodule.c"]["fn"].base_dist, dist.Normal)
    else:
        assert isinstance(tr["submodule.c"]["fn"], dist.Normal)

    # Check values in updated model
    with seed(rng_seed=0):
        updated_model = model_fn()

    assert jnp.allclose(updated_model.a.value, tr["a"]["value"])
    assert jnp.allclose(updated_model.b.value, tr["b"]["value"])
    assert jnp.allclose(updated_model.submodule.c.value, tr["submodule.c"]["value"])

    # Verify original values were different (random sample != 1.0)
    assert not jnp.allclose(updated_model.a.value, 1.0)


def test_register_parameters_custom_priors():
    model = MockModel()

    priors = {"a": dist.Gamma(2.0, 2.0), "submodule.c": dist.Cauchy(0.0, 1.0)}

    def model_fn():
        return register_parameters(model, priors=priors)

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    assert isinstance(tr["a"]["fn"], dist.Gamma)
    # b should use default (Normal wrapped in Independent or Normal)
    if isinstance(tr["b"]["fn"], dist.Independent):
        assert isinstance(tr["b"]["fn"].base_dist, dist.Normal)
    else:
        assert isinstance(tr["b"]["fn"], dist.Normal)
    assert isinstance(tr["submodule.c"]["fn"], dist.Cauchy)


def test_register_parameters_prefix():
    model = MockModel()

    def model_fn():
        return register_parameters(model, prefix="foo")

    with seed(rng_seed=0):
        tr = trace(model_fn).get_trace()

    assert "foo.a" in tr
    assert "foo.b" in tr
    assert "foo.submodule.c" in tr
