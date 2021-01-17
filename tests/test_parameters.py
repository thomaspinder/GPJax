import jax.numpy as jnp
from gpjax.parameters import Parameter
from gpjax.transforms import Softplus, Identity
from tensorflow_probability.substrates.jax import distributions as tfd
import pytest
from numpy.testing import assert_almost_equal
ZeroDist = tfd.Uniform(low=0., high=0.)


def hardcode_softplus(x: jnp.ndarray):
    return jnp.log(jnp.exp(x) - 1.0)


@pytest.mark.parametrize("val", [0.5, 1.0])
def test_transform(val):
    v = jnp.array([val])
    x = Parameter(v, transform=Softplus())
    assert x.untransform == v
    print(f"xval: {x.value}")
    print(f"hcode: {hardcode_softplus(v)}")
    assert x.value == hardcode_softplus(v)


@pytest.mark.parametrize('transform', [Identity(), Softplus()])
@pytest.mark.parametrize('val', [1e-4, 1., 5.])
@pytest.mark.parametrize(
    'dist',
    [ZeroDist, tfd.Normal(loc=0., scale=1.),
     tfd.Gamma(1.0, 1.0)])
def test_prior(transform, val, dist):
    p = Parameter(tensor=jnp.array([val]), transform=transform, prior=dist)
    truth = dist.log_prob(
        jnp.array([val])
    )  # Value must be passed as an array as TF defaults to Float32, whereas GPJax uses double precision
    assert_almost_equal(p.log_density, truth)
