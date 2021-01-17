import pytest
import jax.numpy as jnp
from gpjax import ZeroMean


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_shape(dim):
    x = jnp.linspace(-1., 1., num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    meanf = ZeroMean()
    mu = meanf(x)
    assert mu.shape[0] == x.shape[0]
    assert mu.shape[1] == 1
