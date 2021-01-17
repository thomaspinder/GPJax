import pytest
import jax.numpy as jnp
from gpjax.kernel import RBF


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_shape(dim):
    x = jnp.linspace(-1., 1., num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    kern = RBF()
    gram = kern(x, x)
    assert gram.shape[0] == x.shape[0]
    assert gram.shape[0] == gram.shape[1]


@pytest.mark.parametrize('ell, sigma', [(0.1, 0.1), (0.5, 0.1), (0.1, 0.5),
                                        (0.5, 0.5)])
def test_pos_def(ell, sigma):
    n = 100
    x = jnp.linspace(0., 1., num=n).reshape(-1, 1)
    kern = RBF()
    gram = kern(x, x)
    Inn = jnp.eye(n) * 1e-6
    stable_gram = gram + Inn
    min_eig = jnp.linalg.eigvals(stable_gram).min()
    assert min_eig > 0
