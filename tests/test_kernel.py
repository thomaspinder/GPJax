import pytest
import jax.numpy as jnp
from gpjax.kernels import RBF, Matern12, Matern32, Matern52
all_kernels = [RBF, Matern12, Matern32, Matern52]


@pytest.mark.parametrize('kernel', all_kernels)
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_shape(kernel, dim):
    x = jnp.linspace(-1., 1., num=10).reshape(-1, 1)
    if dim > 1:
        x = jnp.hstack([x] * dim)
    kern = kernel()
    gram = kern(x, x)
    assert gram.shape[0] == x.shape[0]
    assert gram.shape[0] == gram.shape[1]


@pytest.mark.parametrize('kernel', all_kernels)
@pytest.mark.parametrize('ell, sigma', [(0.2, 0.2), (0.5, 0.1), (0.1, 0.5),
                                        (0.5, 0.5)])
@pytest.mark.parametrize('n', [2, 10, 100])
def test_pos_def(kernel, ell, sigma, n):
    x = jnp.linspace(0., 1., num=n).reshape(-1, 1)
    kern = kernel(lengthscale=jnp.array([ell]), variance=jnp.array(sigma))
    gram = kern(x, x)
    Inn = jnp.eye(n) * 1e-6
    stable_gram = gram + Inn
    print(stable_gram[0])
    min_eig = jnp.linalg.eigvals(stable_gram).min()
    print(min_eig)
    assert min_eig > 0
