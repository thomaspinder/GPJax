from gpjax.kernels.approximations import RFF
from gpjax.kernels.nonstationary import (
    Linear,
    Polynomial,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
    StationaryKernel,
)
from gpjax.linalg.operators import Dense
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

config.update("jax_enable_x64", True)
_jitter = 1e-6


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10])
@pytest.mark.parametrize("n_dims", [1, 3])
def test_frequency_sampler(
    kernel: type[StationaryKernel], num_basis_fns: int, n_dims: int
):
    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel=base_kernel, num_basis_fns=num_basis_fns)
    assert approximate.frequencies.shape == (num_basis_fns, n_dims)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10])
@pytest.mark.parametrize("n_dims", [1, 3])
@pytest.mark.parametrize("n_data", [50, 100])
def test_gram(
    kernel: type[StationaryKernel], num_basis_fns: int, n_dims: int, n_data: int
):
    key = jr.key(123)
    x = jr.uniform(key, shape=(n_data, 1), minval=-3.0, maxval=3.0).reshape(-1, 1)
    if n_dims > 1:
        x = jnp.hstack([x] * n_dims)
    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel=base_kernel, num_basis_fns=num_basis_fns)

    linop = approximate.gram(x)

    # Check the return type
    assert isinstance(linop, Dense)

    Kxx = linop.to_dense() + jnp.eye(n_data) * _jitter

    # Check that the shape is correct
    assert Kxx.shape == (n_data, n_data)

    # Check that the Gram matrix is PSD
    evals, _ = jnp.linalg.eigh(Kxx)
    assert jnp.all(evals > 0)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("num_basis_fns", [2, 10])
@pytest.mark.parametrize("n_dims", [1, 3])
@pytest.mark.parametrize("n_datas", [(50, 100), (100, 50)])
def test_cross_covariance(
    kernel: type[StationaryKernel],
    num_basis_fns: int,
    n_dims: int,
    n_datas: tuple[int, int],
):
    nd1, nd2 = n_datas
    key = jr.key(123)
    x1 = jr.uniform(key, shape=(nd1, 1), minval=-3.0, maxval=3.0)
    if n_dims > 1:
        x1 = jnp.hstack([x1] * n_dims)
    x2 = jr.uniform(key, shape=(nd2, 1), minval=-3.0, maxval=3.0)
    if n_dims > 1:
        x2 = jnp.hstack([x2] * n_dims)

    base_kernel = kernel(active_dims=list(range(n_dims)))
    approximate = RFF(base_kernel=base_kernel, num_basis_fns=num_basis_fns)
    Kxx = approximate.cross_covariance(x1, x2)

    # Check the return type
    assert isinstance(Kxx, jax.Array)

    # Check that the shape is correct
    assert Kxx.shape == (nd1, nd2)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
@pytest.mark.parametrize("n_dim", [1, 3])
def test_improvement(kernel: type[StationaryKernel], n_dim: int):
    n_data = 100
    key = jr.key(123)

    x = jr.uniform(key, minval=-3.0, maxval=3.0, shape=(n_data, n_dim))
    base_kernel = kernel(active_dims=list(range(n_dim)))
    exact_linop = base_kernel.gram(x).to_dense()

    crude_approximation = RFF(base_kernel=base_kernel, num_basis_fns=10)
    c_linop = crude_approximation.gram(x).to_dense()

    better_approximation = RFF(base_kernel=base_kernel, num_basis_fns=100)
    b_linop = better_approximation.gram(x).to_dense()

    c_delta = jnp.linalg.norm(exact_linop - c_linop, ord="fro")
    b_delta = jnp.linalg.norm(exact_linop - b_linop, ord="fro")

    # The frobenius norm of the difference between the exact and approximate
    # should improve as we increase the number of basis functions
    assert c_delta > b_delta


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_exactness(kernel: type[StationaryKernel]):
    kernel = kernel(n_dims=1)

    n_data = 100
    key = jr.key(123)

    x = jr.uniform(key, minval=-3.0, maxval=3.0, shape=(n_data, 1))
    exact_linop = kernel.gram(x).to_dense()

    better_approximation = RFF(base_kernel=kernel, num_basis_fns=300)
    b_linop = better_approximation.gram(x).to_dense()

    max_delta = jnp.max(exact_linop - b_linop)
    assert max_delta < 0.1


@pytest.mark.parametrize("kernel", [Polynomial, Linear])
def test_nonstationary_raises_error(kernel):
    with pytest.raises(TypeError):
        RFF(base_kernel=kernel(1), num_basis_fns=10)


@pytest.mark.parametrize(
    "kernel",
    [RationalQuadratic, PoweredExponential, Periodic],
)
def test_missing_spectral_density_raises_error(kernel):
    with pytest.raises(NotImplementedError):
        RFF(base_kernel=kernel(), num_basis_fns=10)


@pytest.mark.parametrize("kernel", [RBF, Matern12, Matern32, Matern52])
def test_stochastic_init(kernel: type[StationaryKernel]):
    with pytest.raises(ValueError):
        # n_dims is not specified, but should be
        RFF(base_kernel=kernel(), num_basis_fns=10)

    kernel = kernel(n_dims=1)

    k1 = RFF(base_kernel=kernel, num_basis_fns=10, key=jr.key(123))
    k2 = RFF(base_kernel=kernel, num_basis_fns=10, key=jr.key(42))

    assert (k1.frequencies != k2.frequencies).any()
