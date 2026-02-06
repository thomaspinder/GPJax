from flax import nnx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.gps import Prior
from gpjax.likelihoods import Gaussian
from gpjax.objectives import (
    collapsed_elbo,
    conjugate_loocv,
    conjugate_mll,
    elbo,
    non_conjugate_mll,
)
from gpjax.parameters import Parameter
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def build_data(n_points: int, n_dims: int, key, binary: bool):
    x = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_points, n_dims))
    if binary:
        y = (
            0.5
            * jnp.sign(
                jnp.cos(
                    3 * x[:, 0].reshape(-1, 1)
                    + jr.normal(key, shape=(n_points, 1)) * 0.05
                )
            )
            + 0.5
        )
    else:
        y = (
            jnp.sin(x[:, 0]).reshape(-1, 1)
            + jr.normal(key=key, shape=(n_points, 1)) * 0.1
        )
    D = Dataset(X=x, y=y)
    return D


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_conjugate_mll(n_points: int, n_dims: int, key_val: int):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=False)

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_points)
    post = p * likelihood

    # test simple call
    res_simple = -conjugate_mll(post, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    graphdef, state, *states = nnx.split(post, Parameter, ...)

    def loss(params):
        posterior = nnx.merge(graphdef, params, *states)
        return -conjugate_mll(posterior, D)

    res_wrapped = loss(state)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(state)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    grad = jax.grad(loss)
    grad_res = grad(state)
    assert isinstance(grad_res, nnx.State)


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_conjugate_loocv(n_points, n_dims, key_val):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=False)

    # Build model
    p = Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = Gaussian(num_datapoints=n_points)
    post = p * likelihood

    # test simple call
    res_simple = -conjugate_loocv(post, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    graphdef, state, *states = nnx.split(post, Parameter, ...)

    def loss(params):
        posterior = nnx.merge(graphdef, params, *states)
        return -conjugate_loocv(posterior, D)

    res_wrapped = loss(state)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(state)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    loss_grad = jax.grad(loss)
    grad_res = loss_grad(state)
    assert isinstance(grad_res, nnx.State)


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_non_conjugate_mll(n_points, n_dims, key_val):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=True)

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Bernoulli(num_datapoints=n_points)
    post = p * likelihood

    # test simple call
    res_simple = -non_conjugate_mll(post, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    graphdef, state, *states = nnx.split(post, Parameter, ...)

    def loss(params):
        posterior = nnx.merge(graphdef, params, *states)
        return -non_conjugate_mll(posterior, D)

    res_wrapped = loss(state)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(state)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    loss_grad = jax.grad(loss)
    grad_res = loss_grad(state)
    assert isinstance(grad_res, nnx.State)


@pytest.mark.parametrize("n_points", [10, 20])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
def test_collapsed_elbo(n_points, n_dims, key_val):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=False)
    z = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_points // 2, n_dims))

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_points)
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p * likelihood, inducing_inputs=z
    )

    # test simple call
    res_simple = -collapsed_elbo(q, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # Data on the full dataset should be the same as the marginal likelihood
    q = gpx.variational_families.CollapsedVariationalGaussian(
        posterior=p * likelihood, inducing_inputs=D.X
    )
    expected_value = -conjugate_mll(p * likelihood, D)
    actual_value = -collapsed_elbo(q, D)
    assert jnp.abs(actual_value - expected_value) / expected_value < 1e-6


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("key_val", [123, 42])
@pytest.mark.parametrize("binary", [True, False])
def test_elbo(n_points, n_dims, key_val, binary: bool):
    key = jr.key(key_val)
    D = build_data(n_points, n_dims, key, binary=binary)
    z = jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_points // 2, n_dims))

    # Build model
    p = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(active_dims=list(range(n_dims))),
        mean_function=gpx.mean_functions.Constant(),
    )
    if binary:
        likelihood = gpx.likelihoods.Bernoulli(num_datapoints=n_points)
    else:
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_points)
    post = p * likelihood

    q = gpx.variational_families.VariationalGaussian(posterior=post, inducing_inputs=z)

    # test simple call
    res_simple = -elbo(q, D)
    assert isinstance(res_simple, jax.Array)
    assert res_simple.shape == ()

    # test call wrapped in loss function
    graphdef, state, *states = nnx.split(q, Parameter, ...)

    def loss(params):
        posterior = nnx.merge(graphdef, params, *states)
        return -elbo(posterior, D)

    res_wrapped = loss(state)
    assert jnp.allclose(res_simple, res_wrapped)

    # test loss with jit
    loss_jit = jax.jit(loss)
    res_jit = loss_jit(state)
    assert jnp.allclose(res_simple, res_jit)

    # test loss with grad
    loss_grad = jax.grad(loss)
    grad_res = loss_grad(state)
    assert isinstance(grad_res, nnx.State)
