# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import equinox as eqx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.fit import (
    _check_batch_size,
    _check_log_rate,
    _check_model,
    _check_num_iters,
    _check_optim,
    _check_train_data,
    _check_verbose,
    fit,
    fit_lbfgs,
    fit_scipy,
    get_batch,
)
from gpjax.gps import (
    ConjugatePosterior,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Constant,
)
from gpjax.objectives import (
    conjugate_mll,
    elbo,
)
from gpjax.parameters import (
    PositiveReal,
)
from gpjax.typing import Array
from gpjax.variational_families import VariationalGaussian
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Float,
    Num,
)
import optax as ox
import paramax
from paramax import AbstractUnwrappable
import pytest
import scipy


def _val(x):
    """Unwrap a paramax parameter or return the value directly."""
    return x.unwrap() if isinstance(x, AbstractUnwrappable) else x


class LinearModel(eqx.Module):
    weight: PositiveReal
    bias: float = eqx.field(static=True, default=1.0)

    def __init__(self, weight: float = 1.0, bias: float = 1.0):
        self.weight = PositiveReal(weight)
        self.bias = bias

    def __call__(self, x):
        return _val(self.weight) * x + self.bias


def test_fit_simple() -> None:
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    def mse(model, data):
        pred = model(data.X)
        return jnp.mean((pred - data.y) ** 2)

    # Train!
    trained_model, hist = fit(
        model=model,
        objective=mse,
        train_data=D,
        optim=ox.sgd(0.001),
        num_iters=100,
        key=jr.key(123),
    )

    # Ensure we return a history of the correct length
    assert len(hist) == 100

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert mse(trained_model, D) < mse(model, D)

    # Test stop_gradient on bias (static field, not trained):
    assert trained_model.bias == 1.0


def test_fit_scipy_simple():
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    def mse(model, data):
        pred = model(data.X)
        return jnp.mean((pred - data.y) ** 2)

    # Train with bfgs!
    trained_model, hist = fit_scipy(
        model=model,
        objective=mse,
        train_data=D,
        max_iters=10,
    )

    # Ensure we return a history of the correct length
    assert len(hist) > 2

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert mse(trained_model, D) < mse(model, D)

    # Test stop_gradient on bias (static field, not trained):
    assert trained_model.bias == 1.0


def test_fit_lbfgs_simple():
    # Create dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.key(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    model = LinearModel(weight=1.0, bias=1.0)

    # Define loss function:
    def mse(model, data):
        pred = model(data.X)
        return jnp.mean((pred - data.y) ** 2)

    # Train with bfgs!
    trained_model, _final_loss = fit_lbfgs(
        model=model,
        objective=mse,
        train_data=D,
        max_iters=10,
    )

    # Ensure we return a model of the same class
    assert isinstance(trained_model, LinearModel)

    # Test reduction in loss:
    assert mse(trained_model, D) < mse(model, D)

    # Test stop_gradient on bias (static field, not trained):
    assert trained_model.bias == 1.0


@pytest.mark.parametrize("n_data", [20])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit_gp_regression(n_data: int, verbose: bool) -> None:
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian(num_datapoints=n_data)
    posterior = prior * likelihood

    # Train!
    trained_model, history = fit(
        model=posterior,
        objective=conjugate_mll,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=15,
        verbose=verbose,
        key=jr.key(123),
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model, ConjugatePosterior)

    # Ensure we return a history of the correct length
    assert len(history) == 15

    # Ensure we reduce the loss
    assert conjugate_mll(trained_model, D) < conjugate_mll(posterior, D)


@pytest.mark.parametrize("n_data", [20])
def test_fit_lbfgs_gp_regression(n_data: int) -> None:
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian(num_datapoints=n_data)
    posterior = prior * likelihood

    # Train with BFGS!
    trained_model_bfgs, _final_loss = fit_lbfgs(
        model=posterior,
        objective=conjugate_mll,
        train_data=D,
        max_iters=40,
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model_bfgs, ConjugatePosterior)

    # Ensure we reduce the loss
    assert conjugate_mll(trained_model_bfgs, D) < conjugate_mll(posterior, D)


def test_fit_scipy_error_raises() -> None:
    # Create dataset:
    D = Dataset(
        X=jnp.array([[0.0]], dtype=jnp.float64), y=jnp.array([[0.0]], dtype=jnp.float64)
    )

    # build crazy mean function so that opt fails
    class CrazyMean(AbstractMeanFunction):
        def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
            return jnp.heaviside(x, 100.0)

    # Define GP model with crazy mean function:
    prior = Prior(kernel=RBF(), mean_function=CrazyMean())
    likelihood = Gaussian(num_datapoints=2)
    posterior = prior * likelihood

    with pytest.raises(scipy.optimize.OptimizeWarning):
        fit_scipy(
            model=posterior,
            objective=conjugate_mll,
            train_data=D,
            max_iters=10,
        )

    # also check fails if no given enough steps
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian(num_datapoints=2)
    posterior = prior * likelihood

    with pytest.raises(scipy.optimize.OptimizeWarning):
        fit_scipy(
            model=posterior,
            objective=conjugate_mll,
            train_data=D,
            max_iters=1,
        )


@pytest.mark.parametrize("num_iters", [1, 5])
@pytest.mark.parametrize("batch_size", [1, 20, 50])
@pytest.mark.parametrize("n_data", [50])
@pytest.mark.parametrize("verbose", [True, False])
def test_fit_batch(num_iters: int, batch_size: int, n_data: int, verbose: bool) -> None:
    # Create dataset:
    key = jr.key(123)
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, 1)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Define GP model:
    prior = Prior(kernel=RBF(), mean_function=Constant())
    likelihood = Gaussian(num_datapoints=n_data)
    posterior = prior * likelihood

    # Define variational family:
    z = jnp.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    q = VariationalGaussian(posterior=posterior, inducing_inputs=z)

    # Train!
    trained_model, history = fit(
        model=q,
        objective=elbo,
        train_data=D,
        optim=ox.adam(0.1),
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose,
        key=jr.key(123),
    )

    # Ensure the trained model is a Gaussian process posterior
    assert isinstance(trained_model, VariationalGaussian)

    # Ensure we return a history of the correct length
    assert len(history) == num_iters

    # Ensure we reduce the loss
    assert elbo(trained_model, D) < elbo(q, D)


@pytest.mark.parametrize("n_data", [50])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 50])
def test_get_batch(n_data: int, n_dim: int, batch_size: int):
    key = jr.key(123)

    # Create dataset:
    x = jnp.sort(
        jr.uniform(key=key, minval=-2.0, maxval=2.0, shape=(n_data, n_dim)), axis=0
    )
    y = jnp.sin(x) + jr.normal(key=key, shape=x.shape) * 0.1
    D = Dataset(X=x, y=y)

    # Sample out a batch:
    B = get_batch(D, batch_size, key)

    # Check batch is correct size and shape dimensions:
    assert B.n == batch_size
    assert B.X.shape[1:] == x.shape[1:]
    assert B.y.shape[1:] == y.shape[1:]

    # Ensure no caching of batches:
    key, subkey = jr.split(key)
    New = get_batch(D, batch_size, subkey)
    assert New.n == batch_size
    assert New.X.shape[1:] == x.shape[1:]
    assert New.y.shape[1:] == y.shape[1:]
    assert jnp.sum(New.X == B.X) <= n_dim * batch_size / n_data
    assert jnp.sum(New.y == B.y) <= n_dim * batch_size / n_data


@pytest.fixture
def valid_model() -> eqx.Module:
    """Return a valid model for testing."""
    return LinearModel(weight=1.0, bias=1.0)


@pytest.fixture
def valid_dataset() -> Dataset:
    """Return a valid dataset for testing."""
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([[1.0], [2.0], [3.0]])
    return Dataset(X=X, y=y)


def test_check_model_valid(valid_model: eqx.Module) -> None:
    """Test that a valid model passes validation."""
    _check_model(valid_model)


def test_check_model_invalid() -> None:
    """Test that an invalid model raises a TypeError."""
    model = "not a model"
    with pytest.raises(
        TypeError, match=r"Expected model to be a subclass of eqx\.Module"
    ):
        _check_model(model)


def test_check_train_data_valid(valid_dataset: Dataset) -> None:
    """Test that valid training data passes validation."""
    _check_train_data(valid_dataset)


def test_check_train_data_invalid() -> None:
    """Test that invalid training data raises a TypeError."""
    train_data = "not a dataset"
    with pytest.raises(
        TypeError, match=r"Expected train_data to be of type gpjax\.Dataset"
    ):
        _check_train_data(train_data)


def test_check_optim_valid() -> None:
    """Test that a valid optimiser passes validation."""
    optim = ox.sgd(0.1)
    _check_optim(optim)


def test_check_optim_invalid() -> None:
    """Test that an invalid optimiser raises a TypeError."""
    optim = "not an optimiser"
    with pytest.raises(
        TypeError, match=r"Expected optim to be of type optax\.GradientTransformation"
    ):
        _check_optim(optim)


@pytest.mark.parametrize("num_iters", [1, 10, 100])
def test_check_num_iters_valid(num_iters: int) -> None:
    """Test that valid number of iterations passes validation."""
    _check_num_iters(num_iters)


def test_check_num_iters_invalid_type() -> None:
    """Test that an invalid num_iters type raises a TypeError."""
    num_iters = "not an int"
    with pytest.raises(TypeError, match="Expected num_iters to be of type int"):
        _check_num_iters(num_iters)


@pytest.mark.parametrize("num_iters", [0, -5])
def test_check_num_iters_invalid_value(num_iters: int) -> None:
    """Test that an invalid num_iters value raises a ValueError."""
    with pytest.raises(ValueError, match="Expected num_iters to be positive"):
        _check_num_iters(num_iters)


@pytest.mark.parametrize("log_rate", [1, 10, 100])
def test_check_log_rate_valid(log_rate: int) -> None:
    """Test that a valid log rate passes validation."""
    _check_log_rate(log_rate)


def test_check_log_rate_invalid_type() -> None:
    """Test that an invalid log_rate type raises a TypeError."""
    log_rate = "not an int"
    with pytest.raises(TypeError, match="Expected log_rate to be of type int"):
        _check_log_rate(log_rate)


@pytest.mark.parametrize("log_rate", [0, -5])
def test_check_log_rate_invalid_value(log_rate: int) -> None:
    """Test that an invalid log_rate value raises a ValueError."""
    with pytest.raises(ValueError, match="Expected log_rate to be positive"):
        _check_log_rate(log_rate)


@pytest.mark.parametrize("verbose", [True, False])
def test_check_verbose_valid(verbose: bool) -> None:
    """Test that valid verbose values pass validation."""
    _check_verbose(verbose)


def test_check_verbose_invalid() -> None:
    """Test that an invalid verbose value raises a TypeError."""
    verbose = "not a bool"
    with pytest.raises(TypeError, match="Expected verbose to be of type bool"):
        _check_verbose(verbose)


@pytest.mark.parametrize("batch_size", [1, 10, 100, -1])
def test_check_batch_size_valid(batch_size: int) -> None:
    """Test that valid batch sizes pass validation."""
    _check_batch_size(batch_size)


def test_check_batch_size_invalid_type() -> None:
    """Test that an invalid batch_size type raises a TypeError."""
    batch_size = "not an int"
    with pytest.raises(TypeError, match="Expected batch_size to be of type int"):
        _check_batch_size(batch_size)


@pytest.mark.parametrize("batch_size", [0, -2, -5])
def test_check_batch_size_invalid_value(batch_size: int) -> None:
    """Test that invalid batch_size values raise a ValueError."""
    with pytest.raises(ValueError, match="Expected batch_size to be positive or -1"):
        _check_batch_size(batch_size)


def test_fit_freeze_kernel_variance() -> None:
    """Test that fit can freeze kernel variance parameter using paramax.non_trainable."""
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(X) + 0.1 * jr.normal(jr.key(43), (20, 1))
    D = Dataset(X, y)

    # Create GP with RBF kernel
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    # Record initial variance value
    initial_variance = posterior.prior.kernel.variance.unwrap()

    # Freeze variance using paramax.non_trainable + eqx.tree_at
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.kernel.variance,
        posterior,
        replace_fn=paramax.non_trainable,
    )

    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=gpx.objectives.conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.01),
        num_iters=10,
        verbose=False,
    )

    # Use paramax.unwrap to fully resolve all wrappers for comparison
    unwrapped = paramax.unwrap(trained_posterior)

    # Assert variance has not changed
    assert jnp.allclose(unwrapped.prior.kernel.variance, initial_variance)

    # Assert lengthscale has changed
    assert not jnp.allclose(unwrapped.prior.kernel.lengthscale, 1.0)


def test_fit_zero_mean_function_frozen_with_non_trainable() -> None:
    """Test that Zero mean function constant can be frozen using paramax.non_trainable.

    In the equinox backend, plain JAX arrays are trainable by default.
    To freeze the Zero mean function's constant, use paramax.non_trainable.
    """
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = jnp.ones_like(X) + 0.1 * jr.normal(jr.key(43), X.shape)  # Non-zero mean data
    D = Dataset(X, y)

    # Create GP with Zero mean function
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    # Record initial mean function constant (should be 0.0)
    initial_constant = meanf.constant

    # Freeze the Zero mean function constant using paramax.non_trainable
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.mean_function.constant,
        posterior,
        replace_fn=paramax.non_trainable,
    )

    # Train with frozen constant
    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=gpx.objectives.conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.01),
        num_iters=10,
        verbose=False,
    )

    # Assert Zero mean function constant has not changed (remains 0.0)
    unwrapped = paramax.unwrap(trained_posterior)
    assert jnp.allclose(unwrapped.prior.mean_function.constant, initial_constant)


def test_fit_constant_mean_function_with_parameter() -> None:
    """Test that Constant mean function works with trainable Parameter."""
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = 5.0 * jnp.ones_like(X) + 0.1 * jr.normal(jr.key(43), X.shape)  # Mean of 5.0
    D = Dataset(X, y)

    # Create GP with Constant mean function using Parameter
    from gpjax.parameters import Real

    meanf = gpx.mean_functions.Constant(constant=Real(1.0))  # Start with mean 1.0
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.1)
    posterior = prior * likelihood

    # Record initial mean function constant
    initial_constant = meanf.constant.unwrap()

    # Train (should train the mean function Parameter)
    trained_posterior, _ = fit(
        model=posterior,
        objective=gpx.objectives.conjugate_mll,
        train_data=D,
        optim=ox.adam(0.01),
        num_iters=20,
        verbose=False,
    )

    # Assert mean function constant has changed (parameter is trainable)
    final_constant = trained_posterior.prior.mean_function.constant.unwrap()
    assert not jnp.allclose(final_constant, initial_constant)
    # Just verify the parameter changed (direction depends on optimization dynamics)
    assert jnp.isfinite(final_constant)  # Not NaN/Inf


def test_fit_constant_mean_function_frozen_with_non_trainable() -> None:
    """Test that Constant mean function raw value can be frozen using paramax.non_trainable.

    In the equinox backend, plain JAX arrays are trainable by default.
    To freeze the constant, use paramax.non_trainable via eqx.tree_at.
    """
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = 5.0 * jnp.ones_like(X) + 0.1 * jr.normal(jr.key(43), X.shape)  # Mean of 5.0
    D = Dataset(X, y)

    # Create GP with Constant mean function using raw value
    meanf = gpx.mean_functions.Constant(constant=1.0)  # Fixed mean 1.0
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=0.1)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.1)
    posterior = prior * likelihood

    # Record initial mean function constant
    initial_constant = meanf.constant

    # Freeze the constant using paramax.non_trainable
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.mean_function.constant,
        posterior,
        replace_fn=paramax.non_trainable,
    )

    # Train (constant should NOT change because it is frozen)
    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=gpx.objectives.conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.1),
        num_iters=50,
        verbose=False,
    )

    # Assert mean function constant has NOT changed (frozen with non_trainable)
    unwrapped = paramax.unwrap(trained_posterior)
    assert jnp.allclose(unwrapped.prior.mean_function.constant, initial_constant)


def test_fit_freeze_by_non_trainable() -> None:
    """Test freezing specific parameter types using paramax.non_trainable."""
    key = jr.key(42)
    X = jr.uniform(key, (20, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(X) + 0.1 * jr.normal(jr.key(43), (20, 1))
    D = Dataset(X, y)

    # Create GP with RBF kernel
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    # Record initial values
    initial_variance = posterior.prior.kernel.variance.unwrap()
    initial_lengthscale = posterior.prior.kernel.lengthscale.unwrap()
    initial_obs_stddev = posterior.likelihood.obs_stddev.unwrap()

    # Freeze variance and obs_stddev, train only lengthscale
    frozen_posterior = eqx.tree_at(
        lambda m: m.prior.kernel.variance,
        posterior,
        replace_fn=paramax.non_trainable,
    )
    frozen_posterior = eqx.tree_at(
        lambda m: m.likelihood.obs_stddev,
        frozen_posterior,
        replace_fn=paramax.non_trainable,
    )

    trained_posterior, _ = fit(
        model=frozen_posterior,
        objective=gpx.objectives.conjugate_mll,
        train_data=D,
        optim=ox.sgd(0.01),
        num_iters=10,
        verbose=False,
    )

    # Use paramax.unwrap to fully resolve all wrappers for comparison
    unwrapped = paramax.unwrap(trained_posterior)

    # Assert that frozen parameters have not changed
    assert jnp.allclose(unwrapped.prior.kernel.variance, initial_variance)
    assert jnp.allclose(unwrapped.likelihood.obs_stddev, initial_obs_stddev)

    # Assert lengthscale has changed
    assert not jnp.allclose(unwrapped.prior.kernel.lengthscale, initial_lengthscale)
