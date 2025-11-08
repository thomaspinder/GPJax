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

import typing as tp

from flax import nnx
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from numpyro.distributions.transforms import Transform
import optax as ox
from scipy.optimize import minimize

from gpjax.dataset import Dataset
from gpjax.objectives import Objective
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)
from gpjax.scan import vscan
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

Model = tp.TypeVar("Model", bound=nnx.Module)


def fit(  # noqa: PLR0913
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    params_bijection: tp.Union[dict[Parameter, Transform], None] = DEFAULT_BIJECTION,
    trainable: nnx.filterlib.Filter = Parameter,
    key: KeyArray = jr.PRNGKey(42),
    num_iters: int = 100,
    batch_size: int = -1,
    log_rate: int = 10,
    verbose: bool = True,
    unroll: int = 1,
    safe: bool = True,
) -> tuple[Model, jax.Array]:
    r"""Train a Module model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Example:
    ```pycon
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>> from gpjax.parameters import PositiveReal
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>> # (2) Define your model:
        >>> class LinearModel(nnx.Module):
        >>>     def __init__(self, weight: float, bias: float):
        >>>         self.weight = PositiveReal(weight)
        >>>         self.bias = bias
        >>>
        >>>     def __call__(self, x):
        >>>         return self.weight.value * x + self.bias
        >>>
        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> def mse(model, data):
        >>>     pred = model(data.X)
        >>>     return jnp.mean((pred - data.y) ** 2)
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
        >>>     model=model, objective=mse, train_data=D, optim=ox.sgd(0.001), num_iters=1000
        >>> )
    ```

    Args:
        model (Model): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        trainable (nnx.filterlib.Filter): Filter to determine which parameters are trainable.
            Defaults to nnx.Param (all Parameter instances).
        num_iters (int): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (int): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (KeyArray): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (int): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (bool): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns:
        A tuple comprising the optimised model and training history.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    # Parameters bijection to unconstrained space
    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    # Loss definition
    def loss(params: nnx.State, batch: Dataset) -> ScalarFloat:
        params = transform(params, params_bijection)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, batch)

    # Initialise optimiser state.
    opt_state = optim.init(params)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)

        carry = params, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (params, _), history = scan(step, (params, opt_state), (iter_keys), unroll=unroll)

    # Parameters bijection to constrained space
    if params_bijection is not None:
        params = transform(params, params_bijection)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, history


def fit_scipy(  # noqa: PLR0913
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    trainable: nnx.filterlib.Filter = Parameter,
    max_iters: int = 500,
    verbose: bool = True,
    safe: bool = True,
) -> tuple[Model, Array]:
    r"""Train a Module model with respect to a supplied Objective function.
    Optimisers used here should originate from Optax. todo

    Args:
        model: the model Module to be optimised.
        objective: The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        max_iters (int): The maximum number of optimisation steps to run. Defaults
            to 500.
        verbose (bool): Whether to print the information about the optimisation. Defaults
            to True.

    Returns:
        A tuple comprising the optimised model and training history.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_num_iters(max_iters)
        _check_verbose(verbose)

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    # Parameters bijection to unconstrained space
    params = transform(params, DEFAULT_BIJECTION, inverse=True)

    # Loss definition
    def loss(params) -> ScalarFloat:
        params = transform(params, DEFAULT_BIJECTION)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, train_data)

    # convert to numpy for interface with scipy
    x0, scipy_to_jnp = ravel_pytree(params)

    @jax.jit
    def scipy_wrapper(x0):
        value, grads = jax.value_and_grad(loss)(scipy_to_jnp(jnp.array(x0)))
        scipy_grads = ravel_pytree(grads)[0]
        return value, scipy_grads

    history = [scipy_wrapper(x0)[0]]
    result = minimize(
        fun=scipy_wrapper,
        x0=x0,
        jac=True,
        callback=lambda X: history.append(scipy_wrapper(X)[0]),
        options={"maxiter": max_iters, "disp": verbose},
    )
    history = jnp.array(history)

    # convert back to nnx.State with JAX arrays
    params = scipy_to_jnp(result.x)

    # Parameters bijection to constrained space
    params = transform(params, DEFAULT_BIJECTION)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, history


def fit_lbfgs(
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    params_bijection: tp.Union[dict[Parameter, Transform], None] = DEFAULT_BIJECTION,
    trainable: nnx.filterlib.Filter = Parameter,
    max_iters: int = 100,
    safe: bool = True,
    max_linesearch_steps: int = 32,
    gtol: float = 1e-5,
) -> tuple[Model, jax.Array]:
    r"""Train a Module model with respect to a supplied Objective function.

    Uses Optax's LBFGS implementation and a jax.lax.while loop.

     Args:
         model: the model Module to be optimised.
         objective: The objective function that we are optimising with
             respect to.
         train_data (Dataset): The training data to be used for the optimisation.
         max_iters (int): The maximum number of optimisation steps to run. Defaults
             to 500.
         safe (bool): Whether to check the types of the inputs.
         max_linesearch_steps (int): The maximum number of linesearch steps to use
            for finding the stepsize.
        gtol (float): Terminate the optimisation if the L2 norm of the gradient is
            below this threshold.

     Returns:
         A tuple comprising the optimised model and final loss.
    """
    if safe:
        # Check inputs
        _check_model(model)
        _check_train_data(train_data)
        _check_num_iters(max_iters)

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    # Parameters bijection to unconstrained space
    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    # Loss definition
    def loss(params: nnx.State) -> ScalarFloat:
        params = transform(params, params_bijection)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, train_data)

    # Initialise optimiser
    optim = ox.lbfgs(
        linesearch=ox.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            initial_guess_strategy="one",
        )
    )
    opt_state = optim.init(params)
    loss_value_and_grad = ox.value_and_grad_from_state(loss)

    # Optimisation step.
    def step(carry):
        params, opt_state = carry

        # Using optax's value_and_grad_from_state is more efficient given LBFGS uses a linesearch
        # See https://optax.readthedocs.io/en/latest/api/utilities.html#optax.value_and_grad_from_state
        loss_val, loss_gradient = loss_value_and_grad(params, state=opt_state)
        updates, opt_state = optim.update(
            loss_gradient,
            opt_state,
            params,
            value=loss_val,
            grad=loss_gradient,
            value_fn=loss,
        )
        params = ox.apply_updates(params, updates)

        return params, opt_state

    def continue_fn(carry):
        _, opt_state = carry
        n = ox.tree_utils.tree_get(opt_state, "count")
        g = ox.tree_utils.tree_get(opt_state, "grad")
        g_l2_norm = ox.tree_utils.tree_l2_norm(g)
        return (n == 0) | ((n < max_iters) & (g_l2_norm >= gtol))

    # Optimisation loop
    params, opt_state = jax.lax.while_loop(
        continue_fn,
        step,
        (params, opt_state),
    )
    final_loss = ox.tree_utils.tree_get(opt_state, "value")

    # Parameters bijection to constrained space
    if params_bijection is not None:
        params = transform(params, params_bijection)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, final_loss


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indices with replacement.
    indices = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indices], y=y[indices])


def _check_model(model: tp.Any) -> None:
    """Check that the model is a subclass of nnx.Module."""
    if not isinstance(model, nnx.Module):
        raise TypeError(
            "Expected model to be a subclass of nnx.Module. "
            f"Got {model} of type {type(model)}."
        )


def _check_train_data(train_data: tp.Any) -> None:
    """Check that the train_data is of type gpjax.Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError(
            "Expected train_data to be of type gpjax.Dataset. "
            f"Got {train_data} of type {type(train_data)}."
        )


def _check_optim(optim: tp.Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError(
            "Expected optim to be of type optax.GradientTransformation. "
            f"Got {optim} of type {type(optim)}."
        )


def _check_num_iters(num_iters: tp.Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError(
            "Expected num_iters to be of type int. "
            f"Got {num_iters} of type {type(num_iters)}."
        )

    if num_iters <= 0:
        raise ValueError(f"Expected num_iters to be positive. Got {num_iters}.")


def _check_log_rate(log_rate: tp.Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError(
            "Expected log_rate to be of type int. "
            f"Got {log_rate} of type {type(log_rate)}."
        )

    if not log_rate > 0:
        raise ValueError(f"Expected log_rate to be positive. Got {log_rate}.")


def _check_verbose(verbose: tp.Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError(
            "Expected verbose to be of type bool. "
            f"Got {verbose} of type {type(verbose)}."
        )


def _check_batch_size(batch_size: tp.Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError(
            "Expected batch_size to be of type int. "
            f"Got {batch_size} of type {type(batch_size)}."
        )

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError(f"Expected batch_size to be positive or -1. Got {batch_size}.")


__all__ = [
    "fit",
    "get_batch",
]
