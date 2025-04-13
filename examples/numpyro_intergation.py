# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: docs
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np

from flax import nnx
from jax import random
import jax.numpy as jnp
import gpjax as gpx

import numpyro
from numpyro.contrib.module import _update_params
import numpyro.distributions as dist
from numpyro.handlers import condition
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import Predictive
from gpjax.objectives import conjugate_mll
from jax import config
import arviz as az

config.update("jax_enable_x64", True)

rng_key = random.PRNGKey(seed=42)

# %%
n = 200
rng_key, rng_subkey = random.split(rng_key)
x = jnp.linspace(1, jnp.pi, n)
mu_true = jnp.sqrt(x + 0.5) * jnp.sin(9 * x)
sigma_true = 0.15
rng_key, rng_subkey = random.split(rng_key)
y = mu_true + sigma_true * random.normal(rng_key, shape=(n,))

# %%
x_train = x[..., None]
y_train = y.reshape(-1, 1)

xtest = jnp.linspace(1, jnp.pi, 1000)[..., None]

data = gpx.Dataset(x_train, y_train)

# %%
plt.plot(x_train, y_train, "x")
plt.plot(x_train, mu_true, label="True mean")

# %%
prior = gpx.gps.Prior(gpx.kernels.Matern52(), gpx.mean_functions.Constant(0.0))

# %%
from collections import namedtuple
from copy import deepcopy
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
import jax.tree_util as jtu

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import mutable as numpyro_mutable


def nnx_module(name, nn_module):
    """
    Declare a :mod:`~flax.nnx` style neural network inside a
    model so that its parameters are registered for optimization via
    :func:`~numpyro.primitives.param` statements.

    Given a flax NNX ``nn_module``, to evaluate the module, we directly call it.
    In a NumPyro model, the pattern will be::

        # Eager initialization outside the model
        module = nn_module(...)

        # Inside the model
        net = nnx_module("net", module)
        y = net(x)

    :param str name: name of the module to be registered.
    :param flax.nnx.Module nn_module: a pre-initialized `flax nnx` Module instance.
    :return: a callable that takes an array as an input and returns
        the neural network transformed output array.
    """
    try:
        from flax import nnx
    except ImportError as e:
        raise ImportError(
            "Looking like you want to use flax.nnx to declare "
            "nn modules. This is an experimental feature. "
            "You need to install the latest version of `flax` to use this feature. "
            "It can be installed with `pip install git+https://github.com/google/flax.git`."
        ) from e

    graph_def, eager_params_state, eager_other_state = nnx.split(
        nn_module, nnx.Variable, nnx.Not(nnx.Variable)
    )

    eager_params_state_dict = nnx.to_pure_dict(eager_params_state)

    module_params = None
    if eager_params_state:
        module_params = numpyro.param(name + "$params")
    if module_params is None:
        module_params = numpyro.param(name + "$params", eager_params_state_dict)

    eager_other_state_dict = nnx.to_pure_dict(eager_other_state)

    mutable_holder = None
    if eager_other_state_dict:
        mutable_holder = numpyro_mutable(name + "$state")
    if mutable_holder is None:
        mutable_holder = numpyro_mutable(
            name + "$state", {"state": eager_other_state_dict}
        )

    def apply_fn(params, *call_args, **call_kwargs):
        params_state = eager_params_state

        if params:
            nnx.replace_by_pure_dict(params_state, params)

        mutable_state = eager_other_state
        if mutable_holder:
            nnx.replace_by_pure_dict(mutable_state, mutable_holder["state"])

        model = nnx.merge(graph_def, params_state, mutable_state)

        model_call = model(*call_args, **call_kwargs)

        if mutable_holder:
            _, _, new_mutable_state = nnx.split(
                model, nnx.Variable, nnx.Not(nnx.Variable)
            )
            new_mutable_state = jax.lax.stop_gradient(new_mutable_state)
            mutable_holder["state"] = nnx.to_pure_dict(new_mutable_state)

        return model_call

    return partial(apply_fn, module_params)


def random_nnx_module(
    name,
    nn_module,
    prior,
):
    """
    A primitive to create a random :mod:`~flax.nnx` style neural network
    which can be used in MCMC samplers. The parameters of the neural network
    will be sampled from ``prior``.

    :param str name: name of the module to be registered.
    :param flax.nnx.Module nn_module: a pre-initialized `flax nnx` Module instance.
    :param prior: a distribution or a dict of distributions or a callable.
        If it is a distribution, all parameters will be sampled from the same
        distribution. If it is a dict, it maps parameter names to distributions.
        If it is a callable, it takes parameter name and parameter shape as
        inputs and returns a distribution. For example::

            class Linear(nnx.Module):
                def __init__(self, din, dout, *, rngs):
                    self.w = nnx.Param(jax.random.uniform(rngs.params(), (din, dout)))
                    self.b = nnx.Param(jnp.zeros((dout,)))

                def __call__(self, x):
                    return x @ self.w + self.b

            # Eager initialization
            linear = Linear(din=4, dout=1, rngs=nnx.Rngs(params=random.PRNGKey(0)))
            net = random_nnx_module("net", linear, prior={"w": dist.Normal(), "b": dist.Cauchy()})

        Alternatively, we can use a callable. For example the following are equivalent::

            prior=(lambda name, shape: dist.Cauchy() if name.endswith("b") else dist.Normal())
            prior={"w": dist.Normal(), "b": dist.Cauchy()}

    :return: a callable that takes an array as an input and returns
        the neural network transformed output array.
    """

    nn = nnx_module(name, nn_module)

    apply_fn = nn.func
    params = nn.args[0]
    other_args = nn.args[1:]
    keywords = nn.keywords

    new_params = deepcopy(params)

    with numpyro.handlers.scope(prefix=name):
        _update_params(params, new_params, prior)

    return partial(apply_fn, new_params, *other_args, **keywords)


# %%
def model(data: gpx.Dataset):
    gp_prior = random_nnx_module(
        "gp",
        prior,
        prior={
            "kernel.lengthscale": dist.HalfNormal(scale=1),
            "kernel.variance": dist.HalfNormal(scale=1),
            "mean_function.constant": dist.Normal(loc=0, scale=1),
        },
    )

    predictions = gp_prior(data.X)
    prior_mean = predictions.mean
    prior_cov = predictions.covariance_matrix

    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1))
    prior_cov = prior_cov + sigma**2 * jnp.eye(data.n)

    numpyro.sample(
        "likelihood",
        dist.MultivariateNormal(loc=prior_mean.squeeze(), covariance_matrix=prior_cov),
        obs=data.y.squeeze(),
    )


numpyro.render_model(
    model=model,
    model_args=(data,),
    render_distributions=True,
    render_params=True,
)

# %%
prior_predictive = Predictive(model=model, num_samples=100)
rng_key, rng_subkey = random.split(key=rng_key)
prior_predictive_samples = prior_predictive(rng_subkey, data)


idata = az.from_dict(
    prior_predictive={
        k: np.expand_dims(a=np.asarray(v), axis=0)
        for k, v in prior_predictive_samples.items()
    },
)

# %%
# We condition the model on the training data
conditioned_model = condition(model, data={"likelihood": y_train})

guide = AutoNormal(model=conditioned_model)
optimizer = numpyro.optim.Adam(step_size=0.025)
svi = SVI(conditioned_model, guide, optimizer, loss=Trace_ELBO())
n_samples = 8_000
rng_key, rng_subkey = random.split(key=rng_key)
svi_result = svi.run(rng_subkey, n_samples, data)

# %%
params = svi_result.params
posterior_predictive = Predictive(
    model=model,
    guide=guide,
    params=params,
    num_samples=2_000,
    return_sites=[
        "kernel.lengthscale",
        "kernel.variance",
        "mean_function.constant",
        "mu",
        "sigma",
        "likelihood",
    ],
)
rng_key, rng_subkey = random.split(key=rng_key)
posterior_predictive_samples = posterior_predictive(rng_subkey, data)

# %%
posterior_predictive_samples["likelihood"].shape

# %%
idata.extend(
    az.from_dict(
        posterior_predictive={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_predictive_samples.items()
        },
    )
)

# %%
fig, ax = plt.subplots()
az.plot_hdi(
    x,
    idata["posterior_predictive"]["likelihood"],
    color="C1",
    fill_kwargs={"alpha": 0.3, "label": "94% HDI"},
    ax=ax,
)
ax.plot(
    x_train,
    idata["posterior_predictive"]["likelihood"].mean(dim=("chain", "draw")),
    color="C1",
    linewidth=3,
    label="SVI Posterior Mean",
)
ax.plot(x, mu_true, color="C0", label=r"$\mu$", linewidth=3)

# %%
az.plot_trace(idata["posterior_predictive"]["sigma"])
