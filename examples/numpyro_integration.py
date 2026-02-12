# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joint Inference with Numpyro
#
# In this notebook, we demonstrate how to use [Numpyro](https://num.pyro.ai/) to perform fully
# Bayesian inference over the hyperparameters of a Gaussian process model.  We will look at a
# scenario where we have a structured mean function in the form of a linear model, and a GP
# capturing the residuals. We will infer the parameters of both the linear model and the GP jointly.

# %%
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
)

from examples.utils import use_mpl_style
import gpjax as gpx
from gpjax.numpyro_extras import register_parameters

config.update("jax_enable_x64", True)

use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.key(123)
keys = jr.split(key, 4)

# %% [markdown]
# ## Data Generation
#
# We generate a synthetic dataset that consists of a linear trend together with a locally periodic
# residual signal whose amplitude varies over time, an additional high-frequency component, and a
# local bump. This data generating process is purposefully designed to illustrate the benefit of
# incorporating a Gaussian process into a larger Bayesian model; however, such structures are
# common.

# %%
N = 200

x = jnp.sort(jr.uniform(keys[0], shape=(N, 1), minval=0.0, maxval=10.0), axis=0)

# True parameters for the linear trend
true_slope = 0.45
true_intercept = 1.5

# Structured residual signal captured by the GP
slow_period = 6.0
fast_period = 0.8
amplitude_envelope = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x / slow_period)
modulated_periodic = amplitude_envelope * jnp.sin(2 * jnp.pi * x / fast_period)
high_frequency_component = 0.3 * jnp.cos(2 * jnp.pi * x / 0.35)
localised_bump = 1.2 * jnp.exp(-0.5 * ((x - 7.0) / 0.45) ** 2)

linear_trend = true_slope * x + true_intercept
residual_signal = modulated_periodic + high_frequency_component + localised_bump
signal = linear_trend + residual_signal

# Observations with homoscedastic noise
observation_noise = 0.7
y = signal + observation_noise * jr.normal(keys[1], shape=x.shape)

D = gpx.Dataset(X=x, y=y)

fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Observations", color=cols[0])
ax.plot(x, signal, "--", label="True Signal", color=cols[1])
ax.legend()

# %% [markdown]
# ## Model Definition
#
# We define a GP model with a zero mean function, as we will handle the linear
# trend explicitly in the Numpyro model. Naturally, one could parameterise the GP with a
# linear mean function; however, this design is purely pedagogical. For the kernel, we specify a
# product of a periodic kernel and an RBF kernel. This choice reflects our prior knowledge that
# the signal is locally periodic. For a more in-depth look at how complex kernels can be designed,
# see our
# [Introduction to Kernels](https://docs.jaxgaussianprocesses.com/_examples/intro_to_kernels/)
# notebook.
#
# We see in the below that priors are specified on the parameters' constrained space. For
# example, the lengthscale parameter must be strictly positive and, therefore, a unit-Gaussian
# would be a poor choice of prior. Instead, we opt for the log-Gaussian as the prior distribution
# as its support matches that of our lengthscale parameter. Attaching a prior to a parameter is
# straightforward using the `prior` argument in the parameter's class and specifying any
# [numpyro distribution](https://num.pyro.ai/en/stable/distributions.html).

# %%
# Define priors
lengthscale_prior = dist.LogNormal(0.0, 1.0)
variance_prior = dist.LogNormal(0.0, 1.0)
period_prior = dist.LogNormal(0.0, 0.5)
noise_prior = dist.LogNormal(0.0, 1.0)

# We can explicitly attach priors to the parameters
lengthscale = gpx.parameters.PositiveReal(1.0, prior=lengthscale_prior)
variance = gpx.parameters.PositiveReal(1.0, prior=variance_prior)
period = gpx.parameters.PositiveReal(1.0, prior=period_prior)
noise = gpx.parameters.NonNegativeReal(1.0, prior=noise_prior)

# %% [markdown]
#
# Now that all of our parameters are defined, we'll proceed to construct the Gaussian process in
# the ordinary fashion. For a deeper look at how this is done, our
# [Regression](https://docs.jaxgaussianprocesses.com/_examples/regression/)
# notebook is a good starting point.

# %%
stationary_component = gpx.kernels.RBF(
    lengthscale=lengthscale,
    variance=variance,
)
periodic_component = gpx.kernels.Periodic(
    lengthscale=lengthscale,
    period=period,
)
kernel = stationary_component * periodic_component

meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

likelihood = gpx.likelihoods.Gaussian(
    num_datapoints=N,
    obs_stddev=noise,
)
posterior = prior * likelihood

# %% [markdown]
# ## Joint Inference Loop
#
# With a GPJax Posterior object now defined, the only outstanding task is to integrate
# it into a full Numpyro model. This notebook is not designed to be a full introduction to
# Numpyro (for that, see the excellent
# [Numpyro Documentation](https://num.pyro.ai/en/stable/)); however, in the below
# model we first sample the slope and intercept parameters of the linear component.
# We then compute the residuals between the observed data and the linear component,
# before then computing the GP marginal log-likelihood of the residual.
#
# The key step in the below is registering the parameters of the GPJax model with
# Numpyro via GPJax's `register_parameters` function. This function automatically
# samples parameters of the model and returns an updated state of the model with those
# sampled values used as parameters.


# %%
def model(X, Y, X_new=None):
    # 1. Sample linear model parameters
    slope = numpyro.sample("slope", dist.Normal(0.0, 2.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 2.0))
    linear_component = slope * X + intercept

    residuals = Y - linear_component

    p_posterior = register_parameters(posterior)
    D_resid = gpx.Dataset(X=X, y=residuals)
    mll = gpx.objectives.conjugate_mll(p_posterior, D_resid)

    numpyro.factor("gp_log_lik", mll)

    if X_new is not None:
        latent_dist = p_posterior.predict(X_new, train_data=D_resid)
        f_new = numpyro.sample("f_new", latent_dist)
        f_new = f_new.reshape((-1, 1))

        # Add observation noise to get noisy predictions
        obs_stddev = p_posterior.likelihood.obs_stddev[...]
        y_noise = numpyro.sample(
            "y_noise",
            dist.Normal(0.0, obs_stddev).expand(f_new.shape).to_event(f_new.ndim),
        )

        total_prediction = slope * X_new + intercept + f_new + y_noise
        numpyro.deterministic("y_pred", total_prediction)
        return total_prediction


# %% [markdown]
# ## Running MCMC
#
# Using Numpyro's NUTS sampler, we can now draw samples from the posterior. To ensure
# our documentation can be quickly built, we limit the number of samples and the length
# of the burn-in phase below. However, in practice, one should draw more samples from
# multiple chains using the `num_chains` argument in the `MCMC` constructor.

# %%
nuts_kernel = NUTS(model)
# In practice, one should run more samples from multiple chains.
mcmc = MCMC(
    nuts_kernel,
    num_warmup=500,
    num_samples=500,
)
mcmc.run(keys[2], x, y)
mcmc.print_summary()

# %% [markdown]
# ## Analysis and Plotting
#
# Having obtained samples from the posterior, we now evaluate the predictive posterior
# at the test sites. In our
# [Poisson Regression](https://docs.jaxgaussianprocesses.com/_examples/poisson/), this
# process is done manually. However, by virtue of using Numpyro here, we may instead
# use Numpyro's `Predictive` object to handle this process for us. Once samples are
# drawn from the predictive posterior distribution, we may evaluate the mean and 95%
# credible interval and compare our model's predictions to the underlying data.

# %%
samples = mcmc.get_samples()
predictive = Predictive(
    model,
    posterior_samples=samples,
    return_sites=["y_pred"],
)

x_test = jnp.linspace(-0.5, 10.5, 1000).reshape(-1, 1)
predictions = predictive(keys[3], x, y, X_new=x_test)
y_pred = predictions["y_pred"]

mean_prediction = jnp.mean(y_pred, axis=0)
lower, upper = jnp.percentile(y_pred, jnp.array([2.5, 97.5]), axis=0)

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.5, label="Observations", color=cols[0])
ax.plot(x, signal, "--", label="True Signal", color=cols[0])

ax.plot(x_test, mean_prediction, "-", label="Posterior Mean", color=cols[1])
ax.fill_between(
    x_test.flatten(),
    lower.flatten(),
    upper.flatten(),
    color=cols[1],
    alpha=0.2,
    label="95% Credible Interval",
)
ax.legend()

# %% [markdown]
# ## Conclusions
# This concludes our introduction to the integration of GPJax with Numpyro. The
# presentation given here is designed to best illustrate *how* the two libraries
# integrate. For a closer look at the more complex models that one may build by
# integrating Numpyro and GPJax, see our
# [Spatial Semi-Linear Model](https://docs.jaxgaussianprocesses.com/_examples/spatial_linear_gp)
# notebook.


# %% [markdown]
# ## System configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder"
