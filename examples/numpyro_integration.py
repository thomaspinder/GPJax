# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joint Inference with Numpyro
#
# In this notebook, we demonstrate how to use [Numpyro](https://num.pyro.ai/) to perform fully Bayesian inference over the hyperparameters of a Gaussian process model.
# We will look at a scenario where we have a structured mean function (a linear model) and a GP capturing the residuals. We will infer the parameters of both the linear model and the GP jointly.

# %%
import numpyro
numpyro.set_host_device_count(4)

from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
)

import gpjax as gpx
from gpjax.numpyro_extras import register_parameters

config.update("jax_enable_x64", True)

key = jr.key(123)

# %% [markdown]
# ## Data Generation
#
# We generate a synthetic dataset that consists of a linear trend together with a locally periodic residual signal whose amplitude varies over time, an additional high-frequency component, and a local bump. This richer structure highlights how a GP can capture deviations from the explicit linear model.

# %%
N = 200
key_x, key_y = jr.split(key)
x = jnp.sort(jr.uniform(key_x, shape=(N, 1), minval=0.0, maxval=10.0), axis=0)

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
y_clean = linear_trend + residual_signal

# Observations with homoscedastic noise
observation_noise = 0.3
y = y_clean + observation_noise * jr.normal(key_y, shape=x.shape)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Data", alpha=0.6)
plt.plot(x, y_clean, "k--", label="True Signal")
plt.legend()
# plt.show()

# %% [markdown]
# ## Model Definition
#
# We define a GP model with a generic mean function (zero for now, as we will handle the linear trend explicitly in the Numpyro model) and a kernel that is the product of a periodic kernel and an RBF kernel. This choice reflects our prior knowledge that the signal is locally periodic.

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

# Define Kernel with priors
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

# We will use a ConjugatePosterior since we assume Gaussian noise
likelihood = gpx.likelihoods.Gaussian(
    num_datapoints=N,
    obs_stddev=gpx.parameters.NonNegativeReal(1.0, prior=dist.LogNormal(0.0, 1.0)),
)
posterior = prior * likelihood

# We initialise the model parameters.
# Note: These values will be overwritten by Numpyro samples during inference.
D = gpx.Dataset(X=x, y=y)

# %% [markdown]
# ## Joint Inference Loop
#
# We define a Numpyro model function that:
# 1. Samples the parameters for the linear trend.
# 2. Computes the residuals (Data - Linear Trend).
# 3. Samples the GP hyperparameters using `register_parameters`.
# 4. Computes the GP marginal log-likelihood on the residuals.
# 5. Adds the GP log-likelihood to the joint density.


# %%
def model(X, Y, X_new=None):
    # 1. Sample linear model parameters
    slope = numpyro.sample("slope", dist.Normal(0.0, 2.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 2.0))

    # Calculate residuals
    trend = slope * X + intercept
    residuals = Y - trend

    # 2. Register GP parameters
    # This automatically samples parameters from the GPJax model
    # and returns a model with updated values.
    # We attached priors to the parameters during model definition,
    # so register_parameters will use those.
    # register_parameters modifies the model in-place (and returns it).
    # Since Numpyro re-runs this function, we are overwriting the parameters
    # of the same object repeatedly, which is fine as they are completely determined
    # by the sample sites.
    p_posterior = register_parameters(posterior)

    # Create dataset for residuals
    D_resid = gpx.Dataset(X=X, y=residuals)

    # 3. Compute MLL
    # We use conjugate_mll which computes log p(y | X, theta) analytically for Gaussian likelihoods.
    mll = gpx.objectives.conjugate_mll(p_posterior, D_resid)

    # 4. Add to potential
    numpyro.factor("gp_log_lik", mll)

    # Optional prediction branch for use with Predictive
    if X_new is not None:
        latent_dist = p_posterior.predict(X_new, train_data=D_resid)
        f_new = numpyro.sample("f_new", latent_dist)
        f_new = f_new.reshape((-1, 1))
        total_prediction = slope * X_new + intercept + f_new
        numpyro.deterministic("y_pred", total_prediction)
        return total_prediction


# %% [markdown]
# ## Running MCMC
#
# We use the NUTS sampler to draw samples from the posterior.

# %%
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=1500, num_samples=2000, num_chains=4, chain_method="parallel")
mcmc.run(jr.key(123), x, y)

mcmc.print_summary()

# %% [markdown]
# ## Analysis and Plotting
#
# We extract the samples and plot the predictions.

# %%
# Draw posterior samples for downstream use
samples = mcmc.get_samples()

# Create predictive utility that reuses the original model
predictive = Predictive(
    model,
    posterior_samples=samples,
    return_sites=["y_pred"],
)

# Generate predictions
predictions = predictive(jr.key(1), x, y, X_new=x)
y_pred = predictions["y_pred"]

# Compute statistics
mean_prediction = jnp.mean(y_pred, axis=0)
std_prediction = jnp.std(y_pred, axis=0)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.5, label="Data", color="gray")
plt.plot(x, y_clean, "k--", label="True Signal")

plt.plot(x, mean_prediction, "b-", label="Posterior Mean")
plt.fill_between(
    x.flatten(),
    mean_prediction.flatten() - 2 * std_prediction.flatten(),
    mean_prediction.flatten() + 2 * std_prediction.flatten(),
    color="b",
    alpha=0.2,
    label="95% CI (GP Uncertainty)",
)

plt.legend()
# plt.show()
