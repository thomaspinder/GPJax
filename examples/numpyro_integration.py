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
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
)

import gpjax as gpx
from gpjax.numpyro_extras import register_parameters

config.update("jax_enable_x64", True)

key = jr.key(42)

# %% [markdown]
# ## Data Generation
#
# We generate a synthetic dataset that consists of a linear trend, a periodic component, and some noise.

# %%
N = 100
x = jnp.sort(jr.uniform(key, shape=(N, 1), minval=0.0, maxval=10.0), axis=0)

# True parameters
true_slope = 0.5
true_intercept = 2.0
true_period = 2.0
true_lengthscale = 1.0
true_noise = 0.1

# Signal
linear_trend = true_slope * x + true_intercept
periodic_signal = jnp.sin(2 * jnp.pi * x / true_period)
y_clean = linear_trend + periodic_signal

# Observations
y = y_clean + true_noise * jr.normal(key, shape=x.shape)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Data", alpha=0.6)
plt.plot(x, y_clean, "k--", label="True Signal")
plt.legend()
plt.show()

# %% [markdown]
# ## Model Definition
#
# We define a GP model with a generic mean function (zero for now, as we will handle the linear trend explicitly in the Numpyro model) and a kernel that is the product of a periodic kernel and an RBF kernel. This choice reflects our prior knowledge that the signal is locally periodic.

# %%
kernel = gpx.kernels.RBF() * gpx.kernels.Periodic()
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

# We will use a ConjugatePosterior since we assume Gaussian noise
likelihood = gpx.likelihoods.Gaussian(num_datapoints=N)
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
def model(X, Y):
    # 1. Sample linear model parameters
    slope = numpyro.sample("slope", dist.Normal(0.0, 2.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 2.0))

    # Calculate residuals
    trend = slope * X + intercept
    residuals = Y - trend

    # 2. Register GP parameters
    # This automatically samples parameters from the GPJax model
    # and returns a model with updated values.
    # We can specify custom priors if needed, but we'll rely on defaults here.
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


# %% [markdown]
# ## Running MCMC
#
# We use the NUTS sampler to draw samples from the posterior.

# %%
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(jr.key(0), x, y)

mcmc.print_summary()

# %% [markdown]
# ## Analysis and Plotting
#
# We extract the samples and plot the predictions.

# %%
samples = mcmc.get_samples()


# Helper to get predictions
def predict(rng_key, sample_idx):
    # Reconstruct model with sampled values

    # Linear part
    slope = samples["slope"][sample_idx]
    intercept = samples["intercept"][sample_idx]
    trend = slope * x + intercept

    # GP part
    # We use numpyro.handlers.substitute to inject the sampled values into register_parameters
    # to reconstruct the GP model state for this sample.
    sample_dict = {k: v[sample_idx] for k, v in samples.items()}

    with numpyro.handlers.substitute(data=sample_dict):
        # We call register_parameters again to update the posterior object with this sample's values
        p_posterior = register_parameters(posterior)

    # Now predict on residuals
    residuals = y - trend
    D_resid = gpx.Dataset(X=x, y=residuals)

    latent_dist = p_posterior.predict(x, train_data=D_resid)
    predictive_mean = latent_dist.mean
    predictive_std = latent_dist.stddev()

    return trend + predictive_mean, predictive_std


# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.5, label="Data", color="gray")
plt.plot(x, y_clean, "k--", label="True Signal")

# Compute mean prediction (using mean of samples for efficiency)
mean_slope = jnp.mean(samples["slope"])
mean_intercept = jnp.mean(samples["intercept"])
mean_trend = mean_slope * x + mean_intercept

mean_samples = {k: jnp.mean(v, axis=0) for k, v in samples.items()}
with numpyro.handlers.substitute(data=mean_samples):
    p_posterior_mean = register_parameters(posterior)

residuals_mean = y - mean_trend
D_resid_mean = gpx.Dataset(X=x, y=residuals_mean)
latent_dist = p_posterior_mean.predict(x, train_data=D_resid_mean)
pred_mean = latent_dist.mean
pred_std = latent_dist.stddev()

total_mean = mean_trend.flatten() + pred_mean.flatten()
std_flat = pred_std.flatten()

plt.plot(x, total_mean, "b-", label="Posterior Mean")
plt.fill_between(
    x.flatten(),
    total_mean - 2 * std_flat,
    total_mean + 2 * std_flat,
    color="b",
    alpha=0.2,
    label="95% CI (GP Uncertainty)",
)

plt.legend()
plt.show()
