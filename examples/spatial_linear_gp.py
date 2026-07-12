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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spatial Modelling with Composable Gaussian Processes
#
# This notebook shows how to construct a semiparametric spatial model by composing a linear
# mean function in NumPyro with a GPJax Gaussian Process (GP) residual. We build two
# components: firstly, a linear component that encodes a physically meaningful global trend
# with Bayesian linear regression. We then define a GP residual component responsible for
# capturing the smooth spatial structure that the linear term leaves behind.
#
# The example highlights the interplay between **GPJax** and **NumPyro**: `GPJax` provides the
# GP prior and likelihood definitions, while `NumPyro` performs Hamiltonian Monte Carlo (HMC)
# inference across all parameters in a unified model and allows us to draw upon a broader set of
# modelling components.

# %% [markdown]
# ## The Swiss Temperature Data
#
# We work with a real spatial climate snapshot: the daily maximum temperature `t_max` recorded
# at 152 MeteoSwiss weather stations on 2023-04-13. Each station reports its `longitude`,
# `latitude` and `elevation` (in metres). The physics of this dataset makes it a near-perfect
# fit for a linear-mean plus spatial-GP decomposition.
#
# The dominant driver of near-surface temperature over complex terrain is **altitude**: the air
# cools as you climb, following the *environmental lapse rate* of roughly $-6.5\,^{\circ}$C per
# kilometre. That is a global, essentially linear, relationship between `elevation` and `t_max`,
# and it is exactly the kind of trend a linear mean function is designed to absorb. What remains
# after removing the elevation effect is a smoothly varying spatial field — the influence of
# latitude, large-scale weather systems, lakes and the sheltering of Alpine valleys — which is
# naturally modelled as a GP over the `(longitude, latitude)` plane. Composing the two lets each
# mechanism explain the part of the signal it is suited to.
#
# > Data: MeteoSwiss station observations via Open-Meteo (CC-BY).

# %%
from examples.utils import use_mpl_style
import gpjax as gpx
import jax
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
import pandas as pd

jax.config.update("jax_enable_x64", True)

use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.key(123)
keys = jr.split(key, 8)

# %% [markdown]
# ### Loading and standardising
#
# We drop the handful of stations with a missing reading, then standardise the spatial inputs
# (`longitude`, `latitude`) and the linear covariate (`elevation`), and centre the target
# `t_max`. Standardisation keeps the kernel and linear-model parameters on a comparable scale,
# which greatly improves the conditioning of the GP and the geometry seen by the sampler. We
# retain the raw means and standard deviations so that predictions and the learned coefficients
# can be mapped back to physical units.

# %%
# Resolve the data file from the repo root (``examples/data``) or from the docs
# build, which executes notebooks with ``docs/_examples`` as the working directory
# and copies the data alongside as ``data``.
try:
    stations = pd.read_csv(
        "examples/data/max_tempeature_switzerland.csv", index_col=0
    )
except FileNotFoundError:
    stations = pd.read_csv(
        "data/max_tempeature_switzerland.csv", index_col=0
    )
stations = stations.dropna(subset=["t_max"])

longitude = jnp.asarray(stations["longitude"].to_numpy())
latitude = jnp.asarray(stations["latitude"].to_numpy())
elevation = jnp.asarray(stations["elevation"].to_numpy())
max_temperature = jnp.asarray(stations["t_max"].to_numpy())

num_stations = max_temperature.shape[0]


def standardise(values):
    return (values - jnp.mean(values)) / jnp.std(values)


spatial_coords = jnp.column_stack([standardise(longitude), standardise(latitude)])
elevation_std = standardise(elevation)
elevation_scale_m = jnp.std(elevation)

target_mean_celsius = jnp.mean(max_temperature)
target_centered = max_temperature - target_mean_celsius

print(f"Retained {num_stations} stations with a valid t_max reading.")

# %% [markdown]
# ## Linear Component
#
# We begin by defining a Bayesian linear regression of the (centred) maximum temperature on
# standardised elevation in NumPyro. This component will later be combined with a GP residual,
# but for now it establishes a baseline that captures only the altitude effect.
#
# $$\begin{aligned} w &\sim \mathcal{N}(0, 5) \\
# b &\sim \mathcal{N}(0, 5) \\
# \sigma &\sim \text{LogNormal}(0, 1) \\
# \mathbf{y} &\sim \mathcal{N}(w\,\mathbf{z}_{\text{elev}} + b, \sigma^2 \mathbf{I}) \end{aligned} $$
#
# We use the No-U-Turn Sampler (NUTS) to draw samples from the posterior distributions of the
# elevation slope $w$, intercept $b$, and noise $\sigma$.


# %%
def linear_model(elevation_covariate, target=None):
    slope = numpyro.sample("slope", dist.Normal(0.0, 5.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))
    obs_noise = numpyro.sample("obs_noise", dist.LogNormal(0.0, 1.0))

    mu = elevation_covariate * slope + intercept
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, obs_noise), obs=target)


nuts_kernel_lin = NUTS(linear_model)
mcmc_lin = MCMC(nuts_kernel_lin, num_warmup=1000, num_samples=1500, num_chains=1)
mcmc_lin.run(keys[2], elevation_std, target_centered)
mcmc_lin.print_summary()

# %% [markdown]
# ## Composing the Linear Component with a Spatial GP
#
# We now augment the linear component with a GP tasked with modelling the residual over space.
#
# $$ y(\mathbf{s}) = \underbrace{w\,z_{\text{elev}} + b}_{\text{Linear Mean (elevation)}} +
# \underbrace{f(\mathbf{s})}_{\text{Spatial GP Residual}} + \epsilon $$
#
# where $\mathbf{s} = (\text{longitude}, \text{latitude})$ is the spatial location.
#
# ### GPJax and NumPyro Integration
#
# We define the GP prior in `GPJax` using a second-order Matérn kernel over the two spatial
# dimensions and a constant mean function (since the elevation trend is handled explicitly by
# the linear term). Hyperparameters are sampled directly with ``numpyro.sample`` and passed to
# the GPJax constructors as raw JAX arrays. We then compute the exact marginal log-likelihood
# (MLL) of the elevation-adjusted residuals under the GP prior using
# `gpx.objectives.conjugate_mll`. This closed-form conjugate term is added to the potential
# function via `numpyro.factor`, guiding the sampler.


# %%
def joint_model(
    spatial_locations,
    elevation_covariate,
    target=None,
    spatial_locations_new=None,
    elevation_covariate_new=None,
):
    slope = numpyro.sample("slope", dist.Normal(0.0, 5.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))

    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    obs_noise = numpyro.sample("obs_noise", dist.LogNormal(0.0, 1.0))

    kernel = gpx.kernels.Matern32(
        active_dims=[0, 1], lengthscale=lengthscale, variance=variance
    )
    meanf = gpx.mean_functions.Constant()
    gp_prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=num_stations, obs_stddev=obs_noise
    )
    gp_posterior = gp_prior * likelihood

    trend = elevation_covariate * slope + intercept

    if target is not None:
        residuals = (target - trend).reshape(-1, 1)
        residual_data = gpx.Dataset(X=spatial_locations, y=residuals)
        mll = gpx.objectives.conjugate_mll(gp_posterior, residual_data)
        numpyro.factor("gp_log_lik", mll)

    if spatial_locations_new is not None and target is not None:
        residuals = (target - trend).reshape(-1, 1)
        residual_data = gpx.Dataset(X=spatial_locations, y=residuals)

        latent_dist = gp_posterior.predict(
            spatial_locations_new, train_data=residual_data
        )
        f_new = numpyro.sample("f_new", latent_dist).reshape(-1, 1)

        trend_new = (elevation_covariate_new * slope + intercept).reshape(-1, 1)
        numpyro.deterministic("y_pred", trend_new + f_new)


nuts_kernel_joint = NUTS(joint_model)
# In practice, one should run more samples from multiple chains.
mcmc_joint = MCMC(nuts_kernel_joint, num_warmup=1000, num_samples=1500, num_chains=1)
mcmc_joint.run(keys[3], spatial_coords, elevation_std, target_centered)
mcmc_joint.print_summary()

# %% [markdown]
# ## The Elevation Coefficient as a Lapse Rate
#
# The `slope` parameter is the effect of standardised elevation on temperature. Dividing by the
# elevation standard deviation converts it to $^{\circ}$C per metre; multiplying by $1000$ and
# flipping the sign gives the implied **environmental lapse rate** in $^{\circ}$C per kilometre.
# We expect a value close to the textbook $6.5\,^{\circ}$C/km.

# %%
samples_joint = mcmc_joint.get_samples()
slope_posterior = samples_joint["slope"]
slope_mean_std_units = jnp.mean(slope_posterior)

slope_per_metre = slope_mean_std_units / elevation_scale_m
lapse_rate_per_km = -slope_per_metre * 1000.0
lapse_rate_samples = -(slope_posterior / elevation_scale_m) * 1000.0
lapse_rate_low, lapse_rate_high = jnp.percentile(
    lapse_rate_samples, jnp.array([2.5, 97.5])
)

print("\nLearned elevation effect:")
print(f"  slope (standardised elevation): {slope_mean_std_units:.3f} °C / std")
print(f"  slope (physical):               {slope_per_metre * 1000:.3f} °C / km")
print(
    f"  implied lapse rate:             {lapse_rate_per_km:.2f} °C/km "
    f"(95% CI: {lapse_rate_low:.2f}–{lapse_rate_high:.2f})"
)

# %% [markdown]
# ## Comparison
#
# We evaluate the elevation-only linear model in isolation, then the joint model where a spatial
# GP has been added to model the residual. Predicting back at the station locations, the joint
# model should track the observations more closely because it captures spatial structure that a
# single altitude gradient cannot.

# %%
samples_lin = mcmc_lin.get_samples()
predictive_lin = Predictive(linear_model, samples_lin, return_sites=["mu"])
mean_pred_lin = jnp.mean(
    predictive_lin(keys[4], elevation_covariate=elevation_std)["mu"], axis=0
)

predictive_joint = Predictive(joint_model, samples_joint, return_sites=["y_pred"])
preds_joint = predictive_joint(
    keys[5],
    spatial_locations=spatial_coords,
    elevation_covariate=elevation_std,
    target=target_centered,
    spatial_locations_new=spatial_coords,
    elevation_covariate_new=elevation_std,
)["y_pred"]
mean_pred_joint = jnp.mean(preds_joint, axis=0).flatten()

rmse_lin = jnp.sqrt(jnp.mean((mean_pred_lin - target_centered) ** 2))
rmse_joint = jnp.sqrt(jnp.mean((mean_pred_joint - target_centered) ** 2))

print("\nIn-sample RMSE (°C) against observed t_max:")
print(f"  Linear (elevation) model: {rmse_lin:.4f}")
print(f"  Joint (linear + GP) model: {rmse_joint:.4f}")

# %% [markdown]
# ## Kriging Map
#
# Finally we produce the interpretable output the method is built for: a **kriging map** of
# maximum temperature across Switzerland. We predict on a dense `(longitude, latitude)` grid,
# holding elevation fixed at the dataset mean so that the map isolates the *spatial* field —
# what the temperature would look like across the country at a common reference altitude. The
# left panel shows the posterior mean, the right panel the posterior standard deviation
# (uncertainty grows away from the stations). Station locations are overlaid, coloured by their
# observed `t_max`.

# %%
n_grid = 45
lon_grid = jnp.linspace(longitude.min(), longitude.max(), n_grid)
lat_grid = jnp.linspace(latitude.min(), latitude.max(), n_grid)
LON, LAT = jnp.meshgrid(lon_grid, lat_grid)

spatial_grid = jnp.column_stack(
    [
        (LON.ravel() - jnp.mean(longitude)) / jnp.std(longitude),
        (LAT.ravel() - jnp.mean(latitude)) / jnp.std(latitude),
    ]
)
# Reference elevation = dataset mean, i.e. standardised elevation of zero.
elevation_grid = jnp.zeros(spatial_grid.shape[0])

# Thin the posterior for the (more expensive) grid prediction.
thinned_samples = {name: value[::5] for name, value in samples_joint.items()}
predictive_grid = Predictive(joint_model, thinned_samples, return_sites=["y_pred"])
preds_grid = predictive_grid(
    keys[6],
    spatial_locations=spatial_coords,
    elevation_covariate=elevation_std,
    target=target_centered,
    spatial_locations_new=spatial_grid,
    elevation_covariate_new=elevation_grid,
)["y_pred"]

# Back to physical °C by adding the target mean that we removed when centring.
grid_mean = (jnp.mean(preds_grid, axis=0).flatten() + target_mean_celsius).reshape(
    n_grid, n_grid
)
grid_std = jnp.std(preds_grid, axis=0).flatten().reshape(n_grid, n_grid)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

mean_map = axes[0].contourf(LON, LAT, grid_mean, levels=20, cmap="magma")
axes[0].scatter(
    longitude,
    latitude,
    c=max_temperature,
    cmap="magma",
    marker="o",
    edgecolor="white",
    linewidth=0.4,
    s=28,
)
axes[0].set_title("Posterior mean t_max at mean elevation (°C)")
fig.colorbar(mean_map, ax=axes[0])

std_map = axes[1].contourf(LON, LAT, grid_std, levels=20, cmap="viridis")
axes[1].scatter(longitude, latitude, c=cols[0], s=10, alpha=0.7)
axes[1].set_title("Posterior standard deviation (°C)")
fig.colorbar(std_map, ax=axes[1])

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

# %% [markdown]
# The mean map recovers the familiar Swiss pattern — warm on the low-lying Plateau and in the
# southern Ticino, cool over the high Alps — while the standard-deviation map shows uncertainty
# tightening around the dense station network and widening towards the sparsely sampled
# margins. The elevation slope and the spatial residual together tell a coherent physical story:
# altitude sets the baseline, and the GP fills in the regional climate.

# %% [markdown]
# ## System configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder"
