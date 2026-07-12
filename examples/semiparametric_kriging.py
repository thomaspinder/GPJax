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
# # Semiparametric kriging
#
# This notebook models centred daily maximum temperature as the sum of a linear elevation
# component and a spatial Gaussian process (GP) residual. GPJax defines the GP prior and Gaussian
# likelihood. NumPyro uses the No-U-Turn Sampler (NUTS) to sample the linear coefficients, kernel
# hyperparameters, and observation-noise scale jointly.

# %% [markdown]
# ## Swiss temperature data
#
# The data contain daily maximum temperature, `t_max`, at 152 MeteoSwiss stations on
# 2023-04-13. Each record also contains longitude and latitude in degrees and elevation in metres.
# The preprocessing below removes the two records with missing `t_max` values.
#
# The model represents the association between elevation and temperature with a linear term. A
# GP over longitude and latitude represents residual spatial dependence after accounting for that
# term. This decomposition separates a global elevation gradient from spatial variation without
# assigning the residual variation to specific physical causes.
#
# > Data: MeteoSwiss station observations via Open-Meteo (CC-BY).

# %%
import json

from examples.utils import use_mpl_style
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
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
# The next cell removes the two missing temperature records, leaving 150 stations. It standardises
# longitude, latitude, and elevation by subtracting each variable's mean and dividing by its
# standard deviation. The target is centred by subtracting its mean but remains in degrees Celsius.
# The dimensionless coordinates put the spatial inputs on comparable numerical scales. The saved
# elevation standard deviation and target mean convert the fitted slope and predictions back to
# physical units.

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
# ## Linear component
#
# The baseline model regresses centred maximum temperature, $\mathbf{y}$, on standardised
# elevation, $\mathbf{z}_{\text{elev}}$. The slope $w$ has units of degrees Celsius per elevation
# standard deviation. The intercept $b$ is the expected centred temperature at mean elevation,
# and $\sigma$ is the observation standard deviation in degrees Celsius.
#
# $$\begin{aligned} w &\sim \mathcal{N}(0, 5) \\
# b &\sim \mathcal{N}(0, 5) \\
# \sigma &\sim \text{LogNormal}(0, 1) \\
# \mathbf{y} &\sim \mathcal{N}(w\,\mathbf{z}_{\text{elev}} + b, \sigma^2 \mathbf{I}) \end{aligned} $$
#
# NUTS draws posterior samples of $w$, $b$, and $\sigma$. This model provides the elevation-only
# reference for the later in-sample comparison.


# %%
def linear_model(elevation_covariate, target=None):
    slope = numpyro.sample("slope", dist.Normal(0.0, 5.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))
    obs_noise = numpyro.sample("obs_noise", dist.LogNormal(0.0, 1.0))

    mu = elevation_covariate * slope + intercept
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, obs_noise), obs=target)


nuts_kernel_lin = NUTS(linear_model)
mcmc_lin = MCMC(nuts_kernel_lin, num_warmup=500, num_samples=1000, num_chains=1)
mcmc_lin.run(keys[2], elevation_std, target_centered)
mcmc_lin.print_summary()

# %% [markdown]
# ## Combining the linear component with a spatial GP
#
# The semiparametric model adds a latent spatial residual to the linear elevation component:
# $$ y(\mathbf{s}) = \underbrace{w\,z_{\text{elev}} + b}_{\text{Linear Mean (elevation)}} +
# \underbrace{f(\mathbf{s})}_{\text{Spatial GP Residual}} + \epsilon $$
#
# Here, $\mathbf{s} = (\text{longitude}, \text{latitude})$, $z_{\text{elev}}$ is standardised
# elevation, $f(\mathbf{s})$ is a zero-mean GP, and $\epsilon$ is independent Gaussian observation
# noise with standard deviation $\sigma$.
#
# ### GPJax and NumPyro integration
#
# GPJax assigns $f$ a Matérn-3/2 covariance over the two standardised spatial coordinates. NumPyro
# samples the shared length scale, marginal variance, and observation-noise scale. For each set of
# sampled parameters, `gpx.objectives.conjugate_mll` evaluates the exact Gaussian marginal
# log-likelihood of the elevation-adjusted residuals after integrating out their latent GP values.
# `numpyro.factor` adds this log-likelihood to the NumPyro model.


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
mcmc_joint = MCMC(nuts_kernel_joint, num_warmup=500, num_samples=1000, num_chains=1)
mcmc_joint.run(keys[3], spatial_coords, elevation_std, target_centered)
mcmc_joint.print_summary()

# %% [markdown]
# ## Elevation coefficient as a lapse rate
#
# Because `t_max` is measured in degrees Celsius and elevation is standardised, `slope` has units
# of degrees Celsius per elevation standard deviation. If $s_z$ is the elevation standard
# deviation in metres, the physical slope is $w/s_z$ °C/m. The code multiplies this slope by
# $-1000$ to report the temperature decrease per kilometre as a positive lapse rate when $w<0$.
# It applies the same conversion to every posterior draw and reports the 2.5th and 97.5th
# percentiles as a 95% posterior credible interval.

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
    f"(95% credible interval: {lapse_rate_low:.2f}–{lapse_rate_high:.2f})"
)

# %% [markdown]
# ## Model comparison
#
# The next cell computes posterior mean fitted values at the observed stations for both models.
# It then reports root mean squared error (RMSE) in degrees Celsius against the centred
# observations. These are in-sample errors: they compare fit at the training locations and do not
# estimate performance at unobserved stations.

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
# ## Kriging map
#
# The map evaluates the joint model on a 35 by 35 longitude-latitude grid spanning Switzerland's
# bounding box. Grid elevation is set to standardised zero, which is the mean station elevation,
# so the mapped surface represents latent maximum temperature at that reference elevation. The
# predictions are converted to degrees Celsius by restoring the target mean and clipped to the
# national boundary.
#
# The left panel shows the posterior mean of the latent temperature surface. Station colours show
# observed `t_max` adjusted to mean elevation with the posterior mean slope; observation noise
# means the fitted surface need not interpolate these adjusted values exactly. The right panel
# shows the posterior standard deviation of the latent surface in degrees Celsius. It measures
# uncertainty across posterior GP and parameter draws but excludes independent observation noise.
# Station markers in that panel indicate observation locations.
#
# > Country outlines: Natural Earth (1:50m admin-0), public domain.

# %%
# Load the bundled country outlines (Natural Earth 1:50m), resolving the path from
# the repo root or the docs build directory exactly as for the station data above.
try:
    with open("examples/data/switzerland_neighbours.geojson") as boundary_file:
        boundaries = json.load(boundary_file)
except FileNotFoundError:
    with open("data/switzerland_neighbours.geojson") as boundary_file:
        boundaries = json.load(boundary_file)


def _iter_polygons(geometry):
    """Yield each polygon (a list of rings) from a Polygon/MultiPolygon geometry."""
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        yield from geometry["coordinates"]


def country_path(iso_a3):
    """Build a matplotlib Path (compound, for multi-part borders) for one country."""
    feature = next(
        f for f in boundaries["features"] if f["properties"]["iso_a3"] == iso_a3
    )
    vertices, codes = [], []
    for polygon in _iter_polygons(feature["geometry"]):
        for ring in polygon:
            vertices.extend(ring)
            vertices.append(ring[0])  # close the ring
            codes += [MplPath.MOVETO, *([MplPath.LINETO] * (len(ring) - 1)), MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


switzerland_path = country_path("CHE")
neighbour_codes = ["FRA", "ITA", "DEU", "AUT", "LIE"]
(lon_min, lat_min), (lon_max, lat_max) = switzerland_path.get_extents().get_points()

# %%
# Predict over a grid spanning Switzerland's bounding box (so the clipped field fills
# the whole country), holding elevation at the dataset mean (standardised zero).
n_grid = 35
lon_grid = jnp.linspace(lon_min, lon_max, n_grid)
lat_grid = jnp.linspace(lat_min, lat_max, n_grid)
LON, LAT = jnp.meshgrid(lon_grid, lat_grid)

spatial_grid = jnp.column_stack(
    [
        (LON.ravel() - jnp.mean(longitude)) / jnp.std(longitude),
        (LAT.ravel() - jnp.mean(latitude)) / jnp.std(latitude),
    ]
)
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

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
# Latitude-corrected aspect so the map is not horizontally stretched.
aspect = 1.0 / float(jnp.cos(jnp.deg2rad((lat_min + lat_max) / 2.0)))
margin = 0.2


def draw_borders(ax):
    for iso in neighbour_codes:
        ax.add_patch(
            PathPatch(
                country_path(iso),
                facecolor="none",
                edgecolor="0.6",
                linewidth=0.6,
                zorder=3,
            )
        )
    ax.add_patch(
        PathPatch(
            switzerland_path,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
    )


# Adjust each station's observed t_max to the common reference elevation by
# removing the fitted elevation effect (slope x standardised elevation). A shared
# colour scale permits direct comparison with the fitted latent surface.
stations_adjusted = max_temperature - slope_mean_std_units * elevation_std
colour_min = float(jnp.minimum(grid_mean.min(), stations_adjusted.min()))
colour_max = float(jnp.maximum(grid_mean.max(), stations_adjusted.max()))
temp_levels = jnp.linspace(colour_min, colour_max, 21)

mean_map = axes[0].contourf(LON, LAT, grid_mean, levels=temp_levels, cmap="magma")
mean_map.set_clip_path(switzerland_path, transform=axes[0].transData)
draw_borders(axes[0])
axes[0].scatter(
    longitude,
    latitude,
    c=stations_adjusted,
    cmap="magma",
    vmin=colour_min,
    vmax=colour_max,
    marker="o",
    edgecolor="white",
    linewidth=0.4,
    s=28,
    zorder=5,
)
axes[0].set_title("Posterior mean t_max at mean elevation (°C)")
fig.colorbar(mean_map, ax=axes[0])

std_map = axes[1].contourf(LON, LAT, grid_std, levels=20, cmap="viridis")
std_map.set_clip_path(switzerland_path, transform=axes[1].transData)
draw_borders(axes[1])
axes[1].scatter(longitude, latitude, c=cols[0], s=10, alpha=0.7, zorder=5)
axes[1].set_title("Latent-surface posterior standard deviation (°C)")
fig.colorbar(std_map, ax=axes[1])

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min - margin, lon_max + margin)
    ax.set_ylim(lat_min - margin, lat_max + margin)
    ax.set_aspect(aspect)

# %% [markdown]
# The posterior mean panel estimates temperature at a common elevation, so it displays the fitted
# spatial component together with the intercept rather than the effect of local terrain height.
# The posterior standard deviation panel quantifies uncertainty in that latent surface. Its values
# depend on station locations and the fitted covariance parameters; they are distinct from the
# posterior mean temperature and from the observation-noise standard deviation.

# %% [markdown]
# ## System configuration
#
# The final cell records the Python and package versions used to run the notebook.

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder"
