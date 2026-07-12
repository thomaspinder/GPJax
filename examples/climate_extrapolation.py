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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Climate extrapolation with Gaussian processes
#
# This notebook fits a Gaussian process (GP) to the ensemble-mean annual temperature
# through 2014 and extrapolates that fitted trend to 2050. It then compares the GP's
# posterior predictive interval with the spread among the supplied climate-model
# projections. These quantities describe different sources of variation and should not be
# interpreted as interchangeable measures of climate projection uncertainty.
#
# The data contain annual mean 2 m air temperatures for Paris from seven CMIP6 HighResMIP
# simulations over 1950–2050. Values through 2014 come from CMIP6 historical simulations,
# not station observations. The notebook therefore treats their ensemble mean as a
# simulated series for a methodological comparison, not as observed climate history.
#
# > Data: CMIP6 HighResMIP projections via Open-Meteo (CC-BY).

# %%
# Enable Float64 for more stable matrix inversions.
from pathlib import Path

import equinox as eqx
from examples.utils import clean_legend, use_mpl_style
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd
import paramax

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


key = jr.key(42)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Data from seven climate models
#
# The data table records year, annual mean temperature, and model. The next plot draws one
# trajectory per model to show the temporal pattern and the variation among the supplied
# simulations. This variation is descriptive; the plot alone does not identify its physical
# causes.

# %%
data_path = Path("data") / "paris_climate_projection.csv"
if not data_path.exists():
    data_path = Path("examples") / "data" / "paris_climate_projection.csv"

climate = pd.read_csv(data_path)
model_names = sorted(climate["model"].unique())

print(
    f"{len(climate)} rows, {len(model_names)} models, "
    f"years {climate['year'].min()}-{climate['year'].max()}"
)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for model_index, model_name in enumerate(model_names):
    model_rows = climate[climate["model"] == model_name]
    ax.plot(
        model_rows["year"],
        model_rows["annual_mean_temp"],
        color=cols[model_index % len(cols)],
        alpha=0.7,
        linewidth=1.0,
        label=model_name,
    )
ax.set_xlabel("Year")
ax.set_ylabel("Annual mean temperature (°C)")
ax.set_title("CMIP6 HighResMIP simulations for Paris (1950-2050)")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")

# %% [markdown]
# ## Ensemble-mean target
#
# The GP target is the mean temperature across the available models in each year. The code
# also computes the sample standard deviation and the minimum and maximum across models for
# the later comparison. These summaries describe this ensemble and are not probability
# intervals for future temperature.

# %%
by_year = climate.groupby("year")["annual_mean_temp"]
ensemble_mean = by_year.mean()
ensemble_std = by_year.std()
ensemble_min = by_year.min()
ensemble_max = by_year.max()

years = ensemble_mean.index.to_numpy()
ensemble_mean_temp = ensemble_mean.to_numpy()

# %% [markdown]
# ## Historical and forecast periods
#
# CMIP6 historical simulations end in 2014, so the training set contains only the
# ensemble-mean values through that year. Values from 2015 onward do not enter the fit; they
# remain available for comparison with the GP extrapolation.

# %%
historical_boundary = 2014
is_historical = years <= historical_boundary

train_years = years[is_historical]
train_temp = ensemble_mean_temp[is_historical]

# %% [markdown]
# ### Standardise years and centre temperatures
#
# The code standardises the training years to zero mean and unit variance, which expresses
# kernel lengthscales relative to the training period. It centres temperatures at their
# training mean because the GP uses a zero mean function. Under this transformation, the
# linear kernel is centred on the middle of the training years. The stored constants map
# predictions back to calendar years and degrees Celsius.

# %%
year_center = train_years.mean()
year_scale = train_years.std()
temp_center = train_temp.mean()


def standardise_year(raw_years):
    return ((raw_years - year_center) / year_scale).reshape(-1, 1)


standardised_train_x = jnp.asarray(standardise_year(train_years))
centred_train_y = jnp.asarray((train_temp - temp_center).reshape(-1, 1))

D = gpx.Dataset(X=standardised_train_x, y=centred_train_y)

# Test inputs span the full record so we can see fit and extrapolation together.
forecast_years = jnp.linspace(1950.0, 2050.0, 400)
standardised_test_x = jnp.asarray(standardise_year(forecast_years))

# %% [markdown]
# ## Kernel for trend and residual variation
#
# The kernel is the sum of a linear component and a Matérn-5/2 component:
#
# $$k(x, x') = \underbrace{\sigma_\text{lin}^2 \, x\,x'}_{\text{Linear: warming trend}}
#   \; + \; \underbrace{k_\text{Matérn}(x, x')}_{\text{Matérn: residual variability}}.$$
#
# The linear kernel represents a straight-line trend in standardised year. The stationary
# Matérn-5/2 kernel represents correlated deviations around that trend. Because annual means
# contain no within-year observations, the model includes no seasonal component.
#
# A flexible stationary kernel can attribute a gradual rise to a long correlation
# lengthscale and then revert toward the zero prior mean outside the training range. To
# assign longer-term variation to the linear component, the code fixes the Matérn
# lengthscale at six calendar years. It estimates the two kernel variances and Gaussian
# observation noise from the training data.
#
# Far from the training inputs, the Matérn contribution to the posterior mean decays toward
# zero. The linear component continues the fitted slope, and its prior variance increases
# with squared distance from the standardised origin. These assumptions determine both the
# extrapolated mean and its uncertainty; the data do not validate them beyond 2014.

# %%
residual_lengthscale_years = 6.0
residual_lengthscale = residual_lengthscale_years / year_scale

mean_function = gpx.mean_functions.Zero()  # a constant mean, initialised at zero
linear_kernel = gpx.kernels.Linear(variance=1.0)
variability_kernel = gpx.kernels.Matern52(
    lengthscale=residual_lengthscale, variance=0.05
)
trend_kernel = gpx.kernels.SumKernel(kernels=[linear_kernel, variability_kernel])

prior = gpx.gps.Prior(mean_function=mean_function, kernel=trend_kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
posterior = prior * likelihood

# Freeze the Matérn lengthscale so it stays a short-range residual kernel.
posterior = eqx.tree_at(
    lambda model: model.prior.kernel.kernels[1].lengthscale,
    posterior,
    replace_fn=paramax.non_trainable,
)

# %% [markdown]
# ## Fit by marginal likelihood
#
# With a Gaussian likelihood, the latent-function posterior is available in closed form for
# fixed hyperparameters. The code uses AdamW to minimise the negative conjugate marginal log
# likelihood over the trainable kernel and likelihood parameters while holding the Matérn
# lengthscale fixed. The resulting point estimates do not account for hyperparameter
# uncertainty.

# %%
negative_mll = lambda model, data: -gpx.objectives.conjugate_mll(model, data)

print(f"Initial negative MLL: {negative_mll(paramax.unwrap(posterior), D):.3f}")

opt_posterior, history = gpx.fit(
    model=posterior,
    objective=negative_mll,
    train_data=D,
    optim=ox.adamw(learning_rate=1e-2),
    num_iters=1500,
    key=key,
)

# Resolve the frozen wrapper into a plain model we can query for predictions.
resolved_posterior = paramax.unwrap(opt_posterior)
print(f"Optimised negative MLL: {negative_mll(resolved_posterior, D):.3f}")

# %% [markdown]
# The following summary reports the fitted parameters and identifies the fixed Matérn
# lengthscale.

# %%
gpx.summarise(opt_posterior)

# %% [markdown]
# ## Posterior predictive distribution
#
# The code evaluates the latent posterior over 1950–2050, applies the Gaussian likelihood,
# and restores the training temperature mean. The plotted band is the predictive mean plus
# or minus two predictive standard deviations, an approximately 95% pointwise interval
# under the fitted GP. It includes likelihood noise but conditions on the selected kernel,
# the fixed lengthscale, and point-estimated hyperparameters.

# %%
latent_dist = resolved_posterior.predict(
    standardised_test_x, train_data=D, return_covariance_type="diagonal"
)
predictive_dist = resolved_posterior.likelihood(latent_dist)

predictive_mean_temp = predictive_dist.mean + temp_center
predictive_std = jnp.sqrt(predictive_dist.variance)
lower_credible = predictive_mean_temp - 2.0 * predictive_std
upper_credible = predictive_mean_temp + 2.0 * predictive_std

# %% [markdown]
# The next plot shows the fitted historical period and the extrapolation after the 2014
# boundary. Changes in the band width follow from the fitted covariance model. The band
# quantifies posterior predictive uncertainty under that model; this plot does not establish
# empirical coverage or calibration beyond the training period.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.axvline(historical_boundary, color="grey", linestyle=":", linewidth=1)
ax.plot(
    train_years,
    train_temp,
    "x",
    color=cols[0],
    alpha=0.6,
    label="Ensemble mean (historical, ≤2014)",
)
ax.fill_between(
    forecast_years,
    lower_credible,
    upper_credible,
    alpha=0.2,
    color=cols[1],
    label="GP predictive interval (mean ± 2 SD)",
)
ax.plot(forecast_years, predictive_mean_temp, color=cols[1], label="GP predictive mean")
ax.set_xlabel("Year")
ax.set_ylabel("Annual mean temperature (°C)")
ax.set_title("GP fit to historical ensemble mean, extrapolated to 2050")
ax.legend(loc="upper left", fontsize="small")
clean_legend(ax)

# %% [markdown]
# ## Posterior predictive uncertainty and ensemble spread
#
# The forecast-period plot overlays two quantities:
#
# 1. The GP band is an approximately 95% pointwise posterior predictive interval for a new
#    value of the ensemble-mean target. It is conditional on the historical training series,
#    the additive kernel, and point-estimated hyperparameters.
# 2. The climate-model summaries are the yearly minimum-to-maximum envelope and one sample
#    standard deviation on either side of the ensemble mean among the available simulations.
#    They describe disagreement within this finite ensemble.
#
# The second quantity is not a posterior interval, and the supplied models are not treated
# as independent draws from a probability distribution over possible climates. Conversely,
# the GP band contains no representation of models or forcing assumptions absent from its
# training series. Neither quantity alone represents total uncertainty about future climate.

# %%
is_forecast = years >= 2015
forecast_period_years = years[is_forecast]

in_forecast_window = forecast_years >= 2015
plot_years = forecast_years[in_forecast_window]

fig, ax = plt.subplots(figsize=(10, 5))

# GP extrapolation predictive interval.
ax.fill_between(
    plot_years,
    lower_credible[in_forecast_window],
    upper_credible[in_forecast_window],
    alpha=0.25,
    color=cols[1],
    label="GP predictive interval (mean ± 2 SD)",
)
ax.plot(
    plot_years,
    predictive_mean_temp[in_forecast_window],
    color=cols[1],
    label="GP predictive mean",
)

# Finite-ensemble spread.
ax.fill_between(
    forecast_period_years,
    ensemble_min[is_forecast].to_numpy(),
    ensemble_max[is_forecast].to_numpy(),
    alpha=0.15,
    color=cols[3],
    label="Between-model min-max envelope",
)
ax.plot(
    forecast_period_years,
    (ensemble_mean[is_forecast] - ensemble_std[is_forecast]).to_numpy(),
    linestyle="--",
    linewidth=1,
    color=cols[3],
    label="Between-model ±1 std",
)
ax.plot(
    forecast_period_years,
    (ensemble_mean[is_forecast] + ensemble_std[is_forecast]).to_numpy(),
    linestyle="--",
    linewidth=1,
    color=cols[3],
)
ax.set_xlabel("Year")
ax.set_ylabel("Annual mean temperature (°C)")
ax.set_title("GP extrapolation uncertainty vs. between-model spread (2015-2050)")
ax.legend(loc="upper left", fontsize="small")
clean_legend(ax)

# %% [markdown]
# The comparison assesses how a continuation of the pre-2015 ensemble-mean trend relates to
# the supplied post-2014 simulations. A difference between the GP mean and the later
# ensemble reflects the GP's extrapolation assumptions and the information available before
# 2015; it does not by itself establish model failure or forecast calibration.
#
# The GP interval describes conditional predictive variation around one fitted trend. The
# across-model standard deviation and range describe spread among the available simulations
# under their simulation configurations. This spread does not isolate individual sources of
# uncertainty, assign probabilities to the models, or cover outcomes outside the ensemble.
# The printed 2050 values provide a numerical comparison of the GP interval and the
# available-model range at one forecast horizon.

# %%
forecast_2050_index = int(jnp.argmin(jnp.abs(forecast_years - 2050.0)))
gp_2050_mean = float(predictive_mean_temp[forecast_2050_index])
gp_2050_lower = float(lower_credible[forecast_2050_index])
gp_2050_upper = float(upper_credible[forecast_2050_index])

ensemble_2050 = climate[climate["year"] == 2050]["annual_mean_temp"]
print(
    f"GP 2050 forecast:         {gp_2050_mean:.2f} °C "
    f"[{gp_2050_lower:.2f}, {gp_2050_upper:.2f}] (predictive mean ± 2 SD)"
)
print(
    f"Between-model 2050 range: "
    f"[{ensemble_2050.min():.2f}, {ensemble_2050.max():.2f}] °C "
    f"(mean {ensemble_2050.mean():.2f})"
)

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'GPJax contributors'
