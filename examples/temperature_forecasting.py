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
# # Forecasting a seasonal time series with a composite kernel
#
# This notebook forecasts daily mean temperature in Reykjavík, Iceland. A stationary RBF
# kernel with a short fitted lengthscale loses covariance with observations as the forecast
# horizon increases, so its posterior mean approaches the prior mean. To represent the
# observed annual structure, the model instead uses three covariance components:
#
# - a periodic kernel with its period fixed at one year for annual dependence;
# - a long-lengthscale Matérn-5/2 kernel multiplied by the periodic kernel, which reduces
#   covariance between corresponding seasons as their separation in years increases;
# - a short-lengthscale Matérn-3/2 kernel added to represent non-periodic, temporally
#   correlated deviations.
#
# The notebook prepares the data, fits the free kernel and likelihood parameters by
# maximising the conjugate marginal log-likelihood, forecasts a held-out interval, and
# compares the result with an RBF model fitted to the same training data. See the
# introductory [regression notebook](https://docs.jaxgaussianprocesses.com/_examples/regression/)
# for a single-kernel example.
#
# Data: ERA5 reanalysis via [Open-Meteo](https://open-meteo.com/) (CC-BY).

# %%
# Enable Float64 for more stable matrix inversions.
from pathlib import Path

from examples.utils import clean_legend, use_mpl_style
from jax import config
import jax.numpy as jnp
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

import paramax

# Set the default style for plotting.
use_mpl_style()

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Prepare the data
#
# The file `data/reykjavik_daily_temperature.csv` contains daily mean 2-metre air
# temperature for Reykjavík from 1 January 2020 through 31 December 2023. The next cell
# parses the dates and expresses elapsed time in years from the first observation, using
# 365.25 days per year. Keeping this unit makes a kernel period of `1.0` correspond to one
# year.

# %%
data_path = Path("data") / "reykjavik_daily_temperature.csv"
if not data_path.exists():
    data_path = Path("examples") / "data" / "reykjavik_daily_temperature.csv"

raw_frame = pd.read_csv(data_path, parse_dates=["date"])

start_date = raw_frame["date"].min()
time_in_years = (raw_frame["date"] - start_date).dt.days.to_numpy() / 365.25
temperature_celsius = raw_frame["temperature_2m_mean"].to_numpy()

print(
    f"{len(raw_frame)} daily records from {start_date.date()} to "
    f"{raw_frame['date'].max().date()}"
)
print(
    f"Temperature range: {temperature_celsius.min():.1f} to "
    f"{temperature_celsius.max():.1f} °C"
)

# %% [markdown]
# The final four months, 1 September through 31 December 2023, form the held-out forecast
# interval. The next cell centres all temperatures with the mean computed from observations
# through 31 August 2023. Excluding held-out values from this mean prevents leakage into
# model fitting.

# %%
forecast_window_years = 4 / 12  # final four months held out for forecasting
split_time = time_in_years.max() - forecast_window_years

is_train = time_in_years <= split_time
is_holdout = ~is_train

# Centre using the training mean only.
training_mean_celsius = temperature_celsius[is_train].mean()
centred_temperature = temperature_celsius - training_mean_celsius

# %% [markdown]
# Exact Gaussian process inference factorises the training Gram matrix. The next cell keeps
# every third training observation to reduce this cost, while retaining all 122 daily
# observations in the held-out interval for evaluation. It then constructs the GPJax
# training dataset from the thinned inputs and centred temperatures.

# %%
train_time_full = time_in_years[is_train]
train_temp_full = centred_temperature[is_train]

thin_every = 3
train_time = train_time_full[::thin_every].reshape(-1, 1)
train_temp = train_temp_full[::thin_every].reshape(-1, 1)

holdout_time = time_in_years[is_holdout].reshape(-1, 1)
holdout_temp = centred_temperature[is_holdout].reshape(-1, 1)

dataset = gpx.Dataset(X=jnp.asarray(train_time), y=jnp.asarray(train_temp))

print(f"Training points (thinned): {dataset.n}")
print(f"Held-out points (daily):   {holdout_time.shape[0]}")

# %% [markdown]
# The plot shows the thinned training observations and the daily held-out observations on
# the centred temperature scale. The dashed line separates observations available for
# fitting from the held-out interval.

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(train_time, train_temp, ".", color=cols[0], alpha=0.6, label="Training data")
ax.plot(holdout_time, holdout_temp, ".", color="grey", alpha=0.6, label="Held-out data")
ax.axvline(split_time, linestyle="--", color="black", linewidth=1)
ax.set_xlabel("Time (years since 2020-01-01)")
ax.set_ylabel("Centred temperature (°C)")
ax.legend(loc="best")
clean_legend(ax)

# %% [markdown]
# ## Design the kernel
#
# The composite kernel specifies separate covariance patterns for annual and shorter-term
# variation. GPJax's `Periodic` kernel has the form
#
# $$k(x, x') = \sigma^2 \exp\!\left(-\frac{1}{2}\left(\frac{\sin(\pi (x - x') / p)}{\ell}\right)^2\right),$$
#
# where $p$ is the period and $\ell$ controls smoothness within each period. The next cell
# fixes $p=1$ year with `paramax.NonTrainable`, so optimisation cannot change the specified
# annual period. The periodic variance and lengthscale remain trainable.

# %%
annual_period = paramax.NonTrainable(jnp.array(1.0))
annual_cycle = gpx.kernels.Periodic(lengthscale=1.0, variance=1.0, period=annual_period)

# %% [markdown]
# Multiplying the periodic kernel by a Matérn-5/2 kernel yields a locally periodic
# covariance. The periodic factor relates observations at similar phases of the annual
# cycle; the Matérn factor reduces that relation as the absolute time separation grows.
# The next cell initialises the Matérn lengthscale at three years.

# %%
seasonal_envelope = gpx.kernels.Matern52(lengthscale=3.0, variance=1.0)
locally_periodic = annual_cycle * seasonal_envelope

# %% [markdown]
# The added Matérn-3/2 component represents non-periodic temporal correlation in deviations
# from the locally periodic component. Its initial lengthscale is `0.05` years, about 18
# days. Under the additive-kernel interpretation, the two terms are independent latent GP
# contributions whose covariances sum.

# %%
weather_residual = gpx.kernels.Matern32(lengthscale=0.05, variance=1.0)

composite_kernel = locally_periodic + weather_residual

# %% [markdown]
# The next cell combines the composite covariance with a zero-mean prior for the centred
# temperatures. A Gaussian likelihood assumes independent Gaussian observation noise and
# gives a conjugate posterior.

# %%
mean_function = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=mean_function, kernel=composite_kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
posterior = prior * likelihood

# %% [markdown]
# ## Fit the hyperparameters
#
# The code minimises the negative conjugate marginal log-likelihood with SciPy's L-BFGS-B
# implementation. The annual period remains fixed; the periodic, envelope, and residual
# lengthscales, their variance parameters, and the likelihood noise scale are trainable.
# GPJax applies parameter transformations that keep scales and variances positive. Within
# the product kernel, only the product of the two variance parameters determines its
# covariance amplitude, so those two values are not separately identifiable.

# %%
negative_mll = lambda model, data: -gpx.objectives.conjugate_mll(model, data)

print(f"Initial negative MLL: {negative_mll(posterior, dataset):.2f}")

opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    objective=negative_mll,
    train_data=dataset,
)

print(f"Optimised negative MLL: {negative_mll(opt_posterior, dataset):.2f}")

# %% [markdown]
# The next cell extracts and prints the fitted lengthscales, fixed period, and observation
# noise scale. These estimates describe the marginal-likelihood optimum returned by this
# fit; they do not by themselves establish the underlying physical time scales.

# %%
fitted_seasonal = opt_posterior.prior.kernel.kernels[0]  # Periodic * Matern52
fitted_periodic = fitted_seasonal.kernels[0]
fitted_envelope = fitted_seasonal.kernels[1]
fitted_weather = opt_posterior.prior.kernel.kernels[1]  # Matern32

print(f"Annual period (fixed):        {fitted_periodic.period.unwrap():.3f} years")
print(f"Seasonal lengthscale:         {fitted_periodic.lengthscale.unwrap():.3f}")
print(f"Drift lengthscale (Matern52): {fitted_envelope.lengthscale.unwrap():.2f} years")
print(
    f"Weather lengthscale (M32):    {fitted_weather.lengthscale.unwrap():.4f} years "
    f"(~{fitted_weather.lengthscale.unwrap() * 365.25:.0f} days)"
)
print(
    f"Observation noise std:        "
    f"{jnp.sqrt(opt_posterior.likelihood.obs_stddev.unwrap() ** 2):.3f} °C"
)

# %% [markdown]
# The printed period should remain exactly one year because it was excluded from
# optimisation. The fitted lengthscales indicate how this model allocates covariance among
# its components for this four-year record. In particular, an envelope lengthscale longer
# than the record makes corresponding seasons strongly correlated over the observed span,
# but it is weak evidence about longer-term changes.
#
# ## Forecast the held-out interval
#
# The next cell evaluates the fitted posterior on a 400-point grid. The grid begins three
# months before the training boundary, covers the held-out interval from 1 September
# through 31 December 2023, and extends approximately two months beyond the final
# observation. Applying the likelihood produces the posterior predictive mean and variance,
# including observation noise.

# %%
extrapolation_years = 2 / 12
forecast_grid = jnp.linspace(
    split_time - 0.25,
    time_in_years.max() + extrapolation_years,
    400,
).reshape(-1, 1)

latent_forecast = opt_posterior.predict(
    forecast_grid, train_data=dataset, return_covariance_type="diagonal"
)
predictive_forecast = opt_posterior.likelihood(latent_forecast)

forecast_mean = predictive_forecast.mean
forecast_std = jnp.sqrt(predictive_forecast.variance)

# %% [markdown]
# The plot restores degrees Celsius by adding the training mean. The dashed line marks the
# start of the held-out interval. Observations exist from that line through 31 December
# 2023; the remaining grid through approximately the end of February 2024 is extrapolation
# beyond the recorded data.


# %%
def to_celsius(values):
    return values + training_mean_celsius


fig, ax = plt.subplots(figsize=(9.0, 3.5))
ax.plot(
    train_time,
    to_celsius(train_temp),
    ".",
    color=cols[0],
    alpha=0.5,
    label="Training data",
)
ax.plot(
    holdout_time,
    to_celsius(holdout_temp),
    ".",
    color="grey",
    alpha=0.7,
    label="Held-out ERA5 values",
)
ax.fill_between(
    forecast_grid.squeeze(),
    to_celsius(forecast_mean - 1.96 * forecast_std),
    to_celsius(forecast_mean + 1.96 * forecast_std),
    alpha=0.2,
    color=cols[1],
    label="95% predictive interval",
)
ax.plot(forecast_grid, to_celsius(forecast_mean), color=cols[1], label="Forecast mean")
ax.axvline(split_time, linestyle="--", color="black", linewidth=1)
ax.set_xlim(split_time - 0.25, forecast_grid.max())
ax.set_xlabel("Time (years since 2020-01-01)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="lower left")
clean_legend(ax)

# %% [markdown]
# The plot permits separate interpretation of the two forecast regions. Agreement with the
# grey points concerns the held-out interval only. Beyond the last grey point, the periodic
# component maintains annual covariance structure, but no observations are available here
# to assess forecast accuracy.
#
# ## Quantify the held-out forecast
#
# The next cell evaluates the predictive distribution at all 122 held-out dates. It reports
# root-mean-square error (RMSE) for the predictive mean and empirical coverage of the
# pointwise mean $\pm 1.96$ predictive standard deviations. Coverage from one four-month
# interval is descriptive and need not equal the nominal 95% probability.

# %%
holdout_latent = opt_posterior.predict(
    jnp.asarray(holdout_time), train_data=dataset, return_covariance_type="diagonal"
)
holdout_predictive = opt_posterior.likelihood(holdout_latent)

holdout_mean = holdout_predictive.mean.reshape(-1, 1)
holdout_pred_std = jnp.sqrt(holdout_predictive.variance).reshape(-1, 1)

rmse = jnp.sqrt(jnp.mean((holdout_mean - holdout_temp) ** 2))

lower = holdout_mean - 1.96 * holdout_pred_std
upper = holdout_mean + 1.96 * holdout_pred_std
inside = (holdout_temp >= lower) & (holdout_temp <= upper)
coverage = jnp.mean(inside.astype(jnp.float64))

print(f"Held-out RMSE:            {rmse:.2f} °C")
print(f"Empirical 95% coverage:   {100 * coverage:.1f}%")

# %% [markdown]
# ## Compare with an RBF kernel
#
# The next cells fit a zero-mean GP with one RBF kernel to the same thinned training data,
# using the same likelihood, objective, forecast grid, and optimiser. This isolates the
# effect of changing the covariance structure in the plotted comparison.

# %%
plain_kernel = gpx.kernels.RBF(lengthscale=0.1, variance=1.0)
plain_prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(), kernel=plain_kernel
)
plain_posterior = plain_prior * gpx.likelihoods.Gaussian(num_datapoints=dataset.n)

plain_opt, _ = gpx.fit_scipy(
    model=plain_posterior, objective=negative_mll, train_data=dataset
)

plain_latent = plain_opt.predict(
    forecast_grid, train_data=dataset, return_covariance_type="diagonal"
)
plain_predictive = plain_opt.likelihood(plain_latent)
plain_mean = plain_predictive.mean
plain_std = jnp.sqrt(plain_predictive.variance)

# %%
fig, ax = plt.subplots(figsize=(9.0, 3.5))
ax.plot(
    train_time,
    to_celsius(train_temp),
    ".",
    color=cols[0],
    alpha=0.5,
    label="Training data",
)
ax.plot(
    holdout_time,
    to_celsius(holdout_temp),
    ".",
    color="grey",
    alpha=0.7,
    label="Held-out ERA5 values",
)
ax.fill_between(
    forecast_grid.squeeze(),
    to_celsius(plain_mean - 1.96 * plain_std),
    to_celsius(plain_mean + 1.96 * plain_std),
    alpha=0.2,
    color=cols[3],
    label="95% predictive interval",
)
ax.plot(forecast_grid, to_celsius(plain_mean), color=cols[3], label="RBF forecast mean")
ax.axvline(split_time, linestyle="--", color="black", linewidth=1)
ax.set_xlim(split_time - 0.25, forecast_grid.max())
ax.set_xlabel("Time (years since 2020-01-01)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="lower left")
clean_legend(ax)

# %% [markdown]
# With a fitted short lengthscale, the RBF covariance between distant forecast points and
# the training data approaches zero. Its predictive mean then approaches the centred prior
# mean, and its predictive variance approaches the prior predictive variance. The plot
# shows how quickly that occurs for this fitted RBF model. Unlike the composite kernel, the
# RBF kernel contains no covariance term that repeats at one-year lags.
#
# ## Summary
#
# - The fixed-period kernel represents annual covariance.
# - Multiplication by a long-lengthscale Matérn-5/2 kernel allows correlation between
#   corresponding seasonal phases to decrease with separation in years.
# - Addition of a short-lengthscale Matérn-3/2 kernel represents non-periodic temporal
#   correlation around the locally periodic component.
#
# The held-out interval covers 1 September through 31 December 2023. The later extension
# through approximately February 2024 is unobserved extrapolation and is not included in
# the reported RMSE or coverage. For more on combining kernels, see the
# [kernel guide](https://docs.jaxgaussianprocesses.com/_examples/intro_to_kernels/) and the
# [advanced kernel notebook](https://docs.jaxgaussianprocesses.com/_examples/constructing_new_kernels/).
#
# Data: ERA5 reanalysis via [Open-Meteo](https://open-meteo.com/) (CC-BY).

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder & Daniel Dodd'
