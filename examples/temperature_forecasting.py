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
# # Forecasting a Seasonal Time Series with a Bespoke Kernel
#
# The introductory [regression notebook](https://docs.jaxgaussianprocesses.com/_examples/regression/)
# fits a single stationary kernel to simulated data. That is the right starting point,
# but it hides a hard truth about stationary kernels: **they cannot forecast a seasonal
# signal**. An RBF or Matérn kernel only "remembers" the data within roughly one
# lengthscale of a test point. Push a test input further into the future than that and
# the posterior mean decays back to the prior mean, taking the credible interval with it.
# Worse, if you hand a stationary kernel a noisy seasonal series and optimise the marginal
# likelihood, the lengthscale collapses to a tiny value that merely interpolates the
# training points — a good in-sample fit that forecasts nothing.
#
# The fix is not a fancier optimiser; it is a kernel that *encodes what we already know
# about the signal*. In this notebook we forecast daily mean temperature in Reykjavík,
# Iceland, whose dominant feature is a strong annual cycle. We design a composite kernel
# from three interpretable ingredients:
#
# - a **periodic kernel** with its period fixed to one year, to represent the annual cycle;
# - a **Matérn-5/2 kernel of long lengthscale** *multiplied* into the periodic kernel, so
#   the seasonal shape is allowed to drift slowly from year to year (a *locally periodic*
#   kernel rather than an exactly repeating one);
# - a **Matérn-3/2 kernel of short lengthscale** *added* on top, to soak up the
#   autocorrelated weather wobble around the seasonal mean.
#
# We compose these with GPJax's `+` and `*` kernel operators, fit the hyperparameters by
# maximising the conjugate marginal log-likelihood, and forecast a held-out window. The
# payoff is a forecast that *tracks the seasonal cycle* rather than reverting to the mean.
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
# ## The data
#
# The file `data/reykjavik_daily_temperature.csv` holds four full years (2020–2023) of
# daily mean 2-metre air temperature for Reykjavík. We read it with pandas and convert the
# `date` column into a numeric input measured **in years** since the start of the record.
#
# The choice of units matters. If we standardised the input to zero mean and unit variance
# — a common reflex — then the interpretable "period of one year" would become some opaque
# number, and initialising it well would be guesswork. By keeping the input in years, a
# period of exactly `1.0` *is* one calendar year, which we can fix by hand.

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
# A zero-mean GP prior is a natural fit once we **centre** the temperatures. We subtract
# the training-period mean (never the test values — that would leak information from the
# forecast window) and keep everything else in physical units. We hold out the final four
# months as the forecast target and train on everything before it.

# %%
forecast_window_years = 4 / 12  # final four months held out for forecasting
split_time = time_in_years.max() - forecast_window_years

is_train = time_in_years <= split_time
is_holdout = ~is_train

# Centre using the training mean only.
training_mean_celsius = temperature_celsius[is_train].mean()
centred_temperature = temperature_celsius - training_mean_celsius

# %% [markdown]
# Exact GP inference costs a Cholesky factorisation of the training Gram matrix. A little
# over 1300 points is perfectly tractable, but the seasonal structure is already crystal
# clear at coarser resolution, so we **thin the training set to every third day**. This
# keeps the fit brisk without throwing away any of the signal we care about. The held-out
# window is left at full daily resolution so that the RMSE and coverage numbers are honest.

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
# Let's look at what we are asking the model to do. The blue points are the thinned
# training data; the grey points are the four months we will forecast without ever showing
# them to the model.

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
# ## Designing the kernel
#
# We now build the composite kernel piece by piece. Each component carries a clear
# physical meaning, which is the whole point of kernel engineering: the model structure is
# a statement of prior belief about the signal.
#
# **The annual cycle.** GPJax's `Periodic` kernel has the form
#
# $$k(x, x') = \sigma^2 \exp\!\left(-\frac{1}{2}\left(\frac{\sin(\pi (x - x') / p)}{\ell}\right)^2\right),$$
#
# where $p$ is the period. Because our input is in years, we fix $p = 1$. Fixing rather
# than learning it injects the one thing we are certain about — the Earth's orbit — and
# spares the optimiser a hard, multimodal search. We freeze the period by wrapping it in
# `paramax.NonTrainable`, which stops the gradient flowing to it so it stays at exactly one
# year throughout fitting.

# %%
annual_period = paramax.NonTrainable(jnp.array(1.0))
annual_cycle = gpx.kernels.Periodic(lengthscale=1.0, variance=1.0, period=annual_period)

# %% [markdown]
# **Letting the season drift.** A pure periodic kernel insists that every winter is
# identical to every other. Real climate is not so tidy: the amplitude and shape of the
# cycle wander a little between years. We capture that by **multiplying** the periodic
# kernel with a Matérn-5/2 kernel of *long* lengthscale (a few years). The product is a
# *locally periodic* kernel — periodic on the scale of a season, but slowly modulated on
# the scale of years.

# %%
seasonal_envelope = gpx.kernels.Matern52(lengthscale=3.0, variance=1.0)
locally_periodic = annual_cycle * seasonal_envelope

# %% [markdown]
# **Weather around the mean.** Day-to-day departures from the seasonal mean are
# autocorrelated — a cold snap tends to last a week or two — but not periodic. A
# Matérn-3/2 kernel with a *short* lengthscale (a few weeks, i.e. a small fraction of a
# year) models this residual. We **add** it to the seasonal term, so the two effects are
# independent contributions to the covariance.

# %%
weather_residual = gpx.kernels.Matern32(lengthscale=0.05, variance=1.0)

composite_kernel = locally_periodic + weather_residual

# %% [markdown]
# With the kernel assembled, we build a zero-mean prior, attach a Gaussian likelihood
# (temperature is a continuous, roughly Gaussian-noise observation, giving us a
# closed-form conjugate posterior), and form the posterior with the `*` operator.

# %%
mean_function = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=mean_function, kernel=composite_kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
posterior = prior * likelihood

# %% [markdown]
# ## Fitting the hyperparameters
#
# We optimise the remaining hyperparameters — the two variances of the seasonal term, the
# seasonal lengthscale and drift lengthscale, the weather lengthscale and variance, and the
# observation noise — by minimising the negative conjugate marginal log-likelihood. We use
# `fit_scipy`, which wraps SciPy's L-BFGS-B and is well suited to the modest number of
# hyperparameters in an exact GP. The constrained-to-unconstrained bijections (positivity
# of lengthscales and variances) are handled internally.

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
# The fitted kernel is fully interpretable. Let's read off what each component learned.
# The seasonal lengthscale controls how "wiggly" the within-year cycle is, the drift
# lengthscale says how many years it takes for the seasonal shape to change appreciably,
# and the weather lengthscale is the memory of the short-term residual.

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
# The weather lengthscale settles at a few days, matching the persistence of a cold snap,
# and the annual period stays pinned at exactly one year as intended. The drift lengthscale
# comes out far longer than the four-year record — the optimiser found no evidence that the
# seasonal shape changes over this span, so the locally periodic kernel behaves like an
# almost exactly periodic one here. The flexibility was available; the data simply did not
# demand it. With a longer record (decades rather than years) this component is where a
# warming trend or a shifting seasonal amplitude would show up.
#
# ## Forecasting the held-out window
#
# Now for the payoff. We predict over a dense daily grid that spans the tail of the
# training data, the full four-month held-out window, and a short extrapolation two months
# beyond the end of the record — into a future with no data at all. If the kernel is doing
# its job, the posterior mean should keep tracing the seasonal cycle throughout, and the
# credible band should widen gently rather than snapping back to the prior mean.

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
# We plot the forecast in physical units by adding the training mean back on. The vertical
# dashed line marks the boundary between training and forecast; everything to its right the
# model has never seen.


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
    label="Held-out truth",
)
ax.fill_between(
    forecast_grid.squeeze(),
    to_celsius(forecast_mean - 1.96 * forecast_std),
    to_celsius(forecast_mean + 1.96 * forecast_std),
    alpha=0.2,
    color=cols[1],
    label="95% credible interval",
)
ax.plot(forecast_grid, to_celsius(forecast_mean), color=cols[1], label="Forecast mean")
ax.axvline(split_time, linestyle="--", color="black", linewidth=1)
ax.set_xlim(split_time - 0.25, forecast_grid.max())
ax.set_xlabel("Time (years since 2020-01-01)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="lower left")
clean_legend(ax)

# %% [markdown]
# The forecast mean follows the descent into winter and the credible band comfortably
# covers the held-out observations. Crucially, in the two-month extrapolation beyond the
# data the mean *keeps oscillating with the season* — it does not flatten to the prior
# mean. That is the behaviour a plain stationary kernel cannot produce, and it is entirely
# down to the periodic component.
#
# ## Quantifying the forecast
#
# Two numbers summarise how good the forecast is. The **root-mean-square error** measures
# the accuracy of the mean over the held-out window, and the **empirical coverage** of the
# nominal 95% interval checks that the uncertainty is honest — well-calibrated intervals
# should contain close to 95% of the truth.

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
# ## Contrast: why a plain RBF cannot forecast this
#
# To make the argument concrete, we fit a single RBF kernel to the same data and forecast
# the same window. Everything about the pipeline is identical; only the kernel changes.

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
    label="Held-out truth",
)
ax.fill_between(
    forecast_grid.squeeze(),
    to_celsius(plain_mean - 1.96 * plain_std),
    to_celsius(plain_mean + 1.96 * plain_std),
    alpha=0.2,
    color=cols[3],
    label="95% credible interval",
)
ax.plot(forecast_grid, to_celsius(plain_mean), color=cols[3], label="RBF forecast mean")
ax.axvline(split_time, linestyle="--", color="black", linewidth=1)
ax.set_xlim(split_time - 0.25, forecast_grid.max())
ax.set_xlabel("Time (years since 2020-01-01)")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="lower left")
clean_legend(ax)

# %% [markdown]
# Just past the last training point the RBF forecast collapses to the (centred) prior mean
# and the band inflates to the marginal prior variance. The GP has learned a short
# lengthscale that interpolates the training noise beautifully and forecasts nothing —
# exactly the failure mode described in the introduction. The bespoke composite kernel
# succeeds not because it is more flexible, but because it is more *opinionated*: it builds
# in the annual cycle that the plain kernel has no way of expressing.
#
# ## Recap
#
# - The **periodic kernel** with a fixed one-year period supplies the annual cycle and lets
#   the forecast extrapolate the season indefinitely.
# - **Multiplying** it by a long-lengthscale Matérn-5/2 makes the cycle *locally* periodic,
#   so the seasonal shape can drift slowly across years instead of repeating exactly.
# - **Adding** a short-lengthscale Matérn-3/2 captures autocorrelated weather noise around
#   the seasonal mean without polluting the long-range forecast.
#
# Composing kernels with `+` and `*` turns qualitative domain knowledge into a quantitative
# model whose parameters — period, lengthscales, variances — remain individually
# interpretable after fitting. For more on combining kernels, see the
# [kernel guide](https://docs.jaxgaussianprocesses.com/_examples/intro_to_kernels/) and the
# [advanced kernel notebook](https://docs.jaxgaussianprocesses.com/_examples/constructing_new_kernels/).
#
# Data: ERA5 reanalysis via [Open-Meteo](https://open-meteo.com/) (CC-BY).

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder & Daniel Dodd'
