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
# # Climate Extrapolation: Trend Forecasting with Quantified Uncertainty
#
# This notebook is a flagship "why Gaussian processes" example. We take real climate
# projection data for Paris, fit a Gaussian process (GP) to the *historical* period, and
# extrapolate the warming trend forward to 2050 — crucially, with a calibrated credible
# interval attached to every forecast. We then hold that GP forecast up against the spread
# of an ensemble of climate models, and use the comparison to draw a distinction that is
# easy to blur in practice: the uncertainty a GP reports when it extrapolates a trend is
# *not* the same object as the structural uncertainty between competing physical models.
#
# The data are annual mean 2 m air temperatures from seven CMIP6 HighResMIP models,
# downscaled to Paris, France, spanning 1950–2050. A clear warming signal runs through the
# record: the ensemble mean rises from roughly 11.1 °C in the 1950s to about 12.9 °C in
# the 2040s, with the individual models disagreeing by around a degree at the 2040s
# horizon. That combination — a strong trend plus a genuine spread of plausible futures —
# is exactly the setting where a GP earns its keep.
#
# > Data: CMIP6 HighResMIP projections via Open-Meteo (CC-BY). Note that the CMIP6
# > "historical" values up to 2014 are themselves *model simulations*, not station
# > observations; we use them here as a self-consistent trend to extrapolate, not as
# > ground truth.

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
# ## The data: seven climate models, one city
#
# We load the projections and inspect the raw trajectories. Each of the seven models
# supplies one annual mean temperature per year, so the frame is a long table of
# `(year, annual_mean_temp, model)` rows. Plotting each model as its own line makes two
# features immediately visible: a shared upward drift (the forced warming response), and a
# persistent vertical spread between models (their differing physics and resolution).

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
ax.set_title("CMIP6 HighResMIP projections for Paris (1950-2050)")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")

# %% [markdown]
# ## Building the target: the ensemble mean
#
# We reduce the seven trajectories to a single series by averaging across models for each
# year — the *ensemble mean*. This is the signal we will model. For each year we also
# record the between-model spread (standard deviation across models, and the min/max
# envelope); we set those aside for now and bring them back for the payoff comparison at
# the end.

# %%
by_year = climate.groupby("year")["annual_mean_temp"]
ensemble_mean = by_year.mean()
ensemble_std = by_year.std()
ensemble_min = by_year.min()
ensemble_max = by_year.max()

years = ensemble_mean.index.to_numpy()
ensemble_mean_temp = ensemble_mean.to_numpy()

# %% [markdown]
# ## Train / forecast split at the historical boundary
#
# CMIP6 splits its runs at 2014: everything up to and including 2014 is the "historical"
# experiment, and 2015 onwards is the future scenario. We honour that boundary. The GP is
# fitted using *only* the ensemble mean up to 2014, and asked to extrapolate across
# 2015–2050. The scenario-period values are held back entirely — they are the yardstick
# the GP forecast will be measured against, never an input to it.

# %%
historical_boundary = 2014
is_historical = years <= historical_boundary

train_years = years[is_historical]
train_temp = ensemble_mean_temp[is_historical]

# %% [markdown]
# ### Standardise the inputs, centre the outputs
#
# Two preprocessing steps make the GP well behaved. First we standardise the year to zero
# mean and unit variance; kernel lengthscales and the linear-kernel variance are then
# expressed on a natural, order-one scale rather than in raw calendar years. Second we
# centre the temperature by subtracting its training mean, so the GP's constant mean starts
# from a sensible default and the linear kernel — whose functions pass through the origin —
# is anchored at the middle of the training window. We keep the shift/scale constants so we
# can map predictions back to °C.

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
# ## A structured kernel: trend plus variability
#
# The Mauna Loa CO₂ recipe teaches a useful decomposition: build a kernel by *summing*
# components, each responsible for one feature of the data. We adapt it to annual means.
# There is no within-year seasonality to capture at this resolution, so we drop the
# periodic term and keep two pieces:
#
# $$k(x, x') = \underbrace{\sigma_\text{lin}^2 \, x\,x'}_{\text{Linear: warming trend}}
#   \; + \; \underbrace{k_\text{Matérn}(x, x')}_{\text{Matérn: residual variability}}.$$
#
# The **linear** kernel is non-stationary and encodes the long-run warming trend; on its
# own it would produce straight-line forecasts. The **Matérn-5/2** kernel is stationary and
# soaks up the smooth wiggles around that trend.
#
# There is a subtlety worth being explicit about. If we let *every* hyperparameter float,
# the flexible stationary kernel is happy to explain the gentle historical rise on its own
# with a multi-decade lengthscale — and a stationary kernel, extrapolated, simply reverts
# to the prior mean, so the "trend" evaporates beyond the data. To keep a clean
# division of labour we fix the Matérn lengthscale to a short, sub-decadal scale, so it can
# only mop up short-range residual variability and the *linear* kernel is forced to own the
# long-run trend. Everything else — the linear variance, the Matérn variance, the constant
# mean, and the observation noise — is still learned from the data.
#
# This decomposition shapes the extrapolation in exactly the way we want. Away from the
# training data the short Matérn term reverts to the prior, while the linear term keeps
# carrying the trend forward and its variance *grows* with distance from the origin: the GP
# commits to continued warming but widens its credible interval the further it reaches.

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
# ## Fit by maximising the marginal likelihood
#
# Because the likelihood is Gaussian, inference over the latent function is closed form and
# we learn the free hyperparameters — the linear variance, the Matérn variance, the
# constant mean, and the observation noise — by maximising the conjugate marginal log
# likelihood (equivalently, minimising its negative). We optimise with Adam; the frozen
# lengthscale is simply held constant throughout.

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
# A quick summary of the fitted model confirms the learned hyperparameters, with the Matérn
# lengthscale marked non-trainable.

# %%
gpx.summarise(opt_posterior)

# %% [markdown]
# ## Predict and map back to degrees Celsius
#
# We query the posterior at every test year, take the predictive (noise-inflated)
# distribution, and undo the centring so the mean and credible interval are in °C. A ±2σ
# band is our 95% credible interval.

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
# Plotting the fit across the whole record shows the GP tracking the historical ensemble
# mean tightly, then extrapolating past the 2014 boundary (the dashed vertical line) along
# the learned linear trend. The credible interval is narrow where data pin it down and
# widens into the forecast — the hallmark of an honest extrapolation.

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
    label="GP 95% credible interval",
)
ax.plot(forecast_years, predictive_mean_temp, color=cols[1], label="GP predictive mean")
ax.set_xlabel("Year")
ax.set_ylabel("Annual mean temperature (°C)")
ax.set_title("GP fit to historical ensemble mean, extrapolated to 2050")
ax.legend(loc="upper left", fontsize="small")
clean_legend(ax)

# %% [markdown]
# ## The payoff: two very different uncertainties
#
# Here is the point of the notebook. On the 2015–2050 forecast window we overlay two bands
# that both *look* like "uncertainty about future warming" but mean different things:
#
# 1. **The GP credible interval** (from extrapolating the historical trend). This answers:
#    *given the single historical trend and how noisily it was realised, how confident is
#    the model in continuing that trend?* It is an epistemic statement about one model's
#    extrapolation, and it grows steadily with the forecast horizon.
# 2. **The between-model spread** of the seven CMIP6 projections (their min–max envelope,
#    plus a ±1σ across-model band). This answers a different question: *how much do
#    independent physical models, run under the scenario, disagree about the future?* It is
#    structural — it reflects genuine scientific uncertainty about climate physics and
#    forcing that no amount of curve-fitting to one historical trend can see.

# %%
is_forecast = years >= 2015
forecast_period_years = years[is_forecast]

in_forecast_window = forecast_years >= 2015
plot_years = forecast_years[in_forecast_window]

fig, ax = plt.subplots(figsize=(10, 5))

# GP extrapolation credible interval.
ax.fill_between(
    plot_years,
    lower_credible[in_forecast_window],
    upper_credible[in_forecast_window],
    alpha=0.25,
    color=cols[1],
    label="GP 95% credible interval",
)
ax.plot(
    plot_years,
    predictive_mean_temp[in_forecast_window],
    color=cols[1],
    label="GP predictive mean",
)

# Between-model structural spread.
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
# The two bands tell genuinely different stories, and the gap between them is the lesson.
# The GP continues the historical trend and forecasts modest further warming — but it
# *undershoots* the scenario ensemble, whose central path climbs faster than anything in
# the pre-2015 record. Even the top of the GP's 95% credible interval only just reaches the
# bottom of the between-model envelope by 2050.
#
# This is not a failure of the GP; it is the GP being honest about what it was shown. The
# post-2014 projections are driven by a future emissions scenario whose forcing accelerates
# beyond the historical rate, and that acceleration is simply not present in the data the
# GP trained on. The GP's credible interval faithfully quantifies uncertainty *about the
# trend it saw*; it cannot manufacture uncertainty about a change of regime it never
# witnessed. The between-model spread, by contrast, is precisely a measure of that
# structural, physics-and-scenario uncertainty.
#
# The practical takeaway: a GP gives beautifully calibrated uncertainty for interpolation
# and short-range extrapolation of an observed process, but structural uncertainty about
# the *future* forcing of a system lives outside the data and must come from elsewhere —
# here, from the ensemble of physical models. A serious climate risk assessment needs both:
# the GP's within-trend credible interval **and** the ensemble's between-model spread,
# because they quantify complementary sources of ignorance.

# %%
forecast_2050_index = int(jnp.argmin(jnp.abs(forecast_years - 2050.0)))
gp_2050_mean = float(predictive_mean_temp[forecast_2050_index])
gp_2050_lower = float(lower_credible[forecast_2050_index])
gp_2050_upper = float(upper_credible[forecast_2050_index])

ensemble_2050 = climate[climate["year"] == 2050]["annual_mean_temp"]
print(
    f"GP 2050 forecast:         {gp_2050_mean:.2f} °C "
    f"[{gp_2050_lower:.2f}, {gp_2050_upper:.2f}] (95% CI)"
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
