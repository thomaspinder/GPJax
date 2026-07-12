# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
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
# # Heteroscedastic Inference
#
# This notebook shows how to fit a heteroscedastic Gaussian processes (GPs) that
# allows one to perform regression where there exists non-constant, or
# input-dependent, noise.
#
#
# ## Background
# A heteroscedastic GP couples two latent functions:
# - A **signal GP** $f(\cdot)$ for the mean response.
# - A **noise GP** $g(\cdot)$ that maps to a positive variance
#   $\sigma^2(x) = \phi(g(x))$ via a positivity transform $\phi$ (typically
#   ${\rm exp}$ or ${\rm softplus}$). Intuitively, we are introducing a pair of GPs;
# one to model the latent mean, and a second that models the log-noise variance. This
# is in direct contrast a
# [homoscedastic GP](https://docs.jaxgaussianprocesses.com/_examples/regression/)
# where we learn a constant value for the noise.
#
# In the Gaussian case, the observed response follows
# $$y \mid f, g \sim \mathcal{N}(f, \sigma^2(x)).$$
# Variational inference works with independent posteriors $q(f)q(g)$, combining the
# moments of each into an ELBO. For non-Gaussian likelihoods the same structure
# remains; only the expected log-likelihood changes.
#
# ## A real heteroscedastic signal: solar irradiance
# Rather than a synthetic construction, we use a genuinely heteroscedastic physical
# process: the surface solar irradiance recorded at Barcelona as a function of the
# **hour of day**. Two effects combine to make the noise strongly input-dependent:
# - **Clear-sky ceiling.** On a cloudless day the irradiance follows an almost
#   deterministic diurnal arc set by the sun's elevation. At night it is exactly
#   zero. These regimes carry essentially *no* variance.
# - **Broken cloud.** Around midday the sun is high and passing cloud can knock
#   hundreds of watts per square metre off the clear-sky value from one day to the
#   next. This is where the variance is *largest*.
# Pooling ~183 days of hourly data, each hour of day therefore has a spread of
# irradiance values that collapses towards zero at night and fans out at midday.
# That spread is precisely the input-dependent variance $\sigma^2(x)$ the model must
# learn.
#
# Data: ERA5 reanalysis via [Open-Meteo](https://open-meteo.com/) (CC-BY).

# %%
from examples.utils import use_mpl_style
import gpjax as gpx
from gpjax.likelihoods import (
    HeteroscedasticGaussian,
    LogNormalTransform,
    SoftplusTransform,
)
from gpjax.variational_families import (
    HeteroscedasticVariationalFamily,
    VariationalGaussianInit,
)
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
import pandas as pd

# Enable Float64 for stable linear algebra.
config.update("jax_enable_x64", True)


use_mpl_style()
key = jr.key(123)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


# %% [markdown]
# ## Loading the data
# The file `data/barcelona_solar_radiation.csv` holds hourly ERA5 shortwave
# (global horizontal) irradiance for Barcelona over 2023-04-01 to 2023-09-30, along
# with the hour of day. We take the input $x$ to be the hour of day and the response
# $y$ to be the irradiance in W/m². Because inducing-point variational inference
# scales with the number of observations, we draw a random subsample that still
# spans every hour of day, so the night-time collapse of the variance remains
# visible.

# %%
# Load the hourly solar-irradiance record (fall back to the docs data path).
try:
    solar_frame = pd.read_csv("data/barcelona_solar_radiation.csv")
except FileNotFoundError:
    solar_frame = pd.read_csv("examples/data/barcelona_solar_radiation.csv")

# Randomly subsample to keep variational inference fast while covering all hours.
key, subsample_key = jr.split(key)
num_observations = 700
subsample_indices = jr.choice(
    subsample_key,
    solar_frame.shape[0],
    shape=(num_observations,),
    replace=False,
)
solar_sample = solar_frame.iloc[np.asarray(subsample_indices)]
hour_of_day = solar_sample["hour"].to_numpy(dtype=float)
irradiance = solar_sample["shortwave_radiation"].to_numpy(dtype=float)

# %% [markdown]
# ### Standardisation
# We standardise both the hour-of-day input and the irradiance response to zero mean
# and unit variance for stable optimisation, retaining the moments so that
# predictions can be mapped back to physical units (hour of day and W/m²) for
# plotting.

# %%
hour_mean, hour_std = hour_of_day.mean(), hour_of_day.std()
irradiance_mean, irradiance_std = irradiance.mean(), irradiance.std()

x = ((hour_of_day - hour_mean) / hour_std)[:, None]
y = ((irradiance - irradiance_mean) / irradiance_std)[:, None]
train = gpx.Dataset(X=jnp.asarray(x), y=jnp.asarray(y))


def hours_to_standardised(
    hours: np.ndarray, mean: float = hour_mean, std: float = hour_std
) -> jnp.ndarray:
    """Map hours of day onto the standardised input scale used for training."""
    return jnp.asarray(((hours - mean) / std)[:, None])


# Dense grid of hours for visualising posterior fits and predictive uncertainty.
hour_grid = np.linspace(0.0, 23.0, 200)
xtest = hours_to_standardised(hour_grid)

# Empirical mean and spread of irradiance within each hour of day (the target the
# heteroscedastic model must recover). We build these with an explicit loop rather
# than a comprehension so the notebook also runs under the integration-test harness,
# which ``exec``s it with separate globals/locals (comprehensions have their own
# scope and would not see the module-level ``irradiance`` array).
empirical_hours = np.arange(24)
empirical_mean_values = []
empirical_std_values = []
for hour in empirical_hours:
    irradiance_at_hour = irradiance[hour_of_day == hour]
    empirical_mean_values.append(irradiance_at_hour.mean())
    empirical_std_values.append(irradiance_at_hour.std())
empirical_mean = np.array(empirical_mean_values)
empirical_std = np.array(empirical_std_values)

fig, ax = plt.subplots()
ax.plot(hour_of_day, irradiance, "o", label="Observations", alpha=0.3, color=cols[0])
ax.plot(empirical_hours, empirical_mean, label="Hourly mean", color=cols[1])
ax.plot(empirical_hours, empirical_std, label="Hourly std. dev.", color=cols[2])
ax.set_xlabel("hour of day")
ax.set_ylabel("solar irradiance (W/m²)")
ax.legend(loc="upper left")

# %% [markdown]
# The hourly mean traces the diurnal arc, peaking in the early afternoon, while the
# hourly standard deviation is near zero overnight and swells to well over 100 W/m²
# around midday — the cloud-driven, input-dependent noise we set out to model. For a
# homoscedastic baseline, compare this with the
# [Gaussian process regression notebook](https://docs.jaxgaussianprocesses.com/_examples/regression/)
# (`examples/regression.py`), where a single latent GP is paired with constant
# observation noise.

# %% [markdown]
# ## Prior specification
# We place independent Gaussian process priors on the signal and noise processes:
# $$f \sim \mathcal{GP}\big(0, k_f\big), \qquad g \sim \mathcal{GP}\big(0, k_g\big),$$
# where $k_f$ and $k_g$ are stationary squared-exponential kernels with unit
# variance and a lengthscale of one half (on the standardised input scale), giving
# the noise process enough flexibility to collapse the variance overnight and swell
# it at midday. The noise process $g$ is mapped to the variance
# via the logarithmic transform in `LogNormalTransform`, giving
# $\sigma^2(x) = \exp\big(g(x)\big)$. The joint prior over $(f, g)$ combines with
# the heteroscedastic Gaussian likelihood,
# $$p(\mathbf{y} \mid f, g) = \prod_{i=1}^n
# \mathcal{N}\!\big(y_i \mid f(x_i), \exp(g(x_i))\big),$$
# to form the posterior target that we shall approximate variationally. The product
# syntax `signal_prior * likelihood` used below constructs this augmented GP model.

# %%
# Signal and noise priors.
signal_prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(),
    kernel=gpx.kernels.RBF(lengthscale=0.5),
)
noise_prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(),
    kernel=gpx.kernels.RBF(lengthscale=0.5),
)
likelihood = HeteroscedasticGaussian(
    num_datapoints=train.n,
    noise_prior=noise_prior,
    noise_transform=LogNormalTransform(),
)
posterior = signal_prior * likelihood

# Variational family over both processes, with inducing inputs spanning the observed
# range of (standardised) hours of day.
z = jnp.linspace(float(x.min()), float(x.max()), 18)[:, None]
q = HeteroscedasticVariationalFamily(
    posterior=posterior,
    inducing_inputs=z,
    inducing_inputs_g=z,
)

# %% [markdown]
# The variational family introduces inducing variables for both latent functions,
# located at the set $Z = \{z_m\}_{m=1}^M$. These inducing variables summarise the
# infinite-dimensional GP priors in terms of multivariate Gaussian parameters.
# Optimising the evidence lower bound (ELBO) corresponds to adjusting the means and
# covariances of the variational posteriors $q(f)$ and $q(g)$ so that they best
# explain the observed data whilst remaining close to the prior. For a deeper look at
# these constructions in the homoscedastic setting, refer to the
# [Sparse Gaussian Process Regression](https://docs.jaxgaussianprocesses.com/_examples/collapsed_vi/)
# (`examples/collapsed_vi.py`) and
# [Sparse Stochastic Variational Inference](https://docs.jaxgaussianprocesses.com/_examples/uncollapsed_vi/)
# (`examples/uncollapsed_vi.py`) notebooks.

# %% [markdown]
# ### Optimisation
# With the model specified, we minimise the negative ELBO,
# $$\mathcal{L} = \mathbb{E}_{q(f)q(g)}\!\big[\log p(\mathbf{y}\mid f, g)\big]
# - \mathrm{KL}\!\left[q(f) \,\|\, p(f)\right]
# - \mathrm{KL}\!\left[q(g) \,\|\, p(g)\right],$$
# using the Adam optimiser. GPJax automatically selects the tight bound of
# Lázaro-Gredilla & Titsias (2011) when the likelihood is Gaussian, yielding an
# analytically tractable expectation over the latent noise process. The resulting
# optimisation iteratively updates the inducing posteriors for both latent GPs.

# %%
# Optimise the heteroscedastic ELBO (selects tighter bound).
objective = lambda model, data: -gpx.objectives.heteroscedastic_elbo(model, data)
optimiser = ox.adam(1e-2)
q_trained, history = gpx.fit(
    model=q,
    objective=objective,
    train_data=train,
    optim=optimiser,
    num_iters=10000,
    verbose=False,
)

loss_trace = jnp.asarray(history)
print(f"Final regression ELBO: {-loss_trace[-1]:.3f}")

# %% [markdown]
# ## Prediction
# After training we obtain posterior marginals for both latent functions. To make a
# prediction we evaluate two quantities:
# 1. The latent posterior over $f$ (mean and variance), which reflects uncertainty
#    in the latent function **prior** to observing noise.
# 2. The marginal predictive over observations, which integrates out both $f$ and
#    $g$ to provide predictive intervals for future noisy measurements.
# The helper method `likelihood.predict` performs the second integration for us.

# %%
# Predict on the dense grid of hours of day (defined above as ``xtest``).
mf, vf, mg, vg = q_trained.predict(xtest)

signal_pred, noise_pred = q_trained.predict_latents(xtest)
predictive = likelihood.predict(signal_pred, noise_pred)


def unstandardise_mean(
    values: jnp.ndarray,
    mean: float = irradiance_mean,
    std: float = irradiance_std,
) -> jnp.ndarray:
    """Map a standardised irradiance mean back to W/m²."""
    return jnp.asarray(values).squeeze() * std + mean


def unstandardise_std(
    values: jnp.ndarray, std: float = irradiance_std
) -> jnp.ndarray:
    """Map a standardised irradiance standard deviation back to W/m²."""
    return jnp.asarray(values).squeeze() * std


latent_mean = unstandardise_mean(mf)
latent_std = unstandardise_std(jnp.sqrt(vf.squeeze()))
predictive_mean = unstandardise_mean(predictive.mean)
predictive_std = unstandardise_std(jnp.sqrt(jnp.diag(predictive.covariance_matrix)))

fig, ax = plt.subplots()
ax.plot(hour_of_day, irradiance, "o", label="Observations", alpha=0.3)
ax.plot(hour_grid, predictive_mean, color="C0", label="Posterior mean")
ax.fill_between(
    hour_grid,
    latent_mean - 2 * latent_std,
    latent_mean + 2 * latent_std,
    color="C0",
    alpha=0.15,
    label="±2 std (latent)",
)
ax.fill_between(
    hour_grid,
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    color="C1",
    alpha=0.15,
    label="±2 std (observed)",
)
ax.set_xlabel("hour of day")
ax.set_ylabel("solar irradiance (W/m²)")
ax.legend(loc="upper left")
ax.set_title("Heteroscedastic regression")

# Report the recovered predictive spread against the empirical hourly spread.
midday_index = int(np.argmin(np.abs(hour_grid - 13.0)))
night_index = int(np.argmin(np.abs(hour_grid - 3.0)))
print(
    "Predicted observation std. dev. — "
    f"night (03:00): {float(predictive_std[night_index]):.1f} W/m², "
    f"midday (13:00): {float(predictive_std[midday_index]):.1f} W/m²"
)

# %% [markdown]
# The latent intervals quantify epistemic uncertainty about $f$, whereas the broader
# observed band adds the aleatoric noise predicted by $g$. The observed band is
# pinched almost shut overnight, when the irradiance is a deterministic zero, and
# fans out around midday, where passing cloud makes the irradiance far less
# predictable — exactly the input-dependent noise structure seen in the raw data.

# %% [markdown]
# ## Sparse Heteroscedastic Regression
#
# We now demonstrate how the aforementioned heteroscedastic approach can be extended
# into sparse scenarios, thus offering more favourable scalability as the size of our
# dataset grows. To achieve this we define separate inducing points for the signal
# and noise processes. Decoupling these grids allows us to focus modelling capacity
# where each latent function varies the most. The solar record is a natural fit: the
# signal (the diurnal irradiance arc) is smooth but structured across every hour,
# whereas the noise process is broad and slowly varying, concentrated around midday.
# We therefore hand the signal a richer inducing set than the noise process, and draw
# a slightly larger subsample to illustrate the sparse scaling.

# %%
# Draw a fresh, larger subsample of the solar record for the sparse model.
key, sparse_key = jr.split(key)
num_sparse_observations = 800
sparse_indices = jr.choice(
    sparse_key,
    solar_frame.shape[0],
    shape=(num_sparse_observations,),
    replace=False,
)
sparse_sample = solar_frame.iloc[np.asarray(sparse_indices)]
sparse_hour = sparse_sample["hour"].to_numpy(dtype=float)
sparse_irradiance = sparse_sample["shortwave_radiation"].to_numpy(dtype=float)

x = ((sparse_hour - hour_mean) / hour_std)[:, None]
y = ((sparse_irradiance - irradiance_mean) / irradiance_std)[:, None]
data_adv = gpx.Dataset(X=jnp.asarray(x), y=jnp.asarray(y))

# %% [markdown]
# ### Model components
# We again adopt RBF priors for both processes but now apply a `SoftplusTransform`
# to the noise GP. This alternative map enforces positivity whilst avoiding the
# heavier tails induced by the log-normal transform. The `HeteroscedasticGaussian`
# likelihood seamlessly accepts the new transform.

# %%
# Define model components
mean_prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(),
    kernel=gpx.kernels.RBF(lengthscale=0.5),
)
noise_prior_adv = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(),
    kernel=gpx.kernels.RBF(lengthscale=0.5),
)
likelihood_adv = HeteroscedasticGaussian(
    num_datapoints=data_adv.n,
    noise_prior=noise_prior_adv,
    noise_transform=SoftplusTransform(),
)
posterior_adv = mean_prior * likelihood_adv

# %%
# Configure variational family
# The signal requires a richer inducing set to trace the diurnal arc, whereas the
# broad, slowly varying noise process can be summarised with fewer points.
z_signal = jnp.linspace(float(x.min()), float(x.max()), 24)[:, None]
z_noise = jnp.linspace(float(x.min()), float(x.max()), 12)[:, None]

# Use VariationalGaussianInit to pass specific configurations
q_init_f = VariationalGaussianInit(inducing_inputs=z_signal)
q_init_g = VariationalGaussianInit(inducing_inputs=z_noise)

q_sparse = HeteroscedasticVariationalFamily(
    posterior=posterior_adv,
    signal_init=q_init_f,
    noise_init=q_init_g,
)

# %% [markdown]
# The initialisation objects `VariationalGaussianInit` allow us to prescribe
# different inducing grids and initial covariance structures for $f$ and $g$. This
# flexibility is invaluable when working with large datasets where the latent
# functions have markedly different smoothness properties.

# %%
# Optimize
objective_adv = lambda model, data: -gpx.objectives.heteroscedastic_elbo(model, data)
optimiser_adv = ox.adam(1e-2)
q_sparse_trained, _ = gpx.fit(
    model=q_sparse,
    objective=objective_adv,
    train_data=data_adv,
    optim=optimiser_adv,
    num_iters=10000,
    verbose=False,
)

# %%
# Plotting
xtest = hours_to_standardised(hour_grid)

# The likelihood expects the *latent* signal and noise distributions to compute the
# predictive, so we obtain them and integrate to get calibrated observation
# intervals, then map back to physical units (W/m²).
signal_dist, noise_dist = q_sparse_trained.predict_latents(xtest)
predictive_dist = likelihood_adv.predict(signal_dist, noise_dist)
predictive_mean = unstandardise_mean(predictive_dist.mean)
predictive_std = unstandardise_std(
    jnp.sqrt(jnp.diag(predictive_dist.covariance_matrix))
)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(sparse_hour, sparse_irradiance, "x", color="black", alpha=0.3, label="Data")

# Plot total uncertainty (signal + noise)
ax.plot(hour_grid, predictive_mean, "--", color=cols[1], linewidth=2)
ax.fill_between(
    hour_grid,
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    color=cols[1],
    alpha=0.3,
    label="One std. dev.",
)
ax.plot(
    hour_grid,
    predictive_mean - predictive_std,
    "--",
    color=cols[1],
    alpha=0.5,
    linewidth=0.75,
)
ax.plot(
    hour_grid,
    predictive_mean + predictive_std,
    "--",
    color=cols[1],
    alpha=0.5,
    linewidth=0.75,
)
ax.fill_between(
    hour_grid,
    predictive_mean - 2 * predictive_std,
    predictive_mean + 2 * predictive_std,
    color=cols[1],
    alpha=0.1,
    label="Two std. dev.",
)
ax.plot(
    hour_grid,
    predictive_mean - 2 * predictive_std,
    "--",
    color=cols[1],
    alpha=0.5,
    linewidth=0.75,
)
ax.plot(
    hour_grid,
    predictive_mean + 2 * predictive_std,
    "--",
    color=cols[1],
    alpha=0.5,
    linewidth=0.75,
)

ax.set_title("Sparse Heteroscedastic Regression")
ax.legend(loc="best", fontsize="small")
ax.set_xlabel("hour of day")
ax.set_ylabel("solar irradiance (W/m²)")

# %% [markdown]
# ## Takeaways
# - Solar irradiance versus hour of day is a textbook heteroscedastic process: the
#   clear-sky arc sets an almost deterministic ceiling (near-zero variance overnight),
#   while broken cloud injects large, cloud-driven variance around midday.
# - The heteroscedastic GP model couples two latent GPs, enabling separate control of
#   epistemic and aleatoric uncertainties, and recovers the collapsing-then-widening
#   noise band directly from the data.
# - We support multiple positivity transforms for the noise process; the choice
#   affects the implied variance tails and should reflect prior beliefs.
# - Inducing points for the signal and noise processes can be tuned independently to
#   balance computational budget against the local complexity of each function.
# - The ELBO implementation automatically selects the tightest analytical bound
#   available, streamlining heteroscedastic inference workflows.
#
# Data: ERA5 reanalysis via [Open-Meteo](https://open-meteo.com/) (CC-BY).

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
