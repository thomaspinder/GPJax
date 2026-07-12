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
# # Count data regression
#
# In this notebook we demonstrate how to perform inference for Gaussian process models
# with non-Gaussian likelihoods via Markov chain Monte
# Carlo (MCMC). We focus on a count data regression task here and use
# [BlackJax](https://github.com/blackjax-devs/blackjax/) for sampling.

# %%
from pathlib import Path

import blackjax
import equinox as eqx
from examples.utils import use_mpl_style
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import paramax

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.key(42)

# %% [markdown]
# ## Dataset
#
# For count data regression, the Poisson distribution is a natural choice for the likelihood
# function. The probability mass function of the Poisson distribution is given by
#
# $$ p(y \,|\, \lambda) = \frac{\lambda^{y} e^{-\lambda}}{y!},$$
#
# where $y$ is the count and the parameter $\lambda \in \mathbb{R}_{>0}$ is the rate of the Poisson
# distribution.
#
# We then set $\lambda = \exp(f)$ where $f$ is the latent Gaussian process. The exponential function
# is the _link function_ for the Poisson distribution: it maps the output of a GP to the positive
# real line, which is suitable for modeling count data.
#
# Rather than a simulated series, we use a genuine count dataset: the number of **hot days**
# recorded each year in Madrid, Spain, defined as days on which the maximum temperature reached at
# least 30 °C. The record spans 1960–2023 (64 annual counts) and is derived from the ERA5
# reanalysis. Over this period the annual hot-day count rises noticeably — roughly 49 days in 1960
# to over 80 days by 2020 — a clear climate-warming signal that is a natural fit for a
# Poisson-likelihood GP. The accompanying `frost_days` column (days with $T_\min \le 0$ °C) tells
# the mirror-image story of a *declining* count over the same period; here we model the hot-day
# series, but the identical workflow applies to that alternative target.
#
# We use the calendar `year` as our input $\mathbf{X}$ and the hot-day count as the output
# $\mathbf{y}$. The input is **standardised** (centred and scaled to unit variance), which improves
# the conditioning of the GP inference. We store the data $\mathcal{D}$ as a GPJax `Dataset` and
# retain the standardisation constants so predictions can be mapped back to calendar years.

# %%
csv_candidates = [
    Path("examples/data/madrid_annual_extreme_days.csv"),
    Path("data/madrid_annual_extreme_days.csv"),
]
csv_path = next(path for path in csv_candidates if path.exists())
madrid_data = pd.read_csv(csv_path)

year = madrid_data["year"].to_numpy()
hot_days = madrid_data["hot_days_30"].to_numpy()

year_mean = year.mean()
year_std = year.std()
year_standardised = ((year - year_mean) / year_std).reshape(-1, 1)
count = hot_days.reshape(-1, 1).astype(float)

D = gpx.Dataset(X=jnp.asarray(year_standardised), y=jnp.asarray(count))

xtest = jnp.linspace(year_standardised.min(), year_standardised.max(), 500).reshape(
    -1, 1
)
year_test = xtest.flatten() * year_std + year_mean

fig, ax = plt.subplots()
ax.plot(year, hot_days, "o", label="Observed counts", color=cols[1])
ax.set_xlabel("year")
ax.set_ylabel("hot days (Tmax ≥ 30 °C) per year")
ax.legend()

# %% [markdown]
# ## Gaussian Process definition
#
# We begin by defining a Gaussian process prior with a radial basis function (RBF)
# kernel, chosen for the purpose of exposition. We adopt the Poisson likelihood available in GPJax.

# %%
kernel = gpx.kernels.RBF()
meanf = gpx.mean_functions.Constant()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.Poisson(num_datapoints=D.n)

# %% [markdown]
# We construct the posterior through the product of our prior and likelihood.

# %%
posterior = prior * likelihood
print(type(posterior))

# %% [markdown]
# Whilst the latent function is Gaussian, the posterior distribution is non-Gaussian
# since our generative model first samples the latent GP and propagates these samples
# through the likelihood function's inverse link function. This step prevents us from
# being able to analytically integrate the latent function's values out of our
# posterior, and we must instead adopt alternative inference techniques. Here, we show
# how to use MCMC methods.


# %% [markdown]
# ## MCMC inference
#
# An MCMC sampler works by starting at an initial position and
# drawing a sample from a cheap-to-simulate distribution known as the _proposal_. The
# next step is to determine whether this sample could be considered a draw from the
# posterior. We accomplish this using an _acceptance probability_ determined via the
# sampler's _transition kernel_ which depends on the current position and the
# unnormalised target posterior distribution. If the new sample is more _likely_, we
# accept it; otherwise, we reject it and stay in our current position. Repeating these
# steps results in a Markov chain (a random sequence that depends only on the last
# state) whose stationary distribution (the long-run empirical distribution of the
# states visited) is the posterior. For a gentle introduction, see the first chapter
# of [A Handbook of Markov Chain Monte Carlo](https://www.mcmchandbook.net/HandbookChapter1.pdf).
#
# ### MCMC through BlackJax
#
# Rather than implementing a suite of MCMC samplers, GPJax relies on MCMC-specific
# libraries for sampling functionality. We focus on
# [BlackJax](https://github.com/blackjax-devs/blackjax/) in this notebook, which we
# recommend adopting for general applications.
#
# We begin by generating _sensible_ initial positions for our sampler before defining
# an inference loop and sampling 200 values from our Markov chain. In practice,
# drawing more samples will be necessary.

# %%
# Adapted from BlackJax's introduction notebook.
num_adapt = 1000
num_samples = 500


params, static = eqx.partition(posterior, eqx.is_array)


def logprob_fn(params):
    model = eqx.combine(params, static)
    model = paramax.unwrap(model)
    return gpx.objectives.log_posterior_density(model, D)


step_size = 1e-3
n_params = sum(jnp.size(leaf) for leaf in jtu.tree_leaves(params))
inverse_mass_matrix = jnp.ones(n_params)
nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)

state = nuts.init(params)

step = jax.jit(nuts.step)


def one_step(state, rng_key):
    state, info = step(rng_key, state)
    return state, (state, info)


keys = jax.random.split(key, num_samples)
_, (states, infos) = jax.lax.scan(one_step, state, keys, unroll=10)

# %% [markdown]
# ### Sampler efficiency
#
# BlackJax gives us easy access to our sampler's efficiency through metrics such as the
# sampler's _acceptance probability_ (the number of times that our chain accepted a
# proposed sample, divided by the total number of steps run by the chain).

# %%
acceptance_rate = jnp.mean(infos.acceptance_rate)
print(f"Acceptance rate: {acceptance_rate:.2f}")

# %%
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
ax0.plot(states.position.prior.kernel.lengthscale._unconstrained)
ax1.plot(states.position.prior.kernel.variance._unconstrained)
ax2.plot(states.position.latent.value[:, 1, :])
ax0.set_title("Kernel Lengthscale")
ax1.set_title("Kernel Variance")
ax2.set_title("Latent Function (index = 1)")

# %% [markdown]
# ## Prediction
#
# Having obtained samples from the posterior, we draw ten instances from our model's
# predictive distribution per MCMC sample. Using these draws, we will be able to
# compute credible values and expected values under our posterior distribution.
#
# An ideal Markov chain would have samples completely uncorrelated with their
# neighbours after a single lag. However, in practice, correlations often exist
# within our chain's sample set. A commonly used technique to try and reduce this
# correlation is _thinning_ whereby we select every $n$-th sample where $n$ is the
# minimum lag length at which we believe the samples are uncorrelated. Although further
# analysis of the chain's autocorrelation is required to find appropriate thinning
# factors, we employ a thin factor of 10 for demonstration purposes.

# %%
thin_factor = 20
posterior_samples = []

for i in range(0, num_samples, thin_factor):
    sample_params = jtu.tree_map(lambda samples, i=i: samples[i], states.position)
    model = eqx.combine(sample_params, static)
    model = paramax.unwrap(model)
    latent_dist = model.predict(xtest, train_data=D)
    predictive_dist = model.likelihood(latent_dist)
    posterior_samples.append(predictive_dist.sample(key=key, sample_shape=(10,)))

posterior_samples = jnp.vstack(posterior_samples)
lower_ci, upper_ci = jnp.percentile(posterior_samples, jnp.array([2.5, 97.5]), axis=0)
expected_val = jnp.mean(posterior_samples, axis=0)


# %% [markdown]
#
# Finally, we end this tutorial by plotting the predictions obtained from our model
# against the observed data.

# %%
fig, ax = plt.subplots()
ax.plot(
    year,
    hot_days,
    "o",
    markersize=5,
    color=cols[1],
    label="Observed counts",
    zorder=2,
    alpha=0.7,
)
ax.plot(
    year_test,
    expected_val,
    linewidth=2,
    color=cols[0],
    label=r"Posterior rate $\lambda$",
    zorder=1,
)
ax.fill_between(
    year_test,
    lower_ci.flatten(),
    upper_ci.flatten(),
    alpha=0.2,
    color=cols[0],
    label="95% CI",
)
ax.set_xlabel("year")
ax.set_ylabel("hot days (Tmax ≥ 30 °C) per year")
ax.legend()

# %% [markdown]
# The inferred rate $\lambda(\text{year})$ increases steadily across the record, tracking the rising
# number of hot days in Madrid and illustrating how a Poisson-likelihood GP recovers a smooth,
# uncertainty-aware trend from noisy annual counts.
#
# Data: ERA5 reanalysis via Open-Meteo (CC-BY).

# %% [markdown]
# ## System configuration

# %%
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Francesco Zanetta"

# %%
