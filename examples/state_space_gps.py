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
# # State-Space (Markovian) Gaussian Processes
#
# A Gaussian process whose inputs lie on the real line — a time axis — can often
# be rewritten as the solution of a linear stochastic differential equation
# (SDE). When that rewriting exists, inference no longer needs the dense
# $\mathcal{O}(N^3)$ Cholesky factorisation of the Gram matrix: a forward Kalman
# filter and backward smoother return the *exact* same posterior in
# $\mathcal{O}(N)$ time. For long, one-dimensional temporal data this turns a
# cubic problem into a linear one.
#
# In this notebook we use the `gpjax.state_space` module to model the Mauna Loa
# atmospheric CO$_2$ record. We:
#
# 1. build a state-space prior from a sum of Matérn and periodic kernels,
# 2. fit its hyperparameters and form the smoothed posterior,
# 3. fill a held-out gap with the `observation_mask` argument,
# 4. contrast the smoothed posterior with the *causal* (filtered) one, and
# 5. confirm empirically that inference scales linearly in $N$.

# %%
# Enable Float64 for more stable matrix factorisations.
from examples.utils import clean_legend, use_mpl_style
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.state_space import (
        StateSpacePrior,
        TruncatedPeriodic,
        fit_scipy,
    )


key = jr.key(123)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Background: Gaussian processes as stochastic differential equations
#
# A *Markovian* Gaussian process is one whose value at time $t$, augmented with
# a finite number of its derivatives, forms a state vector
# $\boldsymbol{s}(t) \in \mathbb{R}^d$ that evolves under a linear SDE
#
# $$ \mathrm{d}\boldsymbol{s}(t) = \mathbf{F}\,\boldsymbol{s}(t)\,\mathrm{d}t + \mathbf{L}\,\mathrm{d}\boldsymbol{\beta}(t), \qquad f(t) = \mathbf{H}\,\boldsymbol{s}(t), $$
#
# where $\boldsymbol{\beta}(t)$ is a Wiener process and $\mathbf{H}$ reads the
# function value off the state. The observations are
# $y_i = f(t_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.
# Crucially, the Matérn family admits an *exact* representation of this form
# <strong data-cite="hartikainen2010kalman"></strong>: the state dimension is
# $d = 1, 2, 3$ for the Matérn-1/2, 3/2 and 5/2 kernels respectively. Periodic
# structure is captured to arbitrary accuracy by a truncated harmonic expansion
# <strong data-cite="solin2014explicit"></strong>, and sums of these kernels
# simply stack their states.
#
# Because the process is Markovian, the marginal log-likelihood and the
# posterior are computed by a forward Kalman filter followed by a backward
# Rauch–Tung–Striebel (RTS) smoother. Each step manipulates $d \times d$
# matrices, so the whole sweep costs $\mathcal{O}(N d^3)$ — linear in the number
# of observations, against the dense path's $\mathcal{O}(N^3)$. The classic
# reference for this construction is
# <strong data-cite="sarkka2019applied"></strong>.
#
# Two structural caveats follow from requiring a *finite* state. The RBF kernel
# has no exact finite-dimensional state (its SDE is infinite order), and kernel
# *products* would multiply state dimensions in a way `gpjax.state_space` does
# not support in v1. The expressible building blocks are therefore Matérn-1/2,
# 3/2, 5/2, `TruncatedPeriodic`, and sums of these.

# %% [markdown]
# ## The Mauna Loa CO$_2$ record
#
# We load the monthly mean CO$_2$ concentration measured at the Mauna Loa
# Observatory in Hawaii. The series has a clear upward trend and a strong annual
# cycle — exactly the trend-plus-seasonality structure that a sum kernel can
# describe. We shift the time axis to start at zero (so lengthscales read in
# years) and centre the targets (so a constant mean function suffices).

# %%
co2_data = pd.read_csv(
    "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", comment="#"
)
co2_data = co2_data.loc[co2_data["average"] > 0]

time_years = co2_data["decimal date"].values
co2_ppm = co2_data["average"].values

t0 = time_years.min()
y_mean = co2_ppm.mean()

x = (time_years - t0).reshape(-1, 1)
y = (co2_ppm - y_mean).reshape(-1, 1)

D = gpx.Dataset(X=x, y=y)
print(f"Loaded N = {D.n} monthly observations spanning {x.max():.1f} years.")

fig, ax = plt.subplots(figsize=(7.5, 2.5))
ax.plot(x + t0, y + y_mean, color=cols[0], linewidth=1)
ax.set(xlabel="Year", ylabel="CO$_2$ (ppm)", title="Mauna Loa CO$_2$ record")

# %% [markdown]
# ## Building the model
#
# We compose three kernels by summation:
#
# - a long-lengthscale `Matern52` for the slow upward trend,
# - a `TruncatedPeriodic` with a one-year period for the seasonal cycle, and
# - a short-lengthscale `Matern32` for medium-scale wiggles.
#
# The only structural difference from a dense GPJax model is the prior: we use
# `StateSpacePrior` in place of `gpx.gps.Prior`. Everything else — the kernels,
# the constant mean function, the Gaussian likelihood, and the `*` operator that
# forms the posterior — is the usual GPJax API.

# %%
trend_kernel = gpx.kernels.Matern52(lengthscale=20.0, variance=100.0)
seasonal_kernel = TruncatedPeriodic(
    lengthscale=1.0, variance=2.0, period=1.0, truncation_order=6
)
short_kernel = gpx.kernels.Matern32(lengthscale=1.0, variance=1.0)
kernel = trend_kernel + seasonal_kernel + short_kernel

prior = StateSpacePrior(
    mean_function=gpx.mean_functions.Constant(),
    kernel=kernel,
)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.5)
posterior = prior * likelihood

# %% [markdown]
# ## Fitting
#
# The `gpjax.state_space` module ships fitting wrappers that mirror the dense
# API. We use `fit_scipy`, which optimises the state-space marginal
# log-likelihood with SciPy's L-BFGS-B; it validates and sorts the inputs and
# threads the (here trivial) observation mask through the objective for us. For
# large or mini-batched problems the module also provides an Optax-based `fit`.

# %%
opt_posterior, history = fit_scipy(
    model=posterior,
    train_data=D,
    verbose=False,
)

# %% [markdown]
# ## Smoothed prediction
#
# Calling `predict` returns the RTS-smoothed posterior: each test point
# conditions on the *entire* observation record, past and future. The marginal
# variances are exact. We predict on a dense grid spanning the data and a short
# extrapolation beyond it.

# %%
xtest = jnp.linspace(0.0, float(x.max()) + 3.0, 600).reshape(-1, 1)

smoothed = opt_posterior.predict(xtest, D)
smoothed_mean = smoothed.mean + y_mean
smoothed_std = jnp.sqrt(smoothed.variance)

fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(x + t0, y + y_mean, "x", color=cols[0], alpha=0.3, label="Observations")
ax.plot(xtest + t0, smoothed_mean, color=cols[1], label="Smoothed mean")
ax.fill_between(
    (xtest + t0).squeeze(),
    smoothed_mean - 2 * smoothed_std,
    smoothed_mean + 2 * smoothed_std,
    color=cols[1],
    alpha=0.2,
    label="Two sigma",
)
ax.set(xlabel="Year", ylabel="CO$_2$ (ppm)")
ax.legend(loc="upper left")
clean_legend(ax)

# %% [markdown]
# ## Gap-filling with an observation mask
#
# Real records have gaps. The state-space predictive accepts an
# `observation_mask` — a boolean vector over the training points — that excludes
# masked observations from the filter updates while still propagating the state
# through them. This yields a principled interpolation across the gap, with the
# posterior uncertainty widening over the unobserved interval and tightening
# again at the edges where data resumes.
#
# Here we mask a contiguous five-year interior interval and predict across it.
# We zoom in on the masked window. The widening is modest, and deliberately so:
# the trend and seasonal components are *global*, so even an unobserved stretch
# stays constrained by the locked phase of the annual cycle and the
# slowly-varying trend.

# %%
gap_lo, gap_hi = 30.0, 35.0
observation_mask = ~((x.squeeze() >= gap_lo) & (x.squeeze() < gap_hi))

xgap = jnp.linspace(gap_lo - 6.0, gap_hi + 6.0, 400).reshape(-1, 1)
in_gap = (xgap.squeeze() >= gap_lo) & (xgap.squeeze() < gap_hi)

gap_pred = opt_posterior.predict(xgap, D, observation_mask=observation_mask)
gap_mean = gap_pred.mean + y_mean
gap_std = jnp.sqrt(gap_pred.variance)

fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(
    (x + t0).squeeze()[observation_mask],
    (y + y_mean).squeeze()[observation_mask],
    "x",
    color=cols[0],
    alpha=0.3,
    label="Retained observations",
)
ax.plot(
    (x + t0).squeeze()[~observation_mask],
    (y + y_mean).squeeze()[~observation_mask],
    "x",
    color="grey",
    alpha=0.5,
    label="Masked observations",
)
ax.plot(xgap + t0, gap_mean, color=cols[1], label="Posterior mean")
ax.fill_between(
    (xgap + t0).squeeze(),
    gap_mean - 2 * gap_std,
    gap_mean + 2 * gap_std,
    color=cols[1],
    alpha=0.2,
    label="Two sigma",
)
ax.axvspan(gap_lo + t0, gap_hi + t0, color="grey", alpha=0.1)
ax.set(xlabel="Year", ylabel="CO$_2$ (ppm)", xlim=(t0 + gap_lo - 6, t0 + gap_hi + 6))
ax.legend(loc="upper left")
clean_legend(ax)

print(
    "Mean two-sigma width inside the gap: "
    f"{float(4 * gap_std[in_gap].mean()):.2f} ppm; outside: "
    f"{float(4 * gap_std[~in_gap].mean()):.2f} ppm."
)

# %% [markdown]
# ## Causal prediction: filtering versus smoothing
#
# The smoother estimates the latent function using *all* the data. In an online
# setting we instead want the *causal* estimate at each time — conditioning only
# on observations up to and including that time. The state-space posterior
# exposes this through `predict_filter`, which reads marginals off the forward
# Kalman trajectory rather than the backward smoother.
#
# On the dense, low-noise CO$_2$ record the two are nearly identical wherever
# data is plentiful. The gap is where causality bites: inside the masked
# interval the filter has only *pre-gap* data, so its mean drifts and its
# uncertainty grows steadily across the window, whereas the smoother — informed
# by the data on *both* sides — stays tight. We reuse the same mask and overlay
# the two predictives.

# %%
filtered_gap = opt_posterior.predict_filter(xgap, D, observation_mask=observation_mask)
filtered_mean = filtered_gap.mean + y_mean
filtered_std = jnp.sqrt(filtered_gap.variance)

fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.plot(
    (x + t0).squeeze()[observation_mask],
    (y + y_mean).squeeze()[observation_mask],
    "x",
    color=cols[0],
    alpha=0.25,
    label="Retained observations",
)
ax.plot(xgap + t0, gap_mean, color=cols[1], label="Smoothed (both sides)")
ax.fill_between(
    (xgap + t0).squeeze(),
    gap_mean - 2 * gap_std,
    gap_mean + 2 * gap_std,
    color=cols[1],
    alpha=0.18,
)
ax.plot(xgap + t0, filtered_mean, color=cols[2], label="Filtered (causal)")
ax.fill_between(
    (xgap + t0).squeeze(),
    filtered_mean - 2 * filtered_std,
    filtered_mean + 2 * filtered_std,
    color=cols[2],
    alpha=0.18,
)
ax.axvspan(gap_lo + t0, gap_hi + t0, color="grey", alpha=0.1)
ax.set(xlabel="Year", ylabel="CO$_2$ (ppm)", xlim=(t0 + gap_lo - 6, t0 + gap_hi + 6))
ax.legend(loc="upper left")
clean_legend(ax)

print(
    "Inside the gap — smoothed two-sigma: "
    f"{float(4 * gap_std[in_gap].mean()):.2f} ppm; "
    f"filtered two-sigma: {float(4 * filtered_std[in_gap].mean()):.2f} ppm."
)

# %% [markdown]
# ## Does it scale?
#
# The motivation for all of this machinery is linear-time inference. We confirm
# it empirically by timing the forward marginal log-likelihood and its gradient
# for the dense and state-space paths over a range of $N$, on synthetic data
# drawn so that both paths see identical inputs. JAX compiles lazily, so we warm
# each function up once and report the minimum of three timed runs.
#
# This in-notebook sweep is deliberately small and only illustrates the slopes;
# the maintained, rigorous benchmarks live in `benchmarks/state_space.py` (run
# with [`asv`](https://asv.readthedocs.io/)).

# %%
import time

from gpjax.state_space import state_space_mll


def block_pytree(pytree):
    for leaf in jax.tree_util.tree_leaves(pytree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_function(fn, num_warmup=1, num_runs=3):
    for _ in range(num_warmup):
        block_pytree(fn())
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn()
        block_pytree(result)
        timings.append(time.perf_counter() - start)
    return min(timings)


def simulate_dataset(n, seed=0):
    sim_key = jr.key(seed)
    key_x, key_f = jr.split(sim_key)
    inputs = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=50.0))
    targets = jnp.sin(inputs) + 0.1 * jr.normal(key_f, shape=(n,))
    return inputs.reshape(-1, 1), targets.reshape(-1, 1)


def make_loss_callables(num_datapoints, state_space):
    bench_kernel = gpx.kernels.Matern52(lengthscale=1.0, variance=1.0)
    prior_cls = StateSpacePrior if state_space else gpx.gps.Prior
    bench_prior = prior_cls(
        mean_function=gpx.mean_functions.Zero(), kernel=bench_kernel
    )
    bench_lik = gpx.likelihoods.Gaussian(num_datapoints=num_datapoints, obs_stddev=0.1)
    model = bench_prior * bench_lik
    bench_x, bench_y = simulate_dataset(num_datapoints)
    data = gpx.Dataset(X=bench_x, y=bench_y)

    if state_space:
        loss = lambda m: -state_space_mll(m, data)
    else:
        loss = lambda m: -gpx.objectives.conjugate_mll(m, data)

    forward_fn = jax.jit(loss)
    grad_fn = jax.jit(jax.grad(loss))
    return lambda: forward_fn(model), lambda: grad_fn(model)


N_VALUES = [200, 500, 1000, 2000, 5000]
DENSE_N_LIMIT = 2000

results = {"dense": {"forward": {}, "grad": {}}, "ss": {"forward": {}, "grad": {}}}

for n in N_VALUES:
    ss_forward, ss_grad = make_loss_callables(n, state_space=True)
    results["ss"]["forward"][n] = time_function(ss_forward)
    results["ss"]["grad"][n] = time_function(ss_grad)
    if n <= DENSE_N_LIMIT:
        dense_forward, dense_grad = make_loss_callables(n, state_space=False)
        results["dense"]["forward"][n] = time_function(dense_forward)
        results["dense"]["grad"][n] = time_function(dense_grad)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, op, op_title in zip(
    axes, ["forward", "grad"], ["Forward MLL", "MLL gradient"], strict=True
):
    dense_xs = sorted(results["dense"][op])
    ss_xs = sorted(results["ss"][op])
    ax.loglog(
        dense_xs,
        [results["dense"][op][n] for n in dense_xs],
        marker="o",
        color=cols[0],
        label="Dense GP",
    )
    ax.loglog(
        ss_xs,
        [results["ss"][op][n] for n in ss_xs],
        marker="s",
        color=cols[1],
        label="State-space GP",
    )
    ax.set(xlabel="N", ylabel="Runtime (s)", title=op_title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

# %% [markdown]
# The dense curves bend upwards — the signature of cubic scaling — while the
# state-space curves track a near-straight line of slope one. For small $N$ the
# dense path is actually faster: a tiny LAPACK Cholesky fits in cache and beats
# the fixed launch and control-flow overhead of the Kalman scan. The state-space
# path earns its keep as $N$ grows, and remains tractable far beyond where the
# dense Gram matrix stops fitting in memory.

# %% [markdown]
# ## Summary
#
# - For one-dimensional temporal data, `gpjax.state_space` replaces the dense
#   $\mathcal{O}(N^3)$ Cholesky with a linear-time Kalman filter and RTS
#   smoother, returning the *same* posterior.
# - The expressible kernels are `Matern12`, `Matern32`, `Matern52`,
#   `TruncatedPeriodic`, and sums thereof; building a `StateSpacePrior` is
#   otherwise identical to a dense `Prior`.
# - `predict` gives the smoothed posterior, `predict_filter` the causal
#   (filtered) one, and the `observation_mask` argument enables principled
#   gap-filling.
# - Dense inference is preferable for small $N$; state-space inference wins
#   decisively once $N$ grows into the thousands and beyond.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
