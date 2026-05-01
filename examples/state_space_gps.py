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
# # State-Space GPs: Runtime Scaling Comparison
#
# This notebook benchmarks the dense Gaussian process implementation against
# the state-space (Markovian) implementation provided by `gpjax.state_space`.
# Dense GP inference scales as $\mathcal{O}(N^3)$ in the number of
# observations, while state-space inference scales as $\mathcal{O}(N \cdot
# d^3)$ where $d$ is the latent state dimension. For 1-D temporal data with
# moderate state dimension, this turns a cubic scaling problem into a linear
# one.
#
# We measure three quantities at increasing $N$:
#
# 1. forward marginal log-likelihood (MLL) evaluation,
# 2. reverse-mode gradient of the MLL with respect to the parameters,
# 3. posterior prediction at $M = 200$ test points.
#
# We then visualise the runtime curves on log-log axes and confirm that the
# two methods produce equivalent posteriors at a moderate $N$.

# %%
import time

from examples.utils import use_mpl_style
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

import gpjax as gpx
from gpjax.state_space import (
    StateSpacePrior,
    state_space_mll,
)

key = jr.key(123)
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Simulating data
#
# We draw a single ground-truth function from a `Matern52` prior at
# $N_{\max} = 10\,000$ sorted timestamps in $[0, 50]$, observed under
# Gaussian noise with standard deviation $0.1$. Smaller-$N$ benchmarks reuse
# the leading prefix of the same dataset, keeping the comparison
# apples-to-apples across sizes.
#
# Note: the simulation itself uses an $\mathcal{O}(N^3)$ Cholesky factor
# of the prior covariance. At $N = 10^4$ this is the most expensive step in
# the notebook; we pay it once.

# %%
LENGTHSCALE_TRUE = 5.0
VARIANCE_TRUE = 1.0
OBS_STDDEV_TRUE = 0.1
N_MAX = 10_000


def simulate_dataset(n, seed=0):
    sim_key = jr.key(seed)
    key_x, key_f, key_eps = jr.split(sim_key, 3)
    inputs = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=50.0))
    kernel = gpx.kernels.Matern52(lengthscale=LENGTHSCALE_TRUE, variance=VARIANCE_TRUE)
    gram = kernel.gram(inputs.reshape(-1, 1)).as_matrix()
    chol = jnp.linalg.cholesky(gram + 1e-6 * jnp.eye(n))
    latent = chol @ jr.normal(key_f, shape=(n,))
    targets = latent + OBS_STDDEV_TRUE * jr.normal(key_eps, shape=(n,))
    return inputs.reshape(-1, 1), targets.reshape(-1, 1)


X_full, y_full = simulate_dataset(N_MAX, seed=0)
print(f"Simulated N = {N_MAX} observations on [0, 50].")

# %% [markdown]
# ## Building dense and state-space posteriors
#
# Both implementations share the same kernel class, mean function and
# likelihood. The only structural difference is the `Prior`: the dense path
# uses `gpx.gps.Prior`, while the state-space path uses
# `gpjax.state_space.StateSpacePrior`. The latent state dimension for
# `Matern52` is $d = 3$.


# %%
def build_dense_posterior(num_datapoints):
    kernel = gpx.kernels.Matern52(lengthscale=1.0, variance=1.0)
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel,
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=num_datapoints, obs_stddev=0.1)
    return prior * likelihood


def build_state_space_posterior(num_datapoints):
    kernel = gpx.kernels.Matern52(lengthscale=1.0, variance=1.0)
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel,
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=num_datapoints, obs_stddev=0.1)
    return prior * likelihood


# %% [markdown]
# ## Timing methodology
#
# JAX compiles functions lazily on first call, so a naive timing call would
# capture compilation cost rather than runtime. We:
#
# - run each function once as a warm-up, blocking on the result,
# - then time several invocations, again blocking, and report the minimum
#   to suppress system jitter,
# - call `block_until_ready()` (or recurse into tuples / pytrees) so that
#   the asynchronous JAX dispatch completes before we stop the clock.


# %%
def block_pytree(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_function(fn, num_warmup=1, num_runs=3):
    """Warm-up `num_warmup` times, then time `num_runs` invocations.

    Returns the minimum elapsed time across timed runs.
    """
    for _ in range(num_warmup):
        block_pytree(fn())
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn()
        block_pytree(result)
        timings.append(time.perf_counter() - start)
    return min(timings)


# %% [markdown]
# ## Sweep
#
# We sweep over a range of $N$ values. The dense path is only run up to
# $N = 5000$ to keep the docs CI runtime reasonable; the state-space path
# is run for the full sweep up to $N = 10\,000$.

# %%
N_VALUES = [200, 500, 1000, 2000, 5000, 10_000, 20_000]
DENSE_N_LIMIT = 5000
M_TEST = 200

dense_results = {"forward": {}, "grad": {}, "predict": {}}
ss_results = {"forward": {}, "grad": {}, "predict": {}}


def make_dense_callables(num_datapoints):
    posterior = build_dense_posterior(num_datapoints)
    data = gpx.Dataset(X=X_full[:num_datapoints], y=y_full[:num_datapoints])
    test_x = jnp.linspace(0.0, 50.0, M_TEST).reshape(-1, 1)

    def loss(model):
        return -gpx.objectives.conjugate_mll(model, data)

    forward_fn = jax.jit(loss)
    grad_fn = jax.jit(jax.grad(loss))

    def predict_raw(model):
        predictive = model.predict(test_x, data)
        return predictive.mean, predictive.variance

    predict_fn = jax.jit(predict_raw)

    return (
        lambda: forward_fn(posterior),
        lambda: grad_fn(posterior),
        lambda: predict_fn(posterior),
    )


def make_state_space_callables(num_datapoints):
    posterior = build_state_space_posterior(num_datapoints)
    data = gpx.Dataset(X=X_full[:num_datapoints], y=y_full[:num_datapoints])
    test_x = jnp.linspace(0.0, 50.0, M_TEST).reshape(-1, 1)

    def loss(model):
        return -state_space_mll(model, data)

    forward_fn = jax.jit(loss)
    grad_fn = jax.jit(jax.grad(loss))

    def predict_raw(model):
        predictive = model.predict(test_x, data)
        return predictive.mean, predictive.variance

    predict_fn = jax.jit(predict_raw)

    return (
        lambda: forward_fn(posterior),
        lambda: grad_fn(posterior),
        lambda: predict_fn(posterior),
    )


for n in N_VALUES:
    print(f"--- N = {n} ---")
    ss_forward, ss_grad, ss_predict = make_state_space_callables(n)
    ss_results["forward"][n] = time_function(ss_forward)
    ss_results["grad"][n] = time_function(ss_grad)
    ss_results["predict"][n] = time_function(ss_predict)
    print(
        f"  state-space  forward: {ss_results['forward'][n]:.4f}s"
        f"  grad: {ss_results['grad'][n]:.4f}s"
        f"  predict: {ss_results['predict'][n]:.4f}s"
    )

    if n <= DENSE_N_LIMIT:
        dense_forward, dense_grad, dense_predict = make_dense_callables(n)
        dense_results["forward"][n] = time_function(dense_forward)
        dense_results["grad"][n] = time_function(dense_grad)
        dense_results["predict"][n] = time_function(dense_predict)
        print(
            f"  dense        forward: {dense_results['forward'][n]:.4f}s"
            f"  grad: {dense_results['grad'][n]:.4f}s"
            f"  predict: {dense_results['predict'][n]:.4f}s"
        )
    else:
        print(f"  dense        skipped (N > {DENSE_N_LIMIT})")

# %% [markdown]
# ## Runtime curves
#
# We plot the timings on log-log axes. A method that scales as $N^p$ shows
# up as a straight line of slope $p$. Dense GPs are expected to have slope
# $\approx 3$ (cubic Cholesky), while state-space inference is expected to
# have slope $\approx 1$ (linear Kalman filter, fixed state dimension).

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
operation_titles = [
    ("forward", "Forward MLL"),
    ("grad", "MLL gradient"),
    ("predict", "Posterior predict"),
]
for ax, (op_key, op_title) in zip(axes, operation_titles, strict=True):
    dense_xs = sorted(dense_results[op_key].keys())
    dense_ys = [dense_results[op_key][n] for n in dense_xs]
    ss_xs = sorted(ss_results[op_key].keys())
    ss_ys = [ss_results[op_key][n] for n in ss_xs]
    ax.loglog(
        dense_xs,
        dense_ys,
        marker="o",
        color=cols[0],
        label="Dense GP",
    )
    ax.loglog(
        ss_xs,
        ss_ys,
        marker="s",
        color=cols[1],
        label="State-space GP",
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(op_title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
fig.tight_layout()

# %% [markdown]
# The dense curves bend upwards: doubling $N$ multiplies runtime by roughly
# eight, the signature of cubic scaling. The state-space curves track a
# straight line with slope close to one. Beyond $N \approx 5000$ the dense
# path becomes uncomfortable on a laptop, while state-space inference
# remains tractable well past $N = 10^4$.

# %% [markdown]
# ## Why does the dense path win for small $N$?
#
# The two curves cross at $N \approx 10^3$. For smaller datasets the dense
# implementation is faster, despite its asymptotically worse complexity.
# Three effects combine to produce this crossover:
#
# 1. **Constant per-step overhead in the scan.** Each step of the Kalman
#    filter performs a small `discretise(dt)` call, two QR factorisations
#    (in `_sqrt_predict` and `_sqrt_update`), a sign normalisation and a
#    `lax.cond` for the observation mask. The total runtime is approximately
#    $c \cdot N$ with a non-trivial constant $c$.
# 2. **Two nested `lax.scan`s plus `jax.checkpoint`.** `kalman_filter`
#    runs an outer scan over chunks of size $\sqrt{N}$ and an inner scan
#    within each chunk, with the inner scan wrapped in a checkpoint to
#    bound reverse-mode AD memory at $\mathcal{O}(\sqrt{N} \cdot d^2)$.
#    At $N = 200$ this overhead buys nothing useful; we still pay it.
# 3. **LAPACK Cholesky is extremely fast on small matrices.** A
#    $200 \times 200$ Cholesky is roughly five million flops, fits in
#    cache, and ships off to a single hand-tuned BLAS kernel. XLA can
#    also fuse the dense path's matrix multiplications more aggressively
#    than it can fuse a sequential scan body, where data dependencies
#    between steps block parallelisation.
#
# The takeaway is that state-space inference is paying a fixed launch and
# control-flow overhead in exchange for linear scaling. For $N \lesssim
# 10^3$, the dense path is strictly preferable. For $N \gtrsim 10^4$, the
# dense path is infeasible regardless: at $N = 10^5$ the gram matrix alone
# requires $\sim 80$ GB of memory.

# %% [markdown]
# ## Sanity check: do the two methods agree?
#
# At a moderate sample size ($N = 1000$) we fit both models and compare
# their posterior means and credible intervals on a shared test grid. The
# two posteriors should be visually indistinguishable apart from numerical
# rounding.

# %%
import optax as ox

N_FIT = 1000
fit_data = gpx.Dataset(X=X_full[:N_FIT], y=y_full[:N_FIT])

dense_posterior = build_dense_posterior(N_FIT)
ss_posterior = build_state_space_posterior(N_FIT)

fitted_dense, _ = gpx.fit(
    model=dense_posterior,
    objective=lambda m, d: -gpx.objectives.conjugate_mll(m, d),
    train_data=fit_data,
    optim=ox.adam(learning_rate=5e-2),
    num_iters=200,
    key=key,
    verbose=False,
)
fitted_ss, _ = gpx.fit(
    model=ss_posterior,
    objective=lambda m, d: -state_space_mll(m, d),
    train_data=fit_data,
    optim=ox.adam(learning_rate=5e-2),
    num_iters=200,
    key=key,
    verbose=False,
)

test_x = jnp.linspace(0.0, 50.0, 400).reshape(-1, 1)
dense_predictive = fitted_dense.predict(test_x, fit_data)
ss_predictive = fitted_ss.predict(test_x, fit_data)

dense_mean = dense_predictive.mean
dense_std = jnp.sqrt(dense_predictive.variance)
ss_mean = ss_predictive.mean
ss_std = jnp.sqrt(ss_predictive.variance)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

for ax, mean, std, title, colour in zip(
    axes,
    [dense_mean, ss_mean],
    [dense_std, ss_std],
    ["Dense GP", "State-space GP"],
    [cols[0], cols[1]],
    strict=True,
):
    ax.plot(
        fit_data.X.squeeze(),
        fit_data.y.squeeze(),
        "x",
        color="grey",
        alpha=0.3,
        label="Observations",
    )
    ax.fill_between(
        test_x.squeeze(),
        mean - 2 * std,
        mean + 2 * std,
        color=colour,
        alpha=0.2,
        label="Two sigma",
    )
    ax.plot(test_x.squeeze(), mean, color=colour, label=f"{title} mean")
    ax.set_xlabel("t")
    ax.set_title(title)
    ax.legend(loc="best")
axes[0].set_ylabel("y")
fig.tight_layout()

# %% [markdown]
# The two posteriors agree closely: the means trace the same curve and the
# uncertainty ribbons overlap. The minor differences come from the two
# optimisers settling at slightly different hyperparameter values.

# %% [markdown]
# ## Summary
#
# - State-space inference replaces the dense $\mathcal{O}(N^3)$ Cholesky
#   factorisation with a linear-time Kalman filter, making 1-D temporal
#   GPs with $N \gtrsim 10^4$ tractable on commodity hardware.
# - For the supported kernel families (`Matern12`, `Matern32`, `Matern52`,
#   `TruncatedPeriodic`, and sums thereof), `gpjax.state_space` is a
#   drop-in replacement for the dense path.
# - The two posteriors are equivalent in distribution, modulo numerical
#   precision and optimiser noise, so the only practical decision is one of
#   problem size: dense for small $N$, state-space for large $N$.
# - The state-space posterior also provides `predict_filter` for causal
#   (filtered) predictions and accepts an `observation_mask` argument for
#   principled gap-filling, which are not the focus of this notebook.
