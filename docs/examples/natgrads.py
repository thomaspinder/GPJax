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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Natural Gradients in Practice
#
# Download this notebook: {nb-download}`natgrads.ipynb`
#
# This notebook assumes the
# [natural gradients notebook](natural_gradients.py) throughout: the
# exponential-family view of $q(\mathbf{u})$, the Fisher=Jacobian identity
# that makes the natural gradient free to compute, the mirror-descent
# reading of the step, the "one step is enough" theorem for conjugate
# models, and the cone-safety theorem with its proof. None of that is
# re-derived here — this notebook connects that geometry to the GPJax API
# instead: `gpx.fit_natgrads`, the lower-level `natural_gradient_step`,
# `partition_variational`, and `WhitenedVariationalGaussian`, on two real
# training runs, checking the theory's predictions against what actually
# happens on this machine.
#
# The route is:
#
# 1. **demo (i)** — a conjugate 1D regression where a single $\gamma=1$ step
#    lands on the exact variational optimum, while Adam is still crawling
#    after two thousand iterations;
# 2. **demo (ii)** — a mini-batched Bernoulli classification benchmark,
#    comparing natural gradients + Adam against Adam alone, per iteration
#    *and* per second;
# 3. the failure mode: what a large $\gamma$ does to the
#    $\boldsymbol{\Theta}_2$ cone, and how the built-in step-size backoff
#    behaves.
#
# If you have not met sparse variational GPs before, read the
# [stochastic sparse GP notebook](uncollapsed_vi.py)
# first — everything below assumes the SVGP evidence lower bound.

# %%
# Enable Float64 for more stable matrix inversions.
import time

import equinox as eqx
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import paramax
from utils import clean_legend, use_mpl_style

config.update("jax_enable_x64", True)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    import gpjax.kernels as jk
    from gpjax.natural_gradients import (
        expectation_from_moments,
        moments_from_expectation,
        natural_from_moments,
        natural_gradient_step,
        partition_variational,
    )
    from gpjax.parameters import LowerTriangular, Real

key = jr.key(123)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


def negative_elbo(model, data):
    """The loss every fit below minimises: GPJax optimisers descend, so negate."""
    return -gpx.objectives.elbo(model, data)


# %% [markdown]
# ## Demo (i): conjugate regression
#
# Recall from the natural gradients notebook that for a conditionally
# conjugate model — here, a Gaussian likelihood — the ELBO is affine in the
# expectation parameters, so the step collapses to
# $\boldsymbol{\theta}_{\text{new}} = (1-\gamma)\,\boldsymbol{\theta} + \gamma\,\boldsymbol{\lambda}$
# for a fixed $\boldsymbol{\lambda}$ that does not depend on $q$. At
# $\gamma=1$ this is not an approximation to the optimum, it *is* the
# optimum: $\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\lambda} = \boldsymbol{\theta}^\star$,
# reached in one step from any starting point (the "one step is enough"
# theorem there, after Sato 2001; for the SVGP it recovers the
# {cite:t}`titsias2009` optimum). We watch that happen, then race it against
# Adam from the same bad start.

# %%
num_data = 200
noise_stddev = 0.3

key, input_key, noise_key = jr.split(key, 3)
regression_inputs = jr.uniform(input_key, (num_data, 1), minval=-3.0, maxval=3.0)
regression_signal = jnp.sin(2.0 * regression_inputs)
regression_outputs = regression_signal + noise_stddev * jr.normal(
    noise_key, regression_signal.shape
)
regression_data = gpx.Dataset(X=regression_inputs, y=regression_outputs)

num_inducing = 20
regression_inducing = jnp.linspace(-3.0, 3.0, num_inducing).reshape(-1, 1)
test_inputs = jnp.linspace(-3.2, 3.2, 300).reshape(-1, 1)

# %%
# A conjugate SVGP, deliberately initialised a long way from its optimum. The
# joint model is prior * likelihood; the variational family approximates its
# posterior.
regression_model = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Constant(),
    kernel=jk.RBF(lengthscale=0.5),
    jitter=1e-8,
) * gpx.likelihoods.Gaussian(obs_stddev=noise_stddev)

key, bad_mean_key, bad_root_key = jr.split(key, 3)
bad_mean = jr.normal(bad_mean_key, (num_inducing, 1))
bad_factor = 0.3 * jr.normal(bad_root_key, (num_inducing, num_inducing))
bad_root = jnp.linalg.cholesky(bad_factor @ bad_factor.T + 0.5 * jnp.eye(num_inducing))

initial_family = gpx.variational_families.WhitenedVariationalGaussian(
    model=regression_model,
    inducing_inputs=regression_inducing,
    variational_mean=bad_mean,
    variational_root_covariance=bad_root,
)

# %% [markdown]
# We use the **whitened** family here, which reparameterises
# $\mathbf{u} = \boldsymbol{\mu}_z + \mathbf{L}_z\mathbf{v}$ with
# $\mathbf{L}_z\mathbf{L}_z^\top = \mathbf{K}_{zz}$ and puts a
# $\mathcal{N}(\mathbf{0},\mathbf{I})$ prior on $\mathbf{v}$. The
# natural-gradient machinery is untouched by this — $q(\mathbf{v})$ belongs
# to the same exponential family as $q(\mathbf{u})$, and the whitening enters
# only through `prior_kl` and `predict`, which the loss calls
# polymorphically. Numerically it helps a great deal, because $\mathbf{m}_w$
# and $\mathbf{S}_w$ are $\mathcal{O}(1)$ regardless of the kernel scale, and
# the conjugate optimum satisfies $\mathbf{S}_w^\star \preceq \mathbf{I}$.
#
# For the whitened family the closed-form optimum is, with
# $\mathbf{A}_w = \mathbf{K}_{xz}\mathbf{L}_z^{-\top}$ and
# $\sigma^2$ the observation variance,
#
# $$\boldsymbol{\Lambda}_w = \mathbf{I}_M + \sigma^{-2}\mathbf{A}_w^\top\mathbf{A}_w, \qquad \mathbf{b}_w = \sigma^{-2}\mathbf{A}_w^\top(\mathbf{y}-\boldsymbol{\mu}_x),$$
# $$\mathbf{S}_w^\star = \boldsymbol{\Lambda}_w^{-1}, \qquad \mathbf{m}_w^\star = \boldsymbol{\Lambda}_w^{-1}\mathbf{b}_w .$$
#
# This is used only as a reference value below, computed once with plain
# linear algebra so that the natural-gradient step has something exact to be
# checked against.

# %%
unwrapped_initial = paramax.unwrap(initial_family)
kernel = unwrapped_initial.model.prior.kernel
mean_function = unwrapped_initial.model.prior.mean_function

Kzz = kernel.gram(regression_inducing).as_matrix()
Kzz = Kzz + initial_family.model.prior.jitter * jnp.eye(num_inducing)
Lz = jnp.linalg.cholesky(Kzz)
Kzx = kernel.cross_covariance(regression_inducing, regression_inputs)
whitened_design = jax.scipy.linalg.solve_triangular(Lz, Kzx, lower=True).T

observation_variance = noise_stddev**2
whitened_precision = (
    jnp.eye(num_inducing) + whitened_design.T @ whitened_design / observation_variance
)
whitened_shift = (
    whitened_design.T
    @ (regression_outputs - mean_function(regression_inputs))
    / observation_variance
)
optimal_covariance = jnp.linalg.inv(whitened_precision)
optimal_mean = jnp.linalg.solve(whitened_precision, whitened_shift)

# The ELBO at the closed-form optimum, used below as the reference for both
# methods.
optimal_family = eqx.tree_at(
    lambda family: (family.variational_mean, family.variational_root_covariance),
    initial_family,
    (Real(optimal_mean), LowerTriangular(jnp.linalg.cholesky(optimal_covariance))),
)
reference_elbo = float(
    gpx.objectives.elbo(paramax.unwrap(optimal_family), regression_data)
)
print(f"ELBO at the closed-form optimum: {reference_elbo:.6f}")

# %%
# One natural-gradient step at gamma = 1. `partition_variational` splits the
# family into the pytree leaves the step is allowed to touch (the
# variational parameters) and everything else (the hyperparameters); the
# step is exactly what `fit_natgrads` calls once per iteration.
variational_partition, hyper_partition = partition_variational(initial_family)
stepped_partition, loss_before = natural_gradient_step(
    variational_partition,
    hyper_partition,
    regression_data,
    negative_elbo,
    1.0,
    map_jitter=0.0,
)
stepped_family = eqx.combine(stepped_partition, hyper_partition)

unwrapped_stepped = paramax.unwrap(stepped_family)
stepped_mean = unwrapped_stepped.variational_mean
stepped_root = unwrapped_stepped.variational_root_covariance
stepped_covariance = stepped_root @ stepped_root.T

stepped_elbo = float(gpx.objectives.elbo(unwrapped_stepped, regression_data))

# A second step from the same place must be a fixed point.
twice_stepped_partition, _ = natural_gradient_step(
    stepped_partition,
    hyper_partition,
    regression_data,
    negative_elbo,
    1.0,
    map_jitter=0.0,
)
twice_stepped_mean = paramax.unwrap(
    eqx.combine(twice_stepped_partition, hyper_partition)
).variational_mean

print(f"ELBO before the step           : {-loss_before:12.6f}")
print(f"ELBO after one gamma=1 step    : {stepped_elbo:12.6f}")
print(f"ELBO at the closed-form optimum: {reference_elbo:12.6f}")
print(
    "max |m_1 - m*|                 : "
    f"{jnp.max(jnp.abs(stepped_mean - optimal_mean)):.3e}"
)
print(
    "max |S_1 - S*|                 : "
    f"{jnp.max(jnp.abs(stepped_covariance - optimal_covariance)):.3e}"
)
print(
    "max |m_2 - m_1| (fixed point)  : "
    f"{jnp.max(jnp.abs(twice_stepped_mean - stepped_mean)):.3e}"
)

# %% [markdown]
# One step, from a random initialisation more than 3000 ELBO nats away, lands
# on the closed-form optimum to $\sim10^{-13}$ in both the mean and the
# covariance — the float64 noise floor for a problem of this size — and the
# ELBO itself matches to all six printed decimal places. A second step moves
# the mean by the same $\sim10^{-14}$, confirming the fixed point.
#
# Notice the `map_jitter=0.0` keyword: the jitter used inside the
# $\boldsymbol{\theta}\leftrightarrow\boldsymbol{\xi}$ maps is a *bias*, not
# a rounding effect, since
# $(\mathbf{S}^{-1}+\varepsilon\mathbf{I})^{-1} = \mathbf{S} - \varepsilon\mathbf{S}^2 + \mathcal{O}(\varepsilon^2)$.
# It defaults to zero in `fit_natgrads` for exactly that reason, and is
# deliberately *not* inherited from the model's `Prior.jitter`, which is a
# different quantity applied to $\mathbf{K}_{zz}$.
#
# Because this model is conjugate, we can also compare the one-step
# posterior against the exact GP posterior, obtained by conditioning the
# joint model on the data with no inducing-point approximation at all.

# %%
exact_posterior = paramax.unwrap(regression_model).condition(regression_data)
exact_predictive = exact_posterior(test_inputs)
exact_mean = exact_predictive.mean
exact_stddev = jnp.sqrt(exact_predictive.variance)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3.0), sharey=True)
for ax, family, title in [
    (axes[0], unwrapped_initial, "Initialisation"),
    (axes[1], unwrapped_stepped, "After one $\\gamma=1$ natural-gradient step"),
]:
    predictive = family(test_inputs)
    predictive_mean = predictive.mean
    predictive_stddev = jnp.sqrt(predictive.variance)
    ax.scatter(
        regression_inputs,
        regression_outputs,
        alpha=0.2,
        s=8,
        color=cols[0],
        label="Observations",
    )
    ax.plot(
        test_inputs, exact_mean, color="black", linestyle="--", label="Exact posterior"
    )
    ax.fill_between(
        test_inputs.flatten(),
        exact_mean - 2 * exact_stddev,
        exact_mean + 2 * exact_stddev,
        alpha=0.15,
        color="black",
    )
    ax.plot(test_inputs, predictive_mean, color=cols[1], label="Variational $q$")
    ax.fill_between(
        test_inputs.flatten(),
        predictive_mean - 2 * predictive_stddev,
        predictive_mean + 2 * predictive_stddev,
        alpha=0.3,
        color=cols[1],
    )
    ax.set(xlabel=r"$x$", title=title, ylim=(-3.0, 3.0))
    clean_legend(ax)
axes[0].set_ylabel(r"$f(x)$")

data_range_mask = (test_inputs[:, 0] >= -3.0) & (test_inputs[:, 0] <= 3.0)
print(
    "max |sparse mean - exact mean|, full grid [-3.2,3.2]   : "
    f"{jnp.max(jnp.abs(unwrapped_stepped(test_inputs).mean - exact_mean)):.3e}"
)
print(
    "max |sparse mean - exact mean|, data range [-3,3]      : "
    f"{jnp.max(jnp.abs((unwrapped_stepped(test_inputs).mean - exact_mean)[data_range_mask])):.3e}"
)

# %% [markdown]
# The right-hand panel is the point of the whole method: a single
# natural-gradient step has taken a deliberately absurd $q$ onto the sparse
# variational optimum, which for $M=20$ inducing points on this problem is
# not distinguishable by eye from the exact posterior. The two printed
# maxima confirm it quantitatively: restricted to the data range $[-3,3]$
# the sparse and exact means agree about fifteen times more closely than
# they do on the full test grid, where the largest gap sits at the grid's
# edge, past the last inducing input. Both are a fraction of a percent of
# the panel height, and both are a property of the sparse approximation, not
# of the optimiser.
#
# Now the comparison. We freeze every hyperparameter with
# `paramax.non_trainable` — applied to `hyper_partition`, the half of the
# pytree `partition_variational` carved off as *not* the natural gradient's
# business — so that both methods solve the *same* problem, namely finding
# the best $(\mathbf{m},\mathbf{L})$ for a fixed kernel, and run Adam on the
# variational parameters from the same bad initialisation.

# %%
frozen_family = eqx.combine(
    variational_partition, paramax.non_trainable(hyper_partition)
)

adam_iterations = 2000
_, adam_history = gpx.fit(
    model=frozen_family,
    objective=negative_elbo,
    train_data=regression_data,
    optim=ox.adam(1e-2),
    num_iters=adam_iterations,
    key=jr.key(0),
    verbose=False,
)

adam_gap = jnp.asarray(adam_history) + reference_elbo
natgrad_gap = reference_elbo - stepped_elbo
iteration_index = jnp.arange(adam_gap.size)
for tolerance in [10.0, 1.0, 0.1]:
    first_hit = jnp.min(jnp.where(adam_gap < tolerance, iteration_index, adam_gap.size))
    reached = "never" if int(first_hit) == adam_gap.size else f"{int(first_hit)}"
    print(
        f"Adam iterations to come within {tolerance:5.1f} nats of the optimum: "
        f"{reached}"
    )
print(
    f"Adam ELBO gap after {adam_iterations} iterations : {float(adam_gap[-1]):.3e} nats"
)
print(f"Natural-gradient ELBO gap after 1 iteration: {natgrad_gap:.3e} nats")

# %%
fig, axes = plt.subplots(ncols=2, figsize=(10, 3.0))

axes[0].plot(
    iteration_index + 1, -adam_history, color=cols[0], label="Adam on $(m, L)$"
)
axes[0].axhline(reference_elbo, color="black", linestyle="--", label="Exact optimum")
axes[0].scatter(
    [1], [stepped_elbo], color=cols[1], zorder=5, s=45, label="Natural gradient, 1 step"
)
axes[0].set(
    xscale="log",
    xlabel="Iteration",
    ylabel="ELBO",
    ylim=(reference_elbo - 250, reference_elbo + 25),
)
clean_legend(axes[0])

axes[1].plot(
    iteration_index + 1,
    jnp.maximum(adam_gap, 1e-16),
    color=cols[0],
    label="Adam on $(m, L)$",
)
axes[1].scatter(
    [1],
    [max(natgrad_gap, 1e-16)],
    color=cols[1],
    zorder=5,
    s=45,
    label="Natural gradient, 1 step",
)
axes[1].set(
    xscale="log", yscale="log", xlabel="Iteration", ylabel="ELBO gap to optimum (nats)"
)
clean_legend(axes[1])

# %% [markdown]
# Read the right-hand panel rather than the left. On log-log axes Adam's gap
# barely bends over the first few tens of iterations and then falls faster
# and faster, its slope steepest of all over the final few hundred — the
# opposite of the usual "fast start, long crawl" picture. That shape is the
# optimiser's, not the problem's: Adam normalises its step, so each
# coordinate moves by at most the learning rate however large the gradient
# is, and from an initialisation this bad it is the *distance* to be
# travelled that binds, not the gradient. The printed numbers say the same
# thing: it takes 867 iterations merely to come within ten nats of the
# optimum, 1800 to come within one nat, and it never gets within a tenth of
# a nat across the full 2000 — the ELBO gap is still $6.0\times10^{-1}$
# nats and still shrinking, while the single natural-gradient step closed
# the gap to zero at double precision. Adam is converging; it is simply
# converging in coordinates that put the optimum a long way away. The
# natural gradient never travels that distance, because the Fisher metric
# rescales it.
#
# Two caveats before this is oversold. The hyperparameters were frozen, so
# this is the problem natural gradients are best at: a pure variational
# optimisation. And the advantage rests on conjugacy, which is what makes
# $\gamma=1$ a solve rather than a step. Neither holds in the next demo.

# %% [markdown]
# ## Demo (ii): non-conjugate banana classification
#
# Outside conjugacy, $\mathbb{E}_q[\log p(\mathbf{y}\mid\mathbf{u})]$ is no
# longer affine in $\boldsymbol{\eta}$, so $\gamma=1$ is no longer a solve —
# it is a large step along a direction that was only computed locally.
# Salimbeni et al. find experimentally that "the initial natural gradient
# step size is a small value that is parameterization and likelihood
# dependent, but then increases to $\gamma = 1$", and in the stochastic
# setting they adopt a two-phase schedule: a log-linear ramp
#
# $$\gamma_t = \gamma_{\text{init}}\left(\frac{\gamma_{\text{final}}}{\gamma_{\text{init}}}\right)^{t/K} \quad (t < K), \qquad \gamma_t = \gamma_{\text{final}} \quad (t \ge K).$$
#
# Their reported settings are $\gamma_{\text{init}}=10^{-4}$,
# $\gamma_{\text{final}}=10^{-1}$ with $K$ between 5 and 40 for UCI-scale
# problems at batch size 256, and $\gamma_{\text{init}}=10^{-6}$,
# $\gamma_{\text{final}}=2\times10^{-2}$, $K=2000$ for MNIST at batch size
# 1024, always with $\gamma^{\text{Adam}} = 10^{-2}$ on the hyperparameters.
# Their conclusion is that "the success of the method relies on $\gamma$
# increasing to a reasonably large value ($\approx 0.1$) sufficiently
# quickly ($<1000$ iterations)".
#
# `natgrad_lr` accepts any Optax schedule — that is the API surface for this
# whole recommendation. We use $K = 100$ below. Their $K$ is
# dataset-dependent — 5 for the smaller UCI sets, 40 for NAVAL, 2000 for
# MNIST — and $100$ buys a little extra cone headroom (see the last
# section) at this $M$ from the default $\mathbf{m}=\mathbf{0}$,
# $\mathbf{S}=\mathbf{I}$ start, while still satisfying their own
# $<1000$-iteration criterion.
#
# Why does $\gamma < 1$ help when mini-batching? Recall the mirror-descent
# reading from the natural gradients notebook: the step is always a convex
# combination
# $\boldsymbol{\theta}_{\text{new}} = (1-\gamma)\,\boldsymbol{\theta} + \gamma\,\boldsymbol{\theta}^{\text{tgt}}$
# — the "when natural gradients fail" section below writes its second block
# out explicitly. The $N/B$ rescaling inside the ELBO keeps the stochastic
# gradient unbiased at every $\gamma$, including $\gamma=1$; what degrades
# is *variance*. Outside conjugacy $\boldsymbol{\theta}^{\text{tgt}}$ is not
# a fixed optimum — it is where one fixed-point iteration from *here* would
# land, and it moves with both $q$ and the mini-batch. At $\gamma=1$ the
# step discards $\boldsymbol{\theta}_t$ entirely and jumps onto that noisy,
# moving target, so nothing averages the mini-batch noise out of it. Taking
# $\gamma<1$ makes the update an exponential moving average in
# $\boldsymbol{\theta}$ towards the target, which is where the variance
# reduction comes from.
#
# Time for a harder problem.


# %%
def make_banana(key, num_points):
    """Two-class banana problem with a curved Bayes-optimal boundary."""
    key_latent, key_label = jr.split(key)
    latent = jr.uniform(key_latent, (num_points, 2), minval=-3.0, maxval=3.0)
    decision = latent[:, 1] - (0.7 * latent[:, 0] ** 2 - 1.5)
    probability = jax.nn.sigmoid(3.0 * decision)
    labels = (jr.uniform(key_label, (num_points,)) < probability).astype(jnp.float64)
    return latent, labels[:, None]


banana_key = jr.key(42)
banana_inputs, banana_labels = make_banana(banana_key, 2000)
banana_data = gpx.Dataset(X=banana_inputs, y=banana_labels)

num_train = 1600
train_inputs, test_inputs_2d = banana_inputs[:num_train], banana_inputs[num_train:]
train_labels, test_labels = banana_labels[:num_train], banana_labels[num_train:]
banana_train = gpx.Dataset(X=train_inputs, y=train_labels)

print(f"train / test  : {banana_train.n} / {banana_data.n - banana_train.n}")
print(f"class balance : {float(banana_data.y.mean()):.3f}")

# %%
boundary_inputs = jnp.linspace(-3.0, 3.0, 200)
boundary_outputs = 0.7 * boundary_inputs**2 - 1.5

fig, ax = plt.subplots(figsize=(5.5, 3.4))
for label, colour, name in [(0.0, cols[0], "$y = 0$"), (1.0, cols[1], "$y = 1$")]:
    mask = banana_labels.ravel() == label
    ax.scatter(
        banana_inputs[mask, 0],
        banana_inputs[mask, 1],
        s=6,
        alpha=0.4,
        color=colour,
        label=name,
    )
ax.plot(
    boundary_inputs,
    boundary_outputs,
    color="black",
    linestyle="--",
    label="Bayes-optimal boundary",
)
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", ylim=(-3.1, 3.1), title="The banana problem")
clean_legend(ax)

# %%
# Two identical models, built from the same arrays, so the comparison is
# fair.
num_banana_inducing = 50
inducing_grid = jnp.meshgrid(jnp.linspace(-2.8, 2.8, 10), jnp.linspace(-2.8, 2.8, 5))
banana_inducing = jnp.stack([axis.ravel() for axis in inducing_grid], axis=1)

banana_model = (
    gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(), kernel=jk.RBF(active_dims=[0, 1])
    )
    * gpx.likelihoods.Bernoulli()
)


def make_banana_family():
    """A fresh SVGP over the banana data, at the default m = 0, S = I."""
    return gpx.variational_families.VariationalGaussian(
        model=banana_model, inducing_inputs=banana_inducing
    )


natgrad_family = make_banana_family()
adam_family = make_banana_family()

print(f"inducing inputs: {banana_inducing.shape}")

# %%
# The log-linear ramp, 1e-4 -> 1e-1 over K = 100 iterations, as an Optax
# schedule handed straight to `natgrad_lr`.
num_iterations = 1000
batch_size = 256
natgrad_schedule = ox.exponential_decay(
    init_value=1e-4, transition_steps=100, decay_rate=1000.0, end_value=1e-1
)
print(
    "gamma at iterations 0, 50, 100, 999: "
    + ", ".join(f"{float(natgrad_schedule(t)):.2e}" for t in [0, 50, 100, 999])
)


def timed_fit(run):
    """Run twice: the first call pays JIT compilation, the second is steady state."""
    model, history = run()
    history.block_until_ready()
    start = time.perf_counter()
    model, history = run()
    history.block_until_ready()
    return model, history, time.perf_counter() - start


# %%
natgrad_model, natgrad_history, natgrad_seconds = timed_fit(
    lambda: gpx.fit_natgrads(
        model=natgrad_family,
        objective=negative_elbo,
        train_data=banana_train,
        optim=ox.adam(1e-2),
        natgrad_lr=natgrad_schedule,
        batch_size=batch_size,
        num_iters=num_iterations,
        key=jr.key(1),
        verbose=False,
    )
)
print(
    f"natural gradients + Adam : {natgrad_seconds:.2f} s "
    f"({1e3 * natgrad_seconds / num_iterations:.2f} ms / iteration)"
)

# %%
adam_model, adam_banana_history, adam_seconds = timed_fit(
    lambda: gpx.fit(
        model=adam_family,
        objective=negative_elbo,
        train_data=banana_train,
        optim=ox.adam(1e-2),
        batch_size=batch_size,
        num_iters=num_iterations,
        key=jr.key(1),
        verbose=False,
    )
)
print(
    f"Adam only                : {adam_seconds:.2f} s "
    f"({1e3 * adam_seconds / num_iterations:.2f} ms / iteration)"
)

# %% [markdown]
# Both runs use `ox.adam(1e-2)` on the kernel hyperparameters and the
# inducing inputs, so the only difference is how $(\mathbf{m},\mathbf{L})$
# move — `gpx.fit_natgrads` alternates a `natural_gradient_step` on those
# with an ordinary `gpx.fit`-style Adam step on everything else; `gpx.fit`
# moves everything with Adam. Timings are steady state: each fit is called
# twice and only the second call is timed, so JIT compilation is excluded
# from both. They were measured on CPU while executing this notebook, and
# will differ on your machine.

# %%
smoothing_window = 25


def smooth(history):
    """Trailing mean over `smoothing_window` iterations."""
    return jnp.convolve(
        history, jnp.ones(smoothing_window) / smoothing_window, mode="valid"
    )


smoothed_iterations = jnp.arange(smoothing_window - 1, num_iterations)
smoothed_natgrad = smooth(natgrad_history)
smoothed_adam = smooth(adam_banana_history)

# Derive the axis limits from the curves, so nothing is silently clipped on a
# machine whose run lands somewhere else.
elbo_floor = 0.95 * float(jnp.minimum(smoothed_natgrad.min(), smoothed_adam.min()))
elbo_ceiling = 1.10 * float(jnp.maximum(smoothed_natgrad.max(), smoothed_adam.max()))

fig, axes = plt.subplots(ncols=2, figsize=(10, 3.0), sharey=True)
for ax, horizontal, xlabel in [
    (axes[0], smoothed_iterations, "Iteration"),
    (
        axes[1],
        jnp.linspace(0.0, natgrad_seconds, num_iterations)[smoothing_window - 1 :],
        "Wall-clock seconds",
    ),
]:
    ax.plot(
        horizontal, smoothed_natgrad, color=cols[1], label="Natural gradients + Adam"
    )
    ax.set(xlabel=xlabel, yscale="log", ylim=(elbo_floor, elbo_ceiling))
axes[0].plot(smoothed_iterations, smoothed_adam, color=cols[0], label="Adam only")
axes[1].plot(
    jnp.linspace(0.0, adam_seconds, num_iterations)[smoothing_window - 1 :],
    smoothed_adam,
    color=cols[0],
    label="Adam only",
)
axes[0].set_ylabel("Negative ELBO (mini-batch)")
clean_legend(axes[0])
clean_legend(axes[1])

target_value = float(smoothed_adam[-1])
# Sentinel above every attainable iteration index, so "never crossed" is
# distinguishable from "crossed on the last iteration".
never = num_iterations + 1
crossing = int(
    jnp.min(jnp.where(smoothed_natgrad < target_value, smoothed_iterations, never))
)
print(
    f"Adam only, negative ELBO after {num_iterations} iterations   : "
    f"{target_value:8.2f}"
)
if crossing == never:
    print("Natural gradients, same value reached at iteration : never")
else:
    print(f"Natural gradients, same value reached at iteration : {crossing}")
    print(
        f"  i.e. {crossing * natgrad_seconds / num_iterations:.2f} s "
        f"versus {adam_seconds:.2f} s"
    )
print(
    "Natural gradients, negative ELBO after "
    f"{num_iterations} iterations: {float(smoothed_natgrad[-1]):8.2f}"
)

# %% [markdown]
# Both curves are mini-batch estimates and therefore noisy; they are shown
# as a 25-iteration trailing mean. Per iteration the natural-gradient run is
# far ahead: it reaches Adam's thousand-iteration bound of $335.31$ by
# iteration $117$, and finishes at $323.98$ against Adam's $335.31$. Per
# second it is still ahead, but by less, because each of its iterations does
# strictly more work — a natural-gradient step converts $(\mathbf{m},\mathbf{L})$
# to $\boldsymbol{\eta}$, differentiates the loss through the inverse map,
# converts back through $\boldsymbol{\theta}$, and *then* takes the Adam
# step on the hyperparameters. On the CPU that rendered this page that came
# to roughly $1.5$–$1.7\times$ the per-iteration cost of Adam alone across
# repeated runs — see the timings printed above, which are what your
# machine actually measured. Salimbeni et al. report a comparable ratio of
# about $1.5\times$ on their own hardware, and their headline experiments
# are on datasets far larger than this one; treat the numbers here as a
# demonstration of the mechanism, not as a benchmark.

# %%
grid_side = 64
grid_axis = jnp.linspace(-3.1, 3.1, grid_side)
grid_x, grid_y = jnp.meshgrid(grid_axis, grid_axis)
grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def predictive_probability(model, inputs, num_chunks=8):
    """Bernoulli success probability, evaluated in chunks to bound memory."""
    unwrapped = paramax.unwrap(model)
    likelihood = unwrapped.model.likelihood
    return jnp.concatenate(
        [likelihood(unwrapped(chunk)).mean for chunk in jnp.split(inputs, num_chunks)]
    )


fig, axes = plt.subplots(ncols=2, figsize=(10, 3.6), sharey=True)
for ax, model, name, seconds in [
    (axes[0], natgrad_model, "Natural gradients + Adam", natgrad_seconds),
    (axes[1], adam_model, "Adam only", adam_seconds),
]:
    probability = predictive_probability(model, grid_points).reshape(
        grid_side, grid_side
    )
    contours = ax.contourf(
        grid_x,
        grid_y,
        probability,
        levels=jnp.linspace(0.0, 1.0, 11),
        cmap="RdBu_r",
        alpha=0.7,
    )
    ax.contour(
        grid_x, grid_y, probability, levels=[0.5], colors="black", linewidths=1.5
    )
    ax.plot(
        boundary_inputs, boundary_outputs, color="black", linestyle="--", linewidth=1
    )
    # Held-out points, encoded by class in the notebook's categorical colours
    # rather than in the contour colourmap, so they stay legible on top of
    # the fill.
    for label, colour, marker in [(0.0, cols[0], "o"), (1.0, cols[1], "^")]:
        mask = test_labels.ravel() == label
        ax.scatter(
            test_inputs_2d[mask, 0],
            test_inputs_2d[mask, 1],
            marker=marker,
            s=12,
            alpha=0.9,
            color=colour,
            edgecolors="white",
            linewidths=0.3,
        )
    inducing = paramax.unwrap(model).inducing_inputs
    ax.scatter(inducing[:, 0], inducing[:, 1], marker="+", s=25, color="black")

    probability_test = predictive_probability(model, test_inputs_2d, num_chunks=1)
    accuracy = jnp.mean((probability_test > 0.5) == (test_labels.ravel() > 0.5))
    log_density = jnp.mean(
        test_labels.ravel() * jnp.log(probability_test)
        + (1.0 - test_labels.ravel()) * jnp.log1p(-probability_test)
    )
    ax.set(
        xlabel=r"$x_1$",
        xlim=(-3.1, 3.1),
        ylim=(-3.1, 3.1),
        title=f"{name}\naccuracy {accuracy:.3f}, NLPD {-log_density:.3f}",
    )
    print(
        f"{name:26s} test accuracy {accuracy:.4f}, test NLPD {-log_density:.4f}, "
        f"{seconds:.2f} s"
    )
axes[0].set_ylabel(r"$x_2$")
colourbar = fig.colorbar(contours, ax=axes, label=r"$q(y=1 \mid x)$")

# %% [markdown]
# The solid black line is each model's $0.5$ contour and the dashed line is
# the Bayes-optimal boundary $x_2 = 0.7x_1^2 - 1.5$; crosses mark the
# inducing inputs after training.
#
# The two panels are very nearly the same picture, and the two sets of
# printed test metrics are very nearly the same numbers: $93.50\%$ accuracy
# and $0.166$ NLPD for natural gradients against $93.25\%$ and $0.169$ for
# Adam alone. That is the honest reading of this experiment, and it is
# worth stating plainly: on a densely-sampled, easily-separated problem the
# natural gradient buys *optimiser speed*, not final predictive quality — it
# reached Adam's thousand-iteration bound at the crossing iteration printed
# above, and both models then classify the held-out points about equally
# well. Both runs also train the kernel and the inducing inputs with Adam
# and finish at different hyperparameters, so whatever small difference
# remains between these contours cannot be attributed to $\mathbf{S}$ alone.
# `make_banana` draws inputs uniformly on $[-3,3]^2$ and the plotted grid is
# $[-3.1,3.1]^2$, so there is no region here that is far from the data; a
# demonstration that natural gradients give better-calibrated
# *extrapolative* uncertainty would need a problem built for it.

# %% [markdown]
# ## When natural gradients fail
#
# Recall the update is
# $\boldsymbol{\theta}\leftarrow\boldsymbol{\theta} - \gamma\,\partial\ell/\partial\boldsymbol{\eta}$,
# and that $\boldsymbol{\Theta}_2$ must stay negative definite, because
# $\boldsymbol{\Theta}_2 = -\tfrac12\mathbf{S}^{-1}$ and $\mathbf{S}$ is a
# covariance. Nothing in the update enforces that automatically. The
# natural gradients notebook's **cone-safety theorem** proves — via Price's
# theorem applied to the ELBO's data-fit term — that for any log-concave
# likelihood and any starting point, $\gamma\in[0,1]$ keeps
# $\boldsymbol{\Theta}_2^{\text{new}}$ inside that cone, mini-batching
# included; the full statement and proof are there, and are not repeated
# here. What escapes the guarantee is $\gamma>1$, which extrapolates past
# the target, and likelihoods that are not log-concave *as computed* rather
# than as written: GPJax's `inv_probit` clips its output into
# $[10^{-3},\,1-10^{-3}]$, which flattens the tail of $\log p$ enough to
# give it a positive second derivative for $f \lesssim -2.44$, so even the
# Bernoulli model used below leaves the guaranteed regime once a point is
# confidently mislabelled. Below we sweep $\gamma$ from an over-confident
# starting point — $\mathbf{S}_0 = 10^{-2}\mathbf{I}$, sharper than the
# target — which is precisely the regime where extrapolation bites.

# %%
overconfident_family = gpx.variational_families.VariationalGaussian(
    model=banana_model,
    inducing_inputs=banana_inducing,
    variational_mean=jnp.zeros((num_banana_inducing, 1)),
    variational_root_covariance=0.1 * jnp.eye(num_banana_inducing),
)
overconfident_mean = overconfident_family.variational_mean.unwrap()
overconfident_root = overconfident_family.variational_root_covariance.unwrap()


def banana_loss_of_expectation(expectation):
    variational_mean, variational_root = moments_from_expectation(*expectation)
    trial = eqx.tree_at(
        lambda family: (family.variational_mean, family.variational_root_covariance),
        overconfident_family,
        (Real(variational_mean), LowerTriangular(variational_root)),
    )
    return negative_elbo(paramax.unwrap(trial), banana_train)


cone_gradient = jax.grad(banana_loss_of_expectation)(
    expectation_from_moments(overconfident_mean, overconfident_root)
)
# The matrix statistic is symmetric, so symmetrise the entrywise autodiff
# gradient.
matrix_gradient = 0.5 * (cone_gradient[1] + cone_gradient[1].T)
_, natural_matrix = natural_from_moments(overconfident_mean, overconfident_root)

print("gamma      max eig(Theta2_new)   status")
for gamma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    largest = jnp.max(jnp.linalg.eigvalsh(natural_matrix - gamma * matrix_gradient))
    status = "negative definite" if largest < 0 else "*** LEFT THE CONE ***"
    print(f"{gamma:6.2f}   {largest:+18.5f}   {status}")

# %% [markdown]
# Read that table as a statement about *this initialisation*, not about
# $\gamma=2$ in general. Here $\mathbf{S}_0 = 10^{-2}\mathbf{I}$ makes
# $\boldsymbol{\Theta}_2 = -50\,\mathbf{I}$, an order of magnitude sharper
# than the target, so the convex combination has very little room to
# extrapolate into: the sign flips between $\gamma=1$ ($-4.67$) and
# $\gamma=2$ ($+40.66$), and interpolating those two rows puts the crossing
# at $\gamma\approx1.10$. Where it lands is entirely a function of how far
# $\boldsymbol{\Theta}_2$ starts from $\boldsymbol{\Theta}_2^{\text{tgt}}$:
# in the limit where the two coincide, every $\gamma$ is safe. What the
# theorem actually guarantees is $\gamma\in[0,1]$, for any log-concave
# likelihood and any starting point, and it says nothing whatsoever beyond
# that — which is the line worth remembering.
#
# When it does go wrong, `jnp.linalg.cholesky` returns `NaN` rather than
# raising, which means validity is a *value* and the fix stays
# `jit`-compatible. `natural_gradient_step` exploits that with a backoff: it
# evaluates the trial steps $\{\gamma\beta^k\}_{k=0}^{K}$ under `vmap` and
# selects the first one whose Cholesky is finite. `backoff` ($\beta$,
# default $0.5$) and `max_backoff` ($K$, default $5$) are exposed by
# `fit_natgrads` and by `natural_gradient_step` directly.

# %%
print("gamma = 100 from the over-confident initialisation")
overconfident_variational, overconfident_hyper = partition_variational(
    overconfident_family
)
for max_backoff in [0, 3, 5, 7, 10]:
    stepped, _ = natural_gradient_step(
        overconfident_variational,
        overconfident_hyper,
        banana_train,
        negative_elbo,
        100.0,
        max_backoff=max_backoff,
    )
    smallest_trial = 100.0 * 0.5**max_backoff
    root = eqx.combine(
        stepped, overconfident_hyper
    ).variational_root_covariance.unwrap()
    outcome = "finite" if bool(jnp.all(jnp.isfinite(root))) else "NaN"
    print(
        f"  max_backoff = {max_backoff:2d}  smallest trial gamma = "
        f"{smallest_trial:7.3f}   result: {outcome}"
    )

# %% [markdown]
# The backoff is a safety net with a finite budget, not a licence to pick
# $\gamma$ carelessly: from this starting point it needs to shrink
# $\gamma=100$ by a factor of $2^7$ before the Cholesky succeeds, so the
# default `max_backoff=5` still returns `NaN`. That is the intended
# behaviour — a silent 32-fold reduction of a step size the user chose
# badly would be worse than a visible failure.

# %% [markdown]
# ## Practical guidance
#
# * **Conjugate and full batch: use $\gamma = 1$.** One iteration is the
#   exact solution, and further iterations are fixed points.
# * **Non-conjugate or mini-batched: ramp $\gamma$.** Salimbeni et al.
#   recommend starting around $10^{-4}$ and reaching $\approx 10^{-1}$
#   "sufficiently quickly ($<1000$ iterations)"; `natgrad_lr` accepts any
#   Optax schedule, and defaults to $10^{-1}$.
# * **Never exceed $\gamma = 1$.** The cone-safety theorem's guarantee stops
#   there, and the backoff exists to catch mistakes, not to enable them.
# * **If a mini-batched run produces `NaN`, raise the batch size before
#   lowering $\gamma$.** Small batches make
#   $\boldsymbol{\Theta}_2^{\text{tgt}}$ badly conditioned, which no step
#   size fully repairs.
# * **Prefer the whitened family.** The natural-gradient direction is
#   parameterisation-invariant, so whitening does not change the sequence
#   of distributions in exact arithmetic; it changes the *conditioning* of
#   every map, and keeps $\mathbf{m}_w$, $\mathbf{S}_w$ at $\mathcal{O}(1)$.
# * **Leave `map_jitter` at $0$.** It biases $\mathbf{S}$ by
#   $\approx\varepsilon\lVert\mathbf{S}\rVert^2$ independently of
#   conditioning. Raise it to $10^{-12}$–$10^{-10}$ only when fighting an
#   ill-conditioned $\mathbf{S}$.
# * **Non-log-concave likelihoods have no guarantee at all.** For a
#   Student-$t$ likelihood with gross outliers the target
#   $\boldsymbol{\Theta}_2^{\text{tgt}}$ can itself be outside the cone, so
#   no positive $\gamma$ is provably safe.
#
# The companion [dual sparse GP notebook](dual_svgp.py) is the applied
# notebook for the other storage convention this geometry admits — the
# site, or dual, parameterisation of {cite:t}`adam2021dual` — and, like this
# one, it assumes the [natural gradients notebook](natural_gradients.py)
# throughout rather than re-deriving anything.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
