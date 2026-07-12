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
# # Multi-Output Gaussian Processes
#
# Standard Gaussian process models map a $D$-dimensional input to a single scalar
# output. In many settings, however, we wish to model several correlated output
# quantities simultaneously. A multi-output Gaussian process captures these
# correlations so that observations of one output can inform predictions of another.
#
# This notebook introduces the Intrinsic Coregionalization Model (ICM) implemented
# in GPJax. We fit an ICM to a real record of North Atlantic ocean waves — the total
# sea state alongside its swell and wind-sea components — by optimising the marginal
# log-likelihood, then inspect the learned coregionalization matrix to see what the
# model has discovered about the physical structure of the sea.

# %%
from pathlib import Path

from examples.utils import use_mpl_style
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

key = jr.key(42)
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## The Intrinsic Coregionalization Model
#
# The ICM assumes that all outputs share a single latent Gaussian process, weighted
# differently for each output through a positive semi-definite coregionalization
# matrix $\mathbf{B} \in \mathbb{R}^{P \times P}$. Given an input-space kernel
# $k(\mathbf{x}, \mathbf{x}')$, the multi-output covariance between output $p$ at
# input $\mathbf{x}$ and output $q$ at input $\mathbf{x}'$ is
# $$
# \operatorname{cov}\bigl(f_p(\mathbf{x}),\, f_q(\mathbf{x}')\bigr) = B_{pq}\, k(\mathbf{x}, \mathbf{x}').
# $$
# Stacking all $N$ observations across $P$ outputs into a single vector of length
# $NP$, the joint covariance matrix takes the Kronecker form
# $$
# \mathbf{K} = \mathbf{B} \otimes \mathbf{K}_{\mathbf{x}\mathbf{x}},
# $$
# where $\mathbf{K}_{\mathbf{x}\mathbf{x}}$ is the $N \times N$ Gram matrix of the
# base kernel.
#
# The coregionalization matrix is parameterised as
# $$
# \mathbf{B} = \mathbf{W}\mathbf{W}^\top + \operatorname{diag}(\boldsymbol{\kappa}),
# $$
# where $\mathbf{W} \in \mathbb{R}^{P \times R}$ is a low-rank factor of rank $R$
# and $\boldsymbol{\kappa} \in \mathbb{R}^P_{>0}$ is a positive diagonal. The rank
# parameter controls how many latent sources of correlation the model can express.
# With $R = 1$ the model captures a single shared direction of variation across the
# outputs; choosing $R = 2$ lets $\mathbf{B}$ represent two independent latent
# drivers, which is exactly what we need for the wave data below.

# %% [markdown]
# ## Atlantic wave data
#
# Instead of a synthetic example, we model a real record of ocean waves: hourly
# significant wave heights for December 2023 at a point in the North Atlantic west of
# Ireland (53.5°N, 11°W), during the winter-storm season. Three quantities are
# recorded, each in metres:
#
# - **total wave height** — the significant wave height of the full sea surface,
# - **swell** — the component due to distant storms, arriving as long-period waves,
# - **wind-sea** — the component raised by the local wind blowing overhead.
#
# The total sea state combines two largely independent physical drivers. Swell is set
# by weather systems hundreds of kilometres away, while the wind-sea responds to the
# wind directly overhead. Consequently the total wave height correlates strongly with
# each component, yet the two components are only weakly correlated with each other. A
# multi-output GP is a natural fit: a shared temporal kernel captures the common
# evolution over the month, while the coregionalization matrix should recover this
# two-driver structure.
#
# Data: ERA5 wave reanalysis via Open-Meteo (CC-BY).

# %%
csv_candidates = [
    Path("examples/data/atlantic_wave_components.csv"),
    Path("data/atlantic_wave_components.csv"),
]
csv_path = next(path for path in csv_candidates if path.exists())
wave_frame = pd.read_csv(csv_path, parse_dates=["time"])

output_columns = ["wave_height", "swell_wave_height", "wind_wave_height"]
output_labels = ["Total wave height", "Swell", "Wind-sea"]
num_outputs = len(output_columns)

# Thin from hourly to every third hour to keep the exact multi-output GP fast.
subsample_stride = 3
wave_frame = wave_frame.iloc[::subsample_stride].reset_index(drop=True)

# Shared input: hours elapsed since the start of the record.
time_hours = (
    wave_frame["time"] - wave_frame["time"].iloc[0]
).dt.total_seconds().to_numpy() / 3600.0
wave_components_m = wave_frame[output_columns].to_numpy()  # [N, P], metres
num_datapoints = time_hours.shape[0]

# Standardise the shared input for stable optimisation.
input_mean = time_hours.mean()
input_std = time_hours.std()
x = ((time_hours - input_mean) / input_std).reshape(-1, 1)

# Standardise each output; keep the moments to map predictions back to metres.
output_means = wave_components_m.mean(axis=0)
output_stds = wave_components_m.std(axis=0)
y = (wave_components_m - output_means) / output_stds

D = gpx.Dataset(X=jnp.asarray(x), y=jnp.asarray(y))

print(f"Modelling {num_outputs} outputs at {num_datapoints} time points.")
empirical_corr = jnp.corrcoef(jnp.asarray(wave_components_m).T)
print("Empirical output correlations (metres):")
print(jnp.round(empirical_corr, 3))

# %% [markdown]
# We plot the three observed wave-height series across the month.

# %%
fig, axes = plt.subplots(1, num_outputs, figsize=(12, 2.5), sharex=True)

for output_index, ax in enumerate(axes):
    ax.plot(
        time_hours,
        wave_components_m[:, output_index],
        "o",
        color=cols[output_index],
        alpha=0.6,
        ms=3,
        label="Observations",
    )
    ax.set_title(output_labels[output_index])
    ax.set_xlabel("time (hours)")
    ax.legend(loc="best", fontsize=7)

axes[0].set_ylabel("wave height (m)")

# %% [markdown]
# ## Model definition
#
# We construct the ICM model in three steps. First, we build a
# `CoregionalizationMatrix` with $P = 3$ outputs and rank $R = 2$: two latent
# directions give the model enough freedom to represent the separate swell and
# wind-sea drivers rather than forcing all three outputs onto a single shared factor.
# Second, we wrap an RBF base kernel together with the coregionalization matrix inside
# an `ICMKernel`. Third, we pair a zero-mean GP prior with a `MultiOutputGaussian`
# likelihood, which allows a separate noise variance for each output.

# %%
key, subkey = jr.split(key)

coreg = gpx.parameters.CoregionalizationMatrix(
    num_outputs=num_outputs, rank=2, key=subkey
)
kernel = gpx.kernels.ICMKernel(
    base_kernel=gpx.kernels.RBF(),
    coregionalization_matrix=coreg,
)

meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.MultiOutputGaussian(
    num_datapoints=num_datapoints, num_outputs=num_outputs, obs_stddev=1.0
)
posterior = prior * likelihood

# %% [markdown]
# Before optimisation we verify that the marginal log-likelihood is finite.

# %%
print(f"Initial negative MLL: {-gpx.objectives.conjugate_mll(posterior, D):.3f}")

# %% [markdown]
# ## Optimisation
#
# We optimise the kernel hyperparameters, the coregionalization matrix entries, and
# the per-output noise standard deviations by maximising the conjugate marginal
# log-likelihood using L-BFGS via `fit_scipy`.

# %%
opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
)

print(f"Optimised negative MLL: {-gpx.objectives.conjugate_mll(opt_posterior, D):.3f}")

# %% [markdown]
# ## Prediction
#
# The multi-output posterior returns predictions as a single Gaussian distribution
# over a flattened vector of length $MP$, where $M$ is the number of test points and
# $P$ is the number of outputs. The ordering is output-major: all $M$ values for
# output 1 appear first, followed by all $M$ values for output 2, and so on. We
# reshape the mean, extract per-output marginal variances from the diagonal of the
# joint covariance, then undo the output standardisation to return to metres.

# %%
num_test = 400
xtest_hours = jnp.linspace(time_hours.min(), time_hours.max(), num_test)
xtest = ((xtest_hours - input_mean) / input_std).reshape(-1, 1)

pred = opt_posterior.predict(xtest, train_data=D)

pred_mean_std = pred.mean.reshape(num_outputs, num_test).T  # [M, P], standardised
pred_var_std = jnp.diag(pred.covariance()).reshape(num_outputs, num_test).T
pred_std_std = jnp.sqrt(pred_var_std)

# Map the standardised predictions back to metres for plotting.
output_means_j = jnp.asarray(output_means)
output_stds_j = jnp.asarray(output_stds)
pred_mean = pred_mean_std * output_stds_j + output_means_j
pred_std = pred_std_std * output_stds_j

# %% [markdown]
# We now plot the predictive distribution for each output over the observed series.
# The shaded region shows the model's two-sigma predictive uncertainty.

# %%
fig, axes = plt.subplots(1, num_outputs, figsize=(12, 2.5), sharex=True)

for output_index, ax in enumerate(axes):
    ax.plot(
        time_hours,
        wave_components_m[:, output_index],
        "o",
        color=cols[output_index],
        alpha=0.4,
        ms=3,
        label="Observations",
    )
    ax.plot(
        xtest_hours,
        pred_mean[:, output_index],
        color=cols[output_index],
        label="Predictive mean",
    )
    ax.fill_between(
        xtest_hours,
        pred_mean[:, output_index] - 2 * pred_std[:, output_index],
        pred_mean[:, output_index] + 2 * pred_std[:, output_index],
        color=cols[output_index],
        alpha=0.2,
        label="Two sigma",
    )
    ax.set_title(output_labels[output_index])
    ax.set_xlabel("time (hours)")
    ax.legend(loc="best", fontsize=7)

axes[0].set_ylabel("wave height (m)")

# %% [markdown]
# ## Learned coregionalization matrix
#
# The coregionalization matrix $\mathbf{B}$ encodes the learned correlations between
# outputs. Its off-diagonal entries indicate how strongly the outputs covary: a large
# positive entry between outputs $p$ and $q$ means they tend to increase together,
# while a value near zero suggests they are largely independent given the shared
# input kernel.
#
# Because the raw scale of $\mathbf{B}$ is confounded with the base-kernel variance,
# we normalise it into implied output correlations,
# $\rho_{pq} = B_{pq} / \sqrt{B_{pp} B_{qq}}$, which are directly comparable to the
# empirical correlations computed above. We print both and visualise the correlations
# as a heatmap.

# %%
B_learned = opt_posterior.prior.kernel.coregionalization_matrix.B

b_diag = jnp.sqrt(jnp.diag(B_learned))
implied_corr = B_learned / jnp.outer(b_diag, b_diag)

print("Learned coregionalization matrix B:")
print(jnp.round(B_learned, 3))
print("Implied output correlations from B:")
print(jnp.round(implied_corr, 3))

short_labels = ["Total", "Swell", "Wind-sea"]

fig, ax = plt.subplots(figsize=(4, 3.5))
im = ax.imshow(implied_corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

for row in range(num_outputs):
    for col in range(num_outputs):
        ax.text(
            col,
            row,
            f"{implied_corr[row, col]:.2f}",
            ha="center",
            va="center",
            fontsize=10,
        )

ax.set_xticks(range(num_outputs))
ax.set_yticks(range(num_outputs))
ax.set_xticklabels(short_labels, rotation=30, ha="right")
ax.set_yticklabels(short_labels)
ax.set_title("Implied output correlations")
fig.colorbar(im, ax=ax, shrink=0.8)

# %% [markdown]
# The learned structure recovers the physics of the sea state. The total wave height
# loads strongly onto **both** the swell and the wind-sea, since it is their
# combination. Yet swell and wind-sea are only weakly coupled to each other: they are
# driven by distinct processes — distant storms versus the local wind — so knowing
# one tells us little about the other. This is precisely the two-driver structure seen
# in the empirical correlations, and the rank-2 coregionalization matrix was able to
# represent it because two latent directions can load the total onto each component
# independently. A rank-1 model would have been forced to tie all three outputs to a
# single shared factor, over-stating the swell/wind-sea coupling.

# %% [markdown]
# ## From ICM to LCM
#
# The Intrinsic Coregionalization Model is powerful but limited: it assumes that
# **all** inter-output correlations are explained by a single shared latent Gaussian
# process. When the outputs are driven by multiple independent sources of variation
# — for example, a slow trend and a fast oscillation — a single latent kernel cannot
# capture both length-scales simultaneously. The ICM must compromise, and the
# resulting fit degrades.
#
# The **Linear Model of Coregionalization** (LCM) removes this limitation by
# combining $Q$ independent latent GPs, each equipped with its own input-space
# kernel and its own output-space coregionalization matrix. The additional
# components give the model the flexibility to assign distinct spectral
# characteristics to different sources of inter-output coupling.

# %% [markdown]
# ## The Linear Model of Coregionalization
#
# Given $Q$ latent kernels $\{k_q\}_{q=1}^{Q}$ and $Q$ coregionalization matrices
# $\{\mathbf{B}^{(q)}\}_{q=1}^{Q}$, each of size $P \times P$, the LCM defines the
# multi-output covariance between output $p$ at input $\mathbf{x}$ and output $r$ at
# input $\mathbf{x}'$ as
# $$
# \operatorname{cov}\bigl(f_p(\mathbf{x}),\, f_r(\mathbf{x}')\bigr)
# = \sum_{q=1}^{Q} B^{(q)}_{pr}\, k_q(\mathbf{x}, \mathbf{x}').
# $$
#
# Stacking all $N$ observations across $P$ outputs into a single vector of length
# $NP$, the joint covariance matrix is the **sum of Kronecker products**
# $$
# \mathbf{K} = \sum_{q=1}^{Q} \mathbf{B}^{(q)} \otimes
#              \mathbf{K}^{(q)}_{\mathbf{x}\mathbf{x}},
# $$
# where $\mathbf{K}^{(q)}_{\mathbf{x}\mathbf{x}}$ is the $N \times N$ Gram matrix
# of the $q$-th latent kernel.
#
# ### Relationship to ICM
#
# Setting $Q = 1$ recovers the ICM exactly: there is one kernel, one
# coregionalization matrix, and the covariance has pure Kronecker structure. GPJax
# exploits this: when an `LCMKernel` has a single component, the compute engine
# returns a `Kronecker` operator, preserving the efficient $\mathcal{O}(N^3 + P^3)$
# decomposition. For $Q > 1$ the sum of Kronecker products no longer admits a
# closed-form Kronecker inverse, so GPJax materialises the full $NP \times NP$ dense
# matrix and solves via a standard Cholesky factorisation in
# $\mathcal{O}((NP)^3)$.
#
# ### Per-component coregionalization
#
# Each coregionalization matrix is parameterised as before:
# $$
# \mathbf{B}^{(q)} = \mathbf{W}^{(q)}{\mathbf{W}^{(q)}}^\top
#   + \operatorname{diag}(\boldsymbol{\kappa}^{(q)}),
# $$
# where $\mathbf{W}^{(q)} \in \mathbb{R}^{P \times R_q}$ is a low-rank factor and
# $\boldsymbol{\kappa}^{(q)} \in \mathbb{R}^P_{>0}$ a positive diagonal. The rank
# $R_q$ of each component can be chosen independently. A component with rank 1
# captures one direction of inter-output correlation at the length-scale determined
# by $k_q$; increasing the rank allows richer coupling patterns at that scale.

# %% [markdown]
# ## A three-output synthetic dataset
#
# To demonstrate the advantage of multiple latent components, we construct a dataset
# with three outputs driven by **two** distinct latent functions:
#
# - $g_1(x) = \sin(2\pi x)$ — a smooth, low-frequency oscillation,
# - $g_2(x) = \cos(4\pi x)$ — a faster oscillation at double the frequency.
#
# The three observed outputs are mixtures of these latent functions:
# \begin{align}
# f_1(x) &= g_1(x), \\
# f_2(x) &= 0.5\,g_1(x) + 0.5\,g_2(x), \\
# f_3(x) &= g_2(x).
# \end{align}
# Outputs 1 and 3 are each dominated by a single latent source, while output 2 is a
# balanced mixture of both. A single-component ICM kernel would struggle here because
# it cannot separate the two frequency scales. An LCM with $Q = 2$ components —
# each learning a different length-scale — should recover the latent structure.

# %%
N_lcm = 50
P_lcm = 3
noise_stds_lcm = jnp.array([0.1, 0.15, 0.2])

key, subkey1, subkey2, subkey3 = jr.split(key, 4)
x_lcm = jnp.sort(jr.uniform(subkey1, shape=(N_lcm,), minval=0.0, maxval=1.0)).reshape(
    -1, 1
)

g1 = lambda x: jnp.sin(2 * jnp.pi * x)
g2 = lambda x: jnp.cos(4 * jnp.pi * x)

f1_lcm = lambda x: g1(x)
f2_lcm = lambda x: 0.5 * g1(x) + 0.5 * g2(x)
f3_lcm = lambda x: g2(x)

y1_lcm = f1_lcm(x_lcm) + jr.normal(subkey1, shape=x_lcm.shape) * noise_stds_lcm[0]
y2_lcm = f2_lcm(x_lcm) + jr.normal(subkey2, shape=x_lcm.shape) * noise_stds_lcm[1]
y3_lcm = f3_lcm(x_lcm) + jr.normal(subkey3, shape=x_lcm.shape) * noise_stds_lcm[2]
y_lcm = jnp.hstack([y1_lcm, y2_lcm, y3_lcm])  # [N, 3]

D_lcm = gpx.Dataset(X=x_lcm, y=y_lcm)

# %% [markdown]
# We plot the three outputs alongside their latent functions.

# %%
xtest_lcm = jnp.linspace(0.0, 1.0, 200).reshape(-1, 1)
output_labels_lcm = [r"$f_1$", r"$f_2$", r"$f_3$"]
latent_fns_lcm = [f1_lcm, f2_lcm, f3_lcm]

fig, axes = plt.subplots(1, 3, figsize=(12, 2.5), sharey=True)

for i, ax in enumerate(axes):
    ax.plot(
        x_lcm, y_lcm[:, i], "o", color=cols[i], alpha=0.6, label="Observations", ms=4
    )
    ax.plot(
        xtest_lcm,
        latent_fns_lcm[i](xtest_lcm),
        color=cols[i],
        ls="--",
        label="Latent function",
    )
    ax.set_title(output_labels_lcm[i])
    ax.set_xlabel(r"$x$")
    ax.legend(loc="best", fontsize=7)

axes[0].set_ylabel(r"$y$")

# %% [markdown]
# ## LCM model definition
#
# We build an LCM with $Q = 2$ components. The first component uses an RBF kernel,
# which is well-suited to capture the smooth, low-frequency latent function $g_1$.
# The second component uses a Matérn-3/2 kernel, whose shorter default length-scale
# can adapt to the faster oscillation in $g_2$. Each component has its own
# `CoregionalizationMatrix` with $P = 3$ outputs and rank $R = 1$, so each component
# captures one direction of inter-output correlation at its characteristic
# length-scale.

# %%
key, subkey1, subkey2 = jr.split(key, 3)

coreg1 = gpx.parameters.CoregionalizationMatrix(num_outputs=P_lcm, rank=1, key=subkey1)
coreg2 = gpx.parameters.CoregionalizationMatrix(num_outputs=P_lcm, rank=1, key=subkey2)

lcm_kernel = gpx.kernels.LCMKernel(
    kernels=[gpx.kernels.RBF(), gpx.kernels.Matern32()],
    coregionalization_matrices=[coreg1, coreg2],
)

meanf_lcm = gpx.mean_functions.Zero()
prior_lcm = gpx.gps.Prior(mean_function=meanf_lcm, kernel=lcm_kernel)
likelihood_lcm = gpx.likelihoods.MultiOutputGaussian(
    num_datapoints=N_lcm, num_outputs=P_lcm, obs_stddev=1.0
)
posterior_lcm = prior_lcm * likelihood_lcm

# %% [markdown]
# Before optimisation we verify that the marginal log-likelihood is finite.

# %%
print(
    f"Initial negative MLL: {-gpx.objectives.conjugate_mll(posterior_lcm, D_lcm):.3f}"
)

# %% [markdown]
# ## Optimisation
#
# As with the ICM example, we maximise the conjugate marginal log-likelihood using
# `fit_scipy`. The optimiser now has more parameters to tune: two sets of kernel
# hyperparameters and two coregionalization matrices.

# %%
opt_posterior_lcm, history_lcm = gpx.fit_scipy(
    model=posterior_lcm,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D_lcm,
)

print(
    "Optimised negative MLL: "
    f"{-gpx.objectives.conjugate_mll(opt_posterior_lcm, D_lcm):.3f}"
)

# %% [markdown]
# ## Prediction
#
# The multi-output posterior returns predictions as a single Gaussian distribution
# over a flattened vector of length $MP$, where $M$ is the number of test points.
# The ordering is output-major: all $M$ values for output 1 appear first, then
# output 2, then output 3. We reshape accordingly.

# %%
M_lcm = xtest_lcm.shape[0]
pred_lcm = opt_posterior_lcm.predict(xtest_lcm, train_data=D_lcm)

pred_mean_lcm = pred_lcm.mean.reshape(P_lcm, M_lcm).T
pred_var_lcm = jnp.diag(pred_lcm.covariance()).reshape(P_lcm, M_lcm).T
pred_std_lcm = jnp.sqrt(pred_var_lcm)

# %% [markdown]
# We now plot the predictive distribution for each of the three outputs. The shaded
# region shows the predictive uncertainty.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 2.5), sharey=True)

for i, ax in enumerate(axes):
    ax.plot(
        x_lcm, y_lcm[:, i], "o", color=cols[i], alpha=0.5, label="Observations", ms=4
    )
    ax.plot(
        xtest_lcm,
        latent_fns_lcm[i](xtest_lcm),
        ls="--",
        color="grey",
        label="Latent function",
    )
    ax.plot(xtest_lcm, pred_mean_lcm[:, i], color=cols[i], label="Predictive mean")
    ax.fill_between(
        xtest_lcm.squeeze(),
        pred_mean_lcm[:, i] - 2 * pred_std_lcm[:, i],
        pred_mean_lcm[:, i] + 2 * pred_std_lcm[:, i],
        color=cols[i],
        alpha=0.2,
        label="Two sigma",
    )
    ax.set_title(output_labels_lcm[i])
    ax.set_xlabel(r"$x$")
    ax.legend(loc="best", fontsize=7)

axes[0].set_ylabel(r"$y$")

# %% [markdown]
# ## Learned coregionalization structure
#
# Unlike the ICM, which produces a single $\mathbf{B}$ matrix, the LCM yields one
# coregionalization matrix per component. Each $\mathbf{B}^{(q)}$ tells us how the
# $q$-th latent process couples the outputs. By inspecting these matrices we can
# recover which outputs share each latent source.
#
# We expect the component paired with the RBF kernel (smooth, low-frequency) to show
# strong coupling between outputs 1 and 2, since both contain $g_1$. The Matérn-3/2
# component (higher frequency) should couple outputs 2 and 3, which share $g_2$.

# %%
kernel_names = ["RBF (component 1)", "Matérn-3/2 (component 2)"]
fig, axes_B = plt.subplots(1, 2, figsize=(8, 3))

for idx, (cm, _k) in enumerate(opt_posterior_lcm.prior.kernel.components):
    B_q = cm.B
    ax = axes_B[idx]
    im = ax.imshow(
        B_q,
        cmap="RdBu_r",
        vmin=-jnp.max(jnp.abs(B_q)),
        vmax=jnp.max(jnp.abs(B_q)),
    )
    for row in range(P_lcm):
        for col in range(P_lcm):
            ax.text(
                col,
                row,
                f"{B_q[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_xticks(range(P_lcm))
    ax.set_yticks(range(P_lcm))
    ax.set_xticklabels(output_labels_lcm)
    ax.set_yticklabels(output_labels_lcm)
    ax.set_title(rf"$\mathbf{{B}}^{{({idx + 1})}}$ — {kernel_names[idx]}")
    fig.colorbar(im, ax=ax, shrink=0.8)

# %% [markdown]
# The two learned coregionalization matrices reveal the latent structure of the data.
# Each component has specialised: one captures the low-frequency correlations driven
# by $g_1$, and the other captures the higher-frequency correlations driven by $g_2$.
# Output 2, which depends on both latent functions, appears with non-negligible
# weight in both matrices — exactly as expected.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
