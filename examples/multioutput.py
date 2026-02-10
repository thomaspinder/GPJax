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
# in GPJax. We construct a synthetic dataset with two correlated outputs, fit an ICM
# model by optimising the marginal log-likelihood, and inspect the learned
# coregionalization matrix to see what the model has discovered about the output
# structure.

# %%
# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt

from examples.utils import use_mpl_style

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
#
# $$\operatorname{cov}\bigl(f_p(\mathbf{x}),\, f_q(\mathbf{x}')\bigr) = B_{pq}\, k(\mathbf{x}, \mathbf{x}').$$
#
# Stacking all $N$ observations across $P$ outputs into a single vector of length
# $NP$, the joint covariance matrix takes the Kronecker form
#
# $$\mathbf{K} = \mathbf{B} \otimes \mathbf{K}_{\mathbf{x}\mathbf{x}},$$
#
# where $\mathbf{K}_{\mathbf{x}\mathbf{x}}$ is the $N \times N$ Gram matrix of the
# base kernel.
#
# The coregionalization matrix is parameterised as
#
# $$\mathbf{B} = \mathbf{W}\mathbf{W}^\top + \operatorname{diag}(\boldsymbol{\kappa}),$$
#
# where $\mathbf{W} \in \mathbb{R}^{P \times R}$ is a low-rank factor of rank $R$
# and $\boldsymbol{\kappa} \in \mathbb{R}^P_{>0}$ is a positive diagonal. The rank
# parameter controls how many latent sources of correlation the model can express.
# When $R = 1$ and $P = 2$, the model captures a single shared direction of
# variation between the two outputs.

# %% [markdown]
# ## Synthetic dataset
#
# We generate two correlated functions on the interval $[0, 1]$. The first output is
# $f_1(x) = \sin(2\pi x)$ and the second is a mixture
# $f_2(x) = 0.5\sin(2\pi x) + 0.5\cos(2\pi x)$, so the two outputs share a
# sinusoidal component. Both are corrupted by Gaussian noise with different standard
# deviations ($\sigma_1 = 0.1$, $\sigma_2 = 0.2$) to illustrate the per-output
# noise capability of the multi-output likelihood.

# %%
N = 40
P = 2
noise_stds = jnp.array([0.1, 0.2])

key, subkey1, subkey2 = jr.split(key, 3)
x = jnp.sort(jr.uniform(subkey1, shape=(N,), minval=0.0, maxval=1.0)).reshape(-1, 1)

f1 = lambda x: jnp.sin(2 * jnp.pi * x)
f2 = lambda x: 0.5 * f1(x) + 0.5 * jnp.cos(2 * jnp.pi * x)

y1 = f1(x) + jr.normal(subkey1, shape=x.shape) * noise_stds[0]
y2 = f2(x) + jr.normal(subkey2, shape=x.shape) * noise_stds[1]
y = jnp.hstack([y1, y2])  # [N, 2]

D = gpx.Dataset(X=x, y=y)

# %% [markdown]
# We plot the two outputs alongside the latent functions that generated them.

# %%
xtest = jnp.linspace(0.0, 1.0, 200).reshape(-1, 1)
output_labels = [r"$f_1$", r"$f_2$"]
latent_fns = [f1, f2]

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

for i, ax in enumerate(axes):
    ax.plot(x, y[:, i], "o", color=cols[i], alpha=0.6, label="Observations", ms=4)
    ax.plot(
        xtest, latent_fns[i](xtest), color=cols[i], ls="--", label="Latent function"
    )
    ax.set_title(output_labels[i])
    ax.set_xlabel(r"$x$")
    ax.legend(loc="best", fontsize=7)

axes[0].set_ylabel(r"$y$")

# %% [markdown]
# ## Model definition
#
# We construct the ICM model in three steps. First, we build a
# `CoregionalizationMatrix` with $P = 2$ outputs and rank $R = 1$. Second, we wrap
# an RBF base kernel together with the coregionalization matrix inside an
# `ICMKernel`. Third, we pair a zero-mean GP prior with a `MultiOutputGaussian`
# likelihood, which allows a separate noise variance for each output.

# %%
key, subkey = jr.split(key)

coreg = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=subkey)
kernel = gpx.kernels.ICMKernel(
    base_kernel=gpx.kernels.RBF(),
    coregionalization_matrix=coreg,
)

meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
likelihood = gpx.likelihoods.MultiOutputGaussian(
    num_datapoints=N, num_outputs=P, obs_stddev=1.0
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
    trainable=gpx.parameters.Parameter,
)

print(f"Optimised negative MLL: {-gpx.objectives.conjugate_mll(opt_posterior, D):.3f}")

# %% [markdown]
# ## Prediction
#
# The multi-output posterior returns predictions as a single Gaussian distribution
# over a flattened vector of length $MP$, where $M$ is the number of test points and
# $P$ is the number of outputs. The ordering is output-major: all $M$ values for
# output 1 appear first, followed by all $M$ values for output 2, and so on. We
# reshape the mean and extract per-output marginal variances from the diagonal of
# the joint covariance.

# %%
M = xtest.shape[0]
pred = opt_posterior.predict(xtest, train_data=D)

pred_mean = pred.mean.reshape(P, M).T  # [M, P]
pred_var = jnp.diag(pred.covariance()).reshape(P, M).T  # [M, P]
pred_std = jnp.sqrt(pred_var)

# %% [markdown]
# We now plot the predictive distribution for each output. The shaded region shows a
# 95% credible interval.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

for i, ax in enumerate(axes):
    ax.plot(x, y[:, i], "o", color=cols[i], alpha=0.5, label="Observations", ms=4)
    ax.plot(xtest, latent_fns[i](xtest), ls="--", color="grey", label="Latent function")
    ax.plot(xtest, pred_mean[:, i], color=cols[i], label="Predictive mean")
    ax.fill_between(
        xtest.squeeze(),
        pred_mean[:, i] - 2 * pred_std[:, i],
        pred_mean[:, i] + 2 * pred_std[:, i],
        color=cols[i],
        alpha=0.2,
        label="95% credible interval",
    )
    ax.set_title(output_labels[i])
    ax.set_xlabel(r"$x$")
    ax.legend(loc="best", fontsize=7)

axes[0].set_ylabel(r"$y$")

# %% [markdown]
# ## Learned coregionalization matrix
#
# The coregionalization matrix $\mathbf{B}$ encodes the learned correlations between
# outputs. Its off-diagonal entries indicate how strongly the outputs covary: a large
# positive entry between outputs $p$ and $q$ means they tend to increase together,
# while a value near zero suggests they are largely independent given the shared
# input kernel.
#
# We visualise $\mathbf{B}$ as a heatmap and print its entries.

# %%
B_learned = opt_posterior.prior.kernel.coregionalization_matrix.B

fig, ax = plt.subplots(figsize=(3.5, 3))
im = ax.imshow(
    B_learned,
    cmap="RdBu_r",
    vmin=-jnp.max(jnp.abs(B_learned)),
    vmax=jnp.max(jnp.abs(B_learned)),
)

for row in range(P):
    for col in range(P):
        ax.text(
            col,
            row,
            f"{B_learned[row, col]:.3f}",
            ha="center",
            va="center",
            fontsize=10,
        )

ax.set_xticks(range(P))
ax.set_yticks(range(P))
ax.set_xticklabels(output_labels)
ax.set_yticklabels(output_labels)
ax.set_title(r"Learned $\mathbf{B}$")
fig.colorbar(im, ax=ax, shrink=0.8)

# %% [markdown]
# Because $f_2$ is defined as a mixture that includes a scaled copy of $f_1$, we
# expect the model to recover a positive correlation between the two outputs. The
# diagonal entries reflect each output's marginal variance contribution from the
# shared latent process.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
