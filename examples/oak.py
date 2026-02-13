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
# # Orthogonal Additive Kernels
#
# In this notebook we demonstrate the **Orthogonal Additive Kernel (OAK)** of
# [Lu, Boukouvalas & Hensman (2022)](https://proceedings.mlr.press/v162/lu22b.html).
# OAK provides an interpretable additive Gaussian process model that decomposes
# the target function into main effects and interaction terms, whilst remaining
# a valid positive-definite kernel.  The key ingredients are:
#
# 1. A per-dimension **constrained SE kernel** that is orthogonal to the
#    constant function under the input density.
# 2. **Newton--Girard recursion** to efficiently combine these constrained
#    kernels into elementary symmetric polynomials up to a chosen interaction
#    order.
# 3. **Analytic Sobol indices** that quantify the relative importance of each
#    interaction order, enabling practitioners to understand which features
#    and feature interactions drive the model's predictions.
#
# We illustrate the full workflow on the UCI Auto MPG dataset.

# %%
# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from examples.utils import use_mpl_style

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.kernels.additive import OrthogonalAdditiveKernel, sobol_indices
    from gpjax.kernels.additive.oak import _constrained_se_kernel
    from gpjax.parameters import Parameter

key = jr.key(42)

# set the default style for plotting
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Mathematical background
#
# ### Additive GP decomposition
#
# A standard GP with a single kernel $k(\mathbf{x}, \mathbf{x}')$ treats all
# input dimensions jointly.  An **additive** GP instead decomposes the latent
# function as
#
# $$
# f(\mathbf{x})
# = f_0
# + \sum_{d=1}^{D} f_d(x_d)
# + \sum_{d < d'} f_{dd'}(x_d, x_{d'})
# + \cdots
# $$
#
# where $f_0$ is a constant offset, $f_d$ are first-order (main) effects,
# $f_{dd'}$ are second-order interactions, and so on.  Truncating at a maximum
# interaction order $\tilde{D} \le D$ yields a model that scales gracefully
# whilst retaining interpretability.
#
# ### The identifiability problem
#
# A naive additive decomposition is **not identifiable**: one can freely shift
# mass between the constant term and a main effect, or between a main effect
# and an interaction.  Lu et al. resolve this by requiring each component to
# be **orthogonal** to all lower-order components under the input density
# $p(\mathbf{x})$.  In particular, the first-order components satisfy
#
# $$
# \int f_d(x_d) \, p(x_d) \, \mathrm{d}x_d = 0 \quad \forall\, d.
# $$
#
# ### Constrained SE kernel
#
# Assuming a standard normal input density $p(x_d) = \mathcal{N}(0, 1)$, the
# orthogonality constraint can be enforced analytically.  The **constrained
# SE kernel** is (Eq. 10 of Lu et al.)
#
# $$
# \tilde{k}(x, y)
# = k(x, y)
# - \frac{\sigma^2 \ell \sqrt{\ell^2 + 2}}{\ell^2 + 1}
#   \exp\!\left(
#     -\frac{x^2 + y^2}{2(\ell^2 + 1)}
#   \right),
# $$
#
# where $k(x,y) = \sigma^2 \exp\!\bigl(-\tfrac{(x-y)^2}{2\ell^2}\bigr)$ is
# the standard SE kernel with lengthscale $\ell$ and variance $\sigma^2$.
# The subtracted projection term removes the component of $k$ that lies along
# the constant function under the $\mathcal{N}(0,1)$ measure.
#
# ### Newton--Girard recursion
#
# The additive kernel across all interaction orders up to $\tilde{D}$ is
#
# $$
# K(\mathbf{x}, \mathbf{x}')
# = \sum_{\ell=0}^{\tilde{D}} \sigma_\ell^2 \, e_\ell\!\bigl(
#     \tilde{k}_1(x_1, x_1'),\, \ldots,\, \tilde{k}_D(x_D, x_D')
#   \bigr),
# $$
#
# where $e_\ell$ denotes the $\ell$-th **elementary symmetric polynomial**
# and $\sigma_\ell^2$ are learnable order variances.  Computing
# $e_\ell$ directly via the combinatorial definition would be prohibitively
# expensive; instead GPJax uses the **Newton--Girard identities** which
# express $e_\ell$ recursively in terms of power sums
# $s_k = \sum_{d=1}^D z_d^k$:
#
# $$
# e_\ell = \frac{1}{\ell} \sum_{k=1}^{\ell} (-1)^{k-1}\, e_{\ell-k}\, s_k,
# \quad e_0 = 1.
# $$
#
# ### Sobol indices
#
# Once the model is fitted, the relative importance of each interaction order
# can be quantified via **Sobol indices**.  The Sobol index for order $d$ is
#
# $$
# S_d
# = \frac{
#     \sigma_d^4 \;\boldsymbol{\alpha}^\top E_d \,\boldsymbol{\alpha}
#   }{
#     \sum_{d'=1}^{\tilde{D}}
#       \sigma_{d'}^4 \;\boldsymbol{\alpha}^\top E_{d'}\,\boldsymbol{\alpha}
#   },
# $$
#
# where $\boldsymbol{\alpha} = (K + \sigma_n^2 I)^{-1}\mathbf{y}$ and $E_d$
# is the matrix-level elementary symmetric polynomial of the per-dimension
# integral matrices (see Appendix G.1 of the paper).

# %% [markdown]
# ## Dataset
#
# We use the [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto-mpg)
# dataset, which contains fuel consumption data for 392 cars described by 7
# continuous features (cylinders, displacement, horsepower, weight,
# acceleration, model year, and origin).
#
# Because the OAK kernel's constrained form assumes a standard normal input
# density ($\mu = 0$, $\sigma^2 = 1$), we fit a per-feature normalising flow
# that maps each marginal to an approximately standard normal distribution.
# Targets are z-score standardised.

# %%
from ucimlrepo import fetch_ucirepo

auto_mpg = fetch_ucirepo(id=9)
X_raw = auto_mpg.data.features
y_raw = auto_mpg.data.targets

# Drop rows with missing values
mask = ~(X_raw.isna().any(axis=1) | y_raw.isna().any(axis=1))
X_np = X_raw[mask].values.astype(np.float64)
y_np = y_raw[mask].values.astype(np.float64)

feature_names = list(X_raw.columns)
D = X_np.shape[1]
print(f"Dataset: {X_np.shape[0]} observations, {D} features")
print(f"Features: {feature_names}")

# %% [markdown]
# ### Normalising flow and train/test split
#
# The constrained SE kernel assumes $p(x_d) = \mathcal{N}(0,1)$.  Simple
# z-scoring removes the first two moments but cannot correct skewness or
# heavy tails.  We therefore fit a lightweight per-feature normalising flow
# (Shift → Log → Standardise → SinhArcsinh) that maps each marginal to an
# approximately standard normal distribution.

# %%
from gpjax.kernels.additive.transforms import fit_all_normalising_flows


# %%
# Z-score standardise targets only
y_mean, y_std = y_np.mean(axis=0), y_np.std(axis=0)
y_scaled = (y_np - y_mean) / y_std

# 80/20 train/test split (split BEFORE fitting the flow to avoid data leakage)
N = y_scaled.shape[0]
key, split_key = jr.split(key)
perm = jr.permutation(split_key, N)
n_train = int(0.8 * N)

train_idx = perm[:n_train]
test_idx = perm[n_train:]

y_train = jnp.array(y_scaled[train_idx])
y_test = jnp.array(y_scaled[test_idx])

# Keep raw X for plotting later
X_train_raw = X_np[train_idx]
X_test_raw = X_np[test_idx]

# Fit per-feature normalising flow on training data only
flows = fit_all_normalising_flows(jnp.asarray(X_train_raw))

X_train = jnp.column_stack(
    [flows[d](jnp.asarray(X_train_raw[:, d])) for d in range(D)]
)
X_test = jnp.column_stack(
    [flows[d](jnp.asarray(X_test_raw[:, d])) for d in range(D)]
)

train_data = gpx.Dataset(X=X_train, y=y_train)
test_data = gpx.Dataset(X=X_test, y=y_test)
print(f"\nTraining points: {X_train.shape[0]}, Test points: {X_test.shape[0]}")

# %% [markdown]
# ## Fitting an OAK GP
#
# We create $D$ independent RBF base kernels, one per input dimension, each
# operating on a single dimension via `active_dims=[i]`.  These are wrapped
# inside `OrthogonalAdditiveKernel` with `max_order=D` (i.e. we allow all
# interaction orders).  The kernel is then used in a standard conjugate GP
# workflow: define a prior and Gaussian likelihood, form the posterior, and
# optimise hyperparameters by maximising the marginal log-likelihood.

# %%
base_kernels = [gpx.kernels.RBF(active_dims=[i]) for i in range(D)]
oak_kernel = OrthogonalAdditiveKernel(base_kernels, max_order=D)

meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=oak_kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_train)
posterior = prior * likelihood

# %%
opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=train_data,
    trainable=Parameter,
)

# Evaluate on test set
latent_dist = opt_posterior.predict(
    X_test, train_data=train_data, return_covariance_type="diagonal"
)
predictive_dist = opt_posterior.likelihood(latent_dist)
oak_pred_mean = predictive_dist.mean

oak_rmse = jnp.sqrt(jnp.mean(jnp.square(oak_pred_mean - y_test.squeeze())))
oak_rmse_orig = float(oak_rmse) * float(y_std.squeeze())
print(f"OAK GP test RMSE (standardised): {oak_rmse:.4f}")
print(f"OAK GP test RMSE (original mpg): {oak_rmse_orig:.2f}")

# %% [markdown]
# ## RBF baseline
#
# For comparison we fit a standard ARD-RBF GP on the same data.  This kernel
# has a single lengthscale per dimension but models all interactions implicitly.

# %%
ard_kernel = gpx.kernels.RBF(
    active_dims=list(range(D)),
    lengthscale=jnp.ones(D),
)
ard_prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=ard_kernel)
ard_likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_train)
ard_posterior = ard_prior * ard_likelihood

opt_ard_posterior, ard_history = gpx.fit_scipy(
    model=ard_posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=train_data,
    trainable=Parameter,
)

ard_latent = opt_ard_posterior.predict(
    X_test, train_data=train_data, return_covariance_type="diagonal"
)
ard_pred_dist = opt_ard_posterior.likelihood(ard_latent)
ard_pred_mean = ard_pred_dist.mean

ard_rmse = jnp.sqrt(jnp.mean(jnp.square(ard_pred_mean - y_test.squeeze())))
ard_rmse_orig = float(ard_rmse) * float(y_std.squeeze())
print(f"ARD-RBF GP test RMSE (standardised): {ard_rmse:.4f}  ({ard_rmse_orig:.2f} mpg)")
print(f"OAK GP test RMSE (standardised):     {oak_rmse:.4f}  ({oak_rmse_orig:.2f} mpg)")

# %% [markdown]
# The OAK model achieves competitive predictive performance with the
# full ARD-RBF GP, whilst additionally providing the interpretability
# benefits shown below.

# %% [markdown]
# ## Sobol indices
#
# We now compute the analytic Sobol indices for each interaction order.
# These indicate what fraction of the posterior variance is explained by
# first-order (main) effects, second-order interactions, and so on.

# %%
noise_var = jnp.square(opt_posterior.likelihood.obs_stddev[...])
si = sobol_indices(
    opt_posterior.prior.kernel,
    X_train,
    y_train,
    float(noise_var),
)

# %%
fig, ax = plt.subplots(figsize=(7, 3))
orders = jnp.arange(1, len(si) + 1)
ax.bar(orders, si, color=cols[0])
ax.set_xlabel("Interaction order")
ax.set_ylabel("Sobol index")
ax.set_title("Sobol indices by interaction order")
ax.set_xticks(np.arange(1, len(si) + 1))

# Print cumulative values
cumulative = jnp.cumsum(si)
print("Sobol indices per interaction order:")
for i, (s, c) in enumerate(zip(si, cumulative)):
    print(f"  Order {i + 1}: {s:.4f}  (cumulative: {c:.4f})")

# %% [markdown]
# Typically the first-order (main) effects dominate, with higher-order
# interactions contributing progressively less.  This validates the additive
# modelling assumption for this dataset.

# %% [markdown]
# ## Decomposed additive components
#
# One of the key advantages of the OAK model is the ability to visualise
# each feature's individual contribution to the prediction.  We extract the
# top 4 first-order main effects and plot the posterior mean and a
# $\pm 2\sigma$ credible band for each, alongside a histogram of the
# training inputs.
#
# For each feature $d$, we evaluate the constrained kernel
# $\tilde{k}_d(x_*, X_{\mathrm{train},d})$ between a 1-D grid and the
# training points, then form the conditional mean and variance in the usual
# GP way.

# %%
oak_kern = opt_posterior.prior.kernel
lengthscales = oak_kern._lengthscales
variances = oak_kern._variances
order_var = oak_kern.order_variances[...]

# Build the full training kernel and solve for alpha.
K_train = oak_kern.gram(X_train).to_dense()
K_train_noisy = K_train + float(noise_var) * jnp.eye(n_train)
alpha = jnp.linalg.solve(K_train_noisy, y_train.squeeze())  # (N,)

# Rank features by their first-order Sobol contribution.
# sobol_d = sigma_1^4 * alpha^T L_d alpha,  where L_d is the per-dimension
# integral matrix.  This directly measures the variance explained by
# each feature's main effect.
from gpjax.kernels.additive.sobol import _sobol_integral_matrix

M_stack = jax.vmap(_sobol_integral_matrix)(X_train.T, lengthscales, variances)
feature_scores = jax.vmap(lambda M_d: alpha @ M_d @ alpha)(M_stack)
feature_scores = jnp.square(order_var[1]) * feature_scores
top4 = jnp.argsort(-feature_scores)[:4]

print("Learned lengthscales and first-order Sobol contributions:")
for i in jnp.argsort(-feature_scores):
    print(
        f"  {feature_names[int(i)]}: ls={lengthscales[int(i)]:.4f}, "
        f"sobol={feature_scores[int(i)]:.6f}"
    )
print()

n_grid = 200

fig, axes = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)

for idx, ax in enumerate(axes.flat):
    dim = int(top4[idx])
    ls_d = lengthscales[dim]
    var_d = variances[dim]
    fname = feature_names[dim]

    # 1-D grid for this feature
    lo = float(X_train[:, dim].min()) - 0.5
    hi = float(X_train[:, dim].max()) + 0.5
    x_grid = jnp.linspace(lo, hi, n_grid)

    # Compute constrained kernel between grid and training points.
    # Use a closure-creating function to avoid late-binding in vmap.
    def make_k_fn(ls, var):
        return lambda xg, xt: _constrained_se_kernel(xg, xt, ls, var)

    k_fn = make_k_fn(ls_d, var_d)
    # K_star: (n_grid, n_train) via double vmap
    K_star = jax.vmap(jax.vmap(k_fn, in_axes=(None, 0)), in_axes=(0, None))(
        x_grid, X_train[:, dim]
    )

    # Scale by order-1 variance
    K_star = order_var[1] * K_star

    # Posterior mean for this component
    f_mean = K_star @ alpha

    # Posterior variance: use the diagonal of K_star K_inv K_star^T
    # K_star_Kinv = K_star @ K_train_noisy^{-1}
    K_star_Kinv = jnp.linalg.solve(K_train_noisy, K_star.T).T  # (n_grid, n_train)

    # Self-kernel on the grid: k_tilde(x*, x*) for the diagonal
    k_self_fn = make_k_fn(ls_d, var_d)
    k_diag = jax.vmap(lambda x: k_self_fn(x, x))(x_grid)
    k_diag = order_var[1] * k_diag

    f_var = k_diag - jnp.sum(K_star_Kinv * K_star, axis=1)
    f_var = jnp.maximum(f_var, 0.0)
    f_std = jnp.sqrt(f_var)

    # Map the transformed grid back to original feature units for plotting
    x_grid_orig = flows[dim].inv(x_grid)

    # Plot posterior mean and +/- 2 sigma band (x-axis in original units)
    ax.plot(x_grid_orig, f_mean, color=cols[0], linewidth=2, label="Posterior mean")
    ax.fill_between(
        x_grid_orig,
        f_mean - 2 * f_std,
        f_mean + 2 * f_std,
        alpha=0.2,
        color=cols[0],
        label=r"$\pm 2\sigma$",
    )

    # Histogram of raw training inputs on a twin axis
    ax2 = ax.twinx()
    ax2.hist(
        X_train_raw[:, dim],
        bins=30,
        alpha=0.15,
        color=cols[1],
        density=True,
    )
    ax2.set_yticks([])

    ax.set_xlabel(fname)
    ax.set_ylabel("Effect")
    ax.set_title(f"{fname} (dim {dim})")
    ax.legend(loc="best", fontsize=8)

fig.suptitle("Top 4 first-order main effects", fontsize=14, y=1.02)

# %% [markdown]
# Each panel shows how the OAK model attributes predictive variation to
# individual features.  Features with large, clearly non-zero effects are
# those that the model identifies as important for predicting fuel
# consumption.  The uncertainty bands widen in regions where training data
# are sparse, reflecting the GP's epistemic uncertainty.

# %% [markdown]
# ## System configuration

# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Thomas Pinder'
