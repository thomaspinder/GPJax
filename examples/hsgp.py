# -*- coding: utf-8 -*-
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
# # Hilbert Space Gaussian Processes
#
# Standard GP inference requires inverting an $n \times n$ Gram matrix at
# $\mathcal{O}(n^3)$ cost, which limits practical use to a few thousand points.
# The Hilbert Space Gaussian Process (HSGP) approximation
# ([Solin and Sarkka, 2020](https://link.springer.com/article/10.1007/s11222-019-09886-w))
# projects the GP onto a truncated basis of Laplacian eigenfunctions, reducing
# the cost to $\mathcal{O}(nm^2)$ for setup and $\mathcal{O}(m^3)$ per
# optimisation step, where $m \ll n$ is the number of basis functions. See
# also the PyMC tutorial by
# [Orduz and Capretto](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html)
# and the standalone introduction by
# [Orduz (2022)](https://juanitorduz.github.io/hsgp_intro/).

# %%
from jax import config

config.update("jax_enable_x64", True)

from examples.utils import use_mpl_style
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

key = jr.key(42)
use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## The HSGP approximation
#
# For a stationary kernel $k$ with spectral density $S$, the HSGP approximates
# the covariance on a bounded domain $[-L, L]$:
#
# $$k(x, x') \approx \sum_{j=1}^{m} S\!\left(\sqrt{\lambda_j}\right) \phi_j(x)\,\phi_j(x').$$
#
# The eigenfunctions $\phi_j(x) = L^{-1/2}\sin\!\bigl(j\pi(x + L) / 2L\bigr)$
# are sinusoids of increasing frequency, and the eigenvalues
# $\lambda_j = (j\pi / 2L)^2$ grow quadratically. The spectral density $S$
# weights each eigenfunction's contribution. In matrix form the approximate
# Gram matrix is $\tilde{K} = \Phi\,\Lambda\,\Phi^\top$ where
# $\Lambda = \mathrm{diag}\!\bigl(S(\sqrt{\lambda_1}), \ldots, S(\sqrt{\lambda_m})\bigr)$.
#
# Two parameters govern approximation quality:
#
# - **$m$** (number of basis functions): more terms capture shorter
#   lengthscales, at higher computational cost.
# - **$L$** (domain half-width): must enclose all inputs after centering,
#   but setting it much larger than necessary wastes basis functions.

# %% [markdown]
# ## Eigenfunctions and spectral weights
#
# The HSGP has two ingredients: Laplacian eigenfunctions (which depend only on
# the domain, not the kernel) and spectral weights (which encode the kernel's
# properties). We visualise both below.

# %%
x_grid = jnp.linspace(-3.0, 3.0, 300)[:, None]

base_rbf = gpx.kernels.RBF(n_dims=1)
hsgp_rbf = gpx.kernels.HSGP(
    base_kernel=base_rbf, num_basis_fns=20, domain_half_width=4.0, center=0.0
)

phi = hsgp_rbf.eigenfunctions(x_grid)

fig, ax = plt.subplots(figsize=(7.5, 2.5))
for j in range(6):
    ax.plot(x_grid, phi[:, j], label=f"$\\phi_{{{j + 1}}}$", color=cols[j % len(cols)])
ax.set_xlabel("$x$")
ax.set_ylabel("$\\phi_j(x)$")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
ax.set_title("Laplacian eigenfunctions")

# %% [markdown]
# The eigenfunctions are sinusoids of increasing frequency, forming an
# orthonormal basis on $[-L, L]$. Because they are independent of the kernel
# hyperparameters, they need only be evaluated once for a given domain.
#
# The spectral weights $S(\sqrt{\lambda_j})$ determine each eigenfunction's
# contribution. We compare weights for the RBF kernel (Gaussian spectral
# density, rapid decay) and the Matern-5/2 (heavier-tailed, slower decay).

# %%
base_m52 = gpx.kernels.Matern52(n_dims=1)
hsgp_m52 = gpx.kernels.HSGP(
    base_kernel=base_m52, num_basis_fns=20, domain_half_width=4.0, center=0.0
)

weights_rbf = hsgp_rbf.spectral_weights()
weights_m52 = hsgp_m52.spectral_weights()

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5), sharey=True)
basis_indices = jnp.arange(1, 21)

axes[0].stem(basis_indices, weights_rbf, linefmt=cols[0], markerfmt="o", basefmt=" ")
axes[0].set_title("RBF")
axes[0].set_xlabel("Basis index $j$")
axes[0].set_ylabel("$S(\\sqrt{\\lambda_j})$")

axes[1].stem(basis_indices, weights_m52, linefmt=cols[1], markerfmt="o", basefmt=" ")
axes[1].set_title("Matern-5/2")
axes[1].set_xlabel("Basis index $j$")

# %% [markdown]
# The RBF weights decay rapidly, so few basis functions suffice. The
# Matern-5/2 retains appreciable weight at higher frequencies and needs more
# terms for the same approximation quality.

# %% [markdown]
# ## Approximation quality
#
# How faithfully does the HSGP Gram matrix reproduce the exact kernel? We
# compare exact and approximate Gram matrices at $m = 10$, $25$, and $50$ for
# both the RBF and Matern-5/2 kernels.

# %%
x_small = jnp.linspace(-3.0, 3.0, 80)[:, None]
m_values = [10, 25, 50]


def gram_comparison(base_kernel, x, m_values):
    """Return exact Gram and HSGP Gram matrices for several values of m."""
    gram_exact = base_kernel.gram(x).to_dense()
    gram_approximations = []
    for num_basis in m_values:
        hsgp = gpx.kernels.HSGP(
            base_kernel=base_kernel,
            num_basis_fns=num_basis,
            domain_half_width=4.0,
            center=0.0,
        )
        gram_approximations.append(hsgp.gram(x).to_dense())
    return gram_exact, gram_approximations


gram_exact_rbf, gram_hsgps_rbf = gram_comparison(base_rbf, x_small, m_values)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
vmin, vmax = float(gram_exact_rbf.min()), float(gram_exact_rbf.max())
titles = ["Exact"] + [f"HSGP ($m = {m}$)" for m in m_values]
matrices = [gram_exact_rbf, *gram_hsgps_rbf]

for ax, matrix, title in zip(axes, matrices, titles, strict=True):
    ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="inferno")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("RBF kernel", fontsize=10, y=1.02)

# %% [markdown]
# For the RBF kernel, the approximation is near-indistinguishable from exact
# by $m = 25$, reflecting the rapid spectral weight decay.

# %%
gram_exact_m52, gram_hsgps_m52 = gram_comparison(base_m52, x_small, m_values)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
vmin, vmax = float(gram_exact_m52.min()), float(gram_exact_m52.max())
titles = ["Exact"] + [f"HSGP ($m = {m}$)" for m in m_values]
matrices = [gram_exact_m52, *gram_hsgps_m52]

for ax, matrix, title in zip(axes, matrices, titles, strict=True):
    ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="inferno")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Matern-5/2 kernel", fontsize=10, y=1.02)

# %% [markdown]
# The Matern-5/2 converges more slowly, with visible discrepancies at
# $m = 10$. Rougher kernels retain more high-frequency content. In practice,
# $m$ between 20 and 50 suffices for most one-dimensional problems.

# %% [markdown]
# ## Regression with real data
#
# We apply the HSGP to daily nitrogen dioxide (NO$_2$) measurements from
# Marylebone Road in central London, one of the UK's most heavily trafficked
# roadside sites. The data come from the
# [Automatic Urban and Rural Network (AURN)](https://uk-air.defra.gov.uk/networks/site-info?site_id=MY1).
# We aggregate to daily means over approximately ten years (2016--2025),
# giving several thousand data points where exact GP inference is impractical.

# %%
from pathlib import Path
import tempfile
from urllib.request import urlretrieve
import warnings

import pandas as pd
import rdata

SITE = "MY1"
YEARS = range(2016, 2026)

frames = []
for year in YEARS:
    url = f"https://uk-air.defra.gov.uk/openair/R_data/{SITE}_{year}.RData"
    tmp = Path(tempfile.gettempdir()) / f"{SITE}_{year}.RData"
    urlretrieve(url, tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.read_rda(str(tmp))
    frames.append(parsed[next(iter(parsed.keys()))])

hourly = pd.concat(frames, ignore_index=True)
hourly["date"] = pd.to_datetime(hourly["date"], unit="s")
daily = hourly[["date", "NO2"]].set_index("date").resample("D").mean().dropna()

# %%
fig, ax = plt.subplots(figsize=(7.5, 2.5))
ax.plot(daily.index, daily["NO2"], linewidth=0.5, color=cols[0])
ax.set_xlabel("Date")
ax.set_ylabel("NO$_2$ ($\\mu$g m$^{-3}$)")
ax.set_title("Daily mean NO$_2$ at Marylebone Road, London")

# %% [markdown]
# Clear features include an annual cycle from heating-season emissions, an
# abrupt dip during the 2020 COVID-19 lockdowns, and a gradual downward trend
# from tightening vehicle emission standards.

# %%
# Convert dates to a numeric index and standardise for numerical stability.
t = (daily.index - daily.index[0]).days.values.astype(float)
y_obs = daily["NO2"].values

t_mean, t_std = t.mean(), t.std()
y_mean, y_std = y_obs.mean(), y_obs.std()

t_norm = ((t - t_mean) / t_std)[:, None]
y_norm = ((y_obs - y_mean) / y_std)[:, None]

D = gpx.Dataset(X=t_norm, y=y_norm)

# %% [markdown]
# We build a GP prior with an HSGP kernel using a Matern-3/2 base kernel. We
# use $m = 30$ basis functions and set $L$ to 1.2 times the half-range of
# the normalised inputs, providing a modest buffer beyond the data boundary.

# %%
data_half_range = float((t_norm.max() - t_norm.min()) / 2.0)
L = 1.2 * data_half_range

base_kernel = gpx.kernels.Matern32(n_dims=1)
hsgp_kernel = gpx.kernels.HSGP(
    base_kernel=base_kernel,
    num_basis_fns=30,
    domain_half_width=L,
    center=float(t_norm.mean()),
)

prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=hsgp_kernel)
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
posterior = prior * likelihood

# %% [markdown]
# We optimise the negative conjugate marginal log-likelihood using L-BFGS-B,
# learning the kernel lengthscale, variance, and observation noise.

# %%
opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=D,
    trainable=gpx.parameters.Parameter,
)

print(
    f"Learned lengthscale: {opt_posterior.prior.kernel.base_kernel.lengthscale[...]:.4f}"
)
print(
    f"Learned variance:    {opt_posterior.prior.kernel.base_kernel.variance[...]:.4f}"
)
print(f"Learned obs. noise:  {opt_posterior.likelihood.obs_stddev[...] ** 2:.4f}")

# %% [markdown]
# With optimised hyperparameters, we compute the posterior predictive on a
# fine grid spanning the training domain.

# %%
t_test = jnp.linspace(t_norm.min() - 0.05, t_norm.max() + 0.05, 500)[:, None]

latent_dist = opt_posterior.predict(
    t_test, train_data=D, return_covariance_type="diagonal"
)
predictive_dist = opt_posterior.likelihood(latent_dist)

pred_mean = predictive_dist.mean * y_std + y_mean
pred_std = jnp.sqrt(predictive_dist.variance) * y_std

# Convert test grid back to dates for plotting.
t_test_days = (t_test.squeeze() * t_std + t_mean).astype(int)
dates_test = daily.index[0] + pd.to_timedelta(t_test_days, unit="D")

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.scatter(daily.index, y_obs, s=1, alpha=0.3, color=cols[0], label="Observations")
ax.plot(dates_test, pred_mean, color=cols[1], linewidth=1.5, label="Predictive mean")
ax.fill_between(
    dates_test,
    pred_mean - 2 * pred_std,
    pred_mean + 2 * pred_std,
    alpha=0.2,
    color=cols[1],
    label="Two sigma",
)
ax.set_xlabel("Date")
ax.set_ylabel("NO$_2$ ($\\mu$g m$^{-3}$)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("HSGP posterior for daily NO$_2$ at Marylebone Road")

# %% [markdown]
# The HSGP posterior captures seasonal structure and the long-term downward
# trend, with credible intervals that widen where data are sparser. The
# computation completed in seconds on a single CPU; an exact GP over the same
# dataset would require forming and decomposing a dense Gram matrix with over
# ten million entries.

# %% [markdown]
# ## References
#
# - Solin, A. and Sarkka, S. (2020).
#   [Hilbert Space Methods for Reduced-Rank Gaussian Process Regression](https://link.springer.com/article/10.1007/s11222-019-09886-w).
#   *Statistics and Computing*.
# - Orduz, J. and Martin, O. A.
#   [PyMC HSGP tutorial](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html).
# - Orduz, J. (2022).
#   [Introduction to the HSGP](https://juanitorduz.github.io/hsgp_intro/).
# - UK [Automatic Urban and Rural Network](https://uk-air.defra.gov.uk/networks/site-info?site_id=MY1)
#   air quality data via the openair data service.
