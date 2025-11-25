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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spatial Modelling: Linear Regression vs. Gaussian Processes
#
# In this notebook, we explore the benefits of combining structured mean functions with Gaussian
# Processes (GPs) for modelling spatial data. We will compare two approaches:
# 1.  **A Baseline Linear Model**: A standard Bayesian linear regression that assumes the target
#     variable is a linear combination of the inputs.
# 2.  **A Joint Linear + GP Model**: A semi-parametric model that captures the global linear trend
#     while using a GP to model the non-linear spatial residuals.
#
# Crucially, this example demonstrates the seamless integration between **GPJax** and **NumPyro**.
# We will show how `GPJax` defines the GP prior and likelihood, while `NumPyro` handles the
# Hamiltonian Monte Carlo (HMC) inference for both the linear coefficients and the GP
# hyperparameters simultaneously.

# %% [markdown]
# ## 1. Setup and Data Simulation
#
# First, we import the necessary libraries. We enable 64-bit precision in JAX to ensure numerical
# stability during matrix decompositions.
#
# We simulate a 2D spatial dataset ($N=200$) on a domain $[0, 5]    imes [0, 5]$. The true
# generating process consists of:
# *   **A Linear Trend**: $y_{\text{lin}} = 2x_1 - 1x_2 + 1.5$
# *   **A Spatial Residual**: $y_{\text{res}} = \sin(x_1) \cos(x_2)$
# *   **Observation Noise**: $\epsilon \sim \mathcal{N}(0, 0.1^2)$
#
# This structure effectively masks the non-linear signal within a dominant linear trend, posing a
# challenge for simple linear models.

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import gpjax as gpx
from gpjax.numpyro_extras import register_parameters
import matplotlib.pyplot as plt

# Enable x64 precision for better stability
jax.config.update("jax_enable_x64", True)

print("Spatial Linear GP Comparison Example")

# --- Step 2: Data Simulation ---
N = 200
key = jr.key(42)
key_x, key_noise = jr.split(key)

# Simulate X in [0, 5] x [0, 5]
X = jr.uniform(key_x, shape=(N, 2), minval=0.0, maxval=5.0)

# True Linear Trend
true_slope = jnp.array([2.0, -1.0])
true_intercept = 1.5
y_lin = X @ true_slope + true_intercept

# Non-linear Spatial Residual
y_res = jnp.sin(X[:, 0]) * jnp.cos(X[:, 1])

# Total Signal + Noise
y_clean = y_lin + y_res
noise_std = 0.1
y = y_clean + noise_std * jr.normal(key_noise, shape=y_clean.shape)

print(f"Generated {N} data points.")

# %% [markdown]
# ## 2. Baseline Linear Model
#
# We begin by defining a standard Bayesian linear regression model in NumPyro. This model assumes
# that the data can be fully explained by a hyperplane and Gaussian noise.
#
# $$\begin{aligned} \mathbf{w} &\sim \mathcal{N}(\mathbf{0}, 5\mathbf{I}) \\
# b &\sim \mathcal{N}(0, 5) \\
# \sigma &\sim \text{LogNormal}(0, 1) \\
# \mathbf{y} &\sim \mathcal{N}(\mathbf{X}\mathbf{w} + b, \sigma^2 \mathbf{I}) \end{aligned} $$
#
# We use the No-U-Turn Sampler (NUTS) to estimate the posterior distributions of the slope
# $\mathbf{w}$, intercept $b$, and noise $\sigma$.

# %%
def linear_model(X, Y=None):
    # Priors
    slope = numpyro.sample("slope", dist.Normal(0.0, 5.0).expand([2]))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))
    obs_noise = numpyro.sample("obs_noise", dist.LogNormal(0.0, 1.0))

    # Mean function
    mu = X @ slope + intercept
    numpyro.deterministic("mu", mu)

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, obs_noise), obs=Y)

# Run MCMC for Linear Model
print("\nRunning MCMC for Baseline Linear Model...")
nuts_kernel_lin = NUTS(linear_model)
mcmc_lin = MCMC(nuts_kernel_lin, num_warmup=500, num_samples=1000, num_chains=1)
mcmc_lin.run(key, X, y)
mcmc_lin.print_summary()

# %% [markdown]
# ## 3. Joint Linear + GP Model
#
# Now we define the joint model. Here, the GP accounts for the residuals that the linear model
# cannot explain.
#
# $$ y(\mathbf{x}) = \underbrace{\mathbf{w}^T \mathbf{x} + b}_{\text{Linear Mean}} +
# \underbrace{f(\mathbf{x})}_{\text{GP Residual}} + \epsilon $$
#
# ### GPJax and NumPyro Integration
#
# This section highlights the interoperability between GPJax and NumPyro.
#
# 1.  **GP Definition**: We define the GP prior in `GPJax` using an RBF kernel and a zero mean
#     function (since the linear trend is handled explicitly). We attach `dist.LogNormal` priors to
#     the kernel's hyperparameters (lengthscale and variance) directly within the GPJax object.
# 2.  **`register_parameters`**: Inside the NumPyro model, we call
#     `gpx.numpyro_extras.register_parameters(gp_posterior)`. This function traverses the GPJax
#     object, identifies parameters with attached priors, and registers them as NumPyro sample
#     sites. It returns a new GPJax object where the parameters have been replaced by the values
#     sampled by NumPyro.
# 3.  **Conjugate Marginal Log-Likelihood**: We compute the exact marginal log-likelihood (MLL) of
#     the residuals under the GP prior using `gpx.objectives.conjugate_mll`. This term is added to
#     the potential function using `numpyro.factor`, guiding the sampler.

# %%
# GP Definition
lengthscale = gpx.parameters.PositiveReal(1.0, prior=dist.LogNormal(0.0, 1.0))
variance = gpx.parameters.PositiveReal(1.0, prior=dist.LogNormal(0.0, 1.0))

# active_dims=[0, 1] ensures the kernel operates on both spatial dimensions
kernel = gpx.kernels.RBF(active_dims=[0, 1], lengthscale=lengthscale, variance=variance)
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

obs_stddev = gpx.parameters.NonNegativeReal(0.1, prior=dist.LogNormal(0.0, 1.0))
likelihood = gpx.likelihoods.Gaussian(num_datapoints=N, obs_stddev=obs_stddev)
gp_posterior = prior * likelihood

def joint_model(X, Y, gp_posterior, X_new=None):
    # 1. Sample Linear Model Parameters
    slope = numpyro.sample("slope", dist.Normal(0.0, 5.0).expand([2]))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 5.0))

    trend = X @ slope + intercept

    # 2. Register GP Parameters with NumPyro
    # This draws samples for lengthscale, variance, and obs_noise from their priors
    p_posterior = register_parameters(gp_posterior)

    if Y is not None:
        # Calculate residuals for the GP to model
        residuals = Y - trend
        # Reshape residuals to (N, 1) for GPJax Dataset
        residuals = residuals.reshape(-1, 1)
        D_resid = gpx.Dataset(X=X, y=residuals)

        # 3. Compute GP Marginal Log-Likelihood
        mll = gpx.objectives.conjugate_mll(p_posterior, D_resid)
        numpyro.factor("gp_log_lik", mll)

    if X_new is not None:
        # Prediction logic
        if Y is not None:
             residuals = Y - trend
             residuals = residuals.reshape(-1, 1)
             D_resid = gpx.Dataset(X=X, y=residuals)

             # Compute predictive distribution for the GP component
             latent_dist = p_posterior.predict(X_new, train_data=D_resid)
             f_new = numpyro.sample("f_new", latent_dist)
             f_new = f_new.reshape((-1, 1))

             # Combine Linear Trend + GP Residual
             total_prediction = (X_new @ slope + intercept).reshape(-1, 1) + f_new
             numpyro.deterministic("y_pred", total_prediction)

# Run MCMC for Joint Model
print("\nRunning MCMC for Joint Linear + GP Model...")
# Use a closure to pass the static gp_posterior object to the model
def joint_model_wrapper(X, Y, X_new=None):
    joint_model(X, Y, gp_posterior, X_new)

nuts_kernel_joint = NUTS(joint_model_wrapper)
mcmc_joint = MCMC(nuts_kernel_joint, num_warmup=500, num_samples=1000, num_chains=1)
mcmc_joint.run(key, X, y)
mcmc_joint.print_summary()

# %% [markdown]
# ## 4. Comparison and Visualization
#
# We evaluate both models by comparing their Root Mean Squared Error (RMSE) against the true
# noise-free signal. We also visualise the predictions over the 2D domain.
#
# We expect the **Joint Model** to significantly outperform the linear baseline because it can
# capture the spatial correlations ($\sin(x_1)\cos(x_2)$) that the linear model ignores.

# %%
# --- Step 6: Comparison & Visualization ---

# 1. Prediction on Training Data (for RMSE)
# Linear Model
samples_lin = mcmc_lin.get_samples()
predictive_lin = Predictive(linear_model, samples_lin, return_sites=["mu"])
preds_lin = predictive_lin(jr.key(1), X=X)["mu"]
mean_pred_lin = jnp.mean(preds_lin, axis=0)

# Joint Model
samples_joint = mcmc_joint.get_samples()
predictive_joint = Predictive(joint_model_wrapper, samples_joint, return_sites=["y_pred"])
preds_joint = predictive_joint(jr.key(2), X=X, Y=y, X_new=X)["y_pred"]
mean_pred_joint = jnp.mean(preds_joint, axis=0)

# Calculate RMSE
rmse_lin = jnp.sqrt(jnp.mean((mean_pred_lin.flatten() - y_clean.flatten())**2))
rmse_joint = jnp.sqrt(jnp.mean((mean_pred_joint.flatten() - y_clean.flatten())**2))

print(f"\nRMSE Comparison (vs True Signal):")
print(f"Linear Model: {rmse_lin:.4f}")
print(f"Joint Model:  {rmse_joint:.4f}")

# 2. Visualization on a Grid
n_grid = 30
x1 = jnp.linspace(0, 5, n_grid)
x2 = jnp.linspace(0, 5, n_grid)
X1, X2 = jnp.meshgrid(x1, x2)
X_grid = jnp.column_stack([X1.ravel(), X2.ravel()])

# True Signal on Grid
y_grid_true = (X_grid @ true_slope + true_intercept) + (jnp.sin(X_grid[:, 0]) * jnp.cos(X_grid[:, 1]))

# Linear Prediction on Grid
preds_lin_grid = predictive_lin(jr.key(3), X=X_grid)["mu"]
mean_pred_lin_grid = jnp.mean(preds_lin_grid, axis=0)

# Joint Prediction on Grid
preds_joint_grid = predictive_joint(jr.key(4), X=X, Y=y, X_new=X_grid)["y_pred"]
mean_pred_joint_grid = jnp.mean(preds_joint_grid, axis=0)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Truth
c0 = axes[0].tricontourf(X_grid[:,0], X_grid[:,1], y_grid_true, levels=20, cmap='viridis')
axes[0].set_title("True Signal")
plt.colorbar(c0, ax=axes[0])

# Linear
c1 = axes[1].tricontourf(X_grid[:,0], X_grid[:,1], mean_pred_lin_grid.flatten(), levels=20, cmap='viridis')
axes[1].set_title(f"Linear Model (RMSE: {rmse_lin:.2f})")
plt.colorbar(c1, ax=axes[1])

# Joint
c2 = axes[2].tricontourf(X_grid[:,0], X_grid[:,1], mean_pred_joint_grid.flatten(), levels=20, cmap='viridis')
axes[2].set_title(f"Joint Model (RMSE: {rmse_joint:.2f})")
plt.colorbar(c2, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.scatter(X[:,0], X[:,1], c='k', s=10, alpha=0.3, label="Data")

plt.tight_layout()
