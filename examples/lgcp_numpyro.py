# %%
import jax.numpy as jnp
from jax import random
from jax import config
import numpy as np

import gpjax as gpx
from gpjax import numpyro_extras
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

import matplotlib.pyplot as plt

# Enable x64 support for JAX
config.update("jax_enable_x64", True)

# Set random seed
key = random.PRNGKey(42)

# Configure MCMC
num_warmup = 1000
num_samples = 1000
num_chains = 4

# Set device count for numpyro for parallel chains
numpyro.set_host_device_count(num_chains)

# %%
# 1. Data: Coal Mining Disasters (1851-1962)
# Counts of disasters per year
counts = jnp.array([
    4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1
], dtype=jnp.float64)

years = jnp.arange(1851, 1851 + len(counts), dtype=jnp.float64).reshape(-1, 1)
# Normalize years for better numerical stability in GP
years_norm = (years - years.min()) / (years.max() - years.min())

# %%
# 2. Model Definition
# We model the log-intensity log(lambda(t)) as a Gaussian Process.
# lambda(t) = exp(f(t))
# y_i ~ Poisson(lambda(t_i))

# Mean function: Constant mean
mean_f = gpx.mean_functions.Constant(constant=jnp.array([0.0]))

# Kernel: Matern52
# We expect changes over decades, so lengthscale should be non-trivial.
# Since x is normalized to [0, 1], a lengthscale of 0.1 corresponds to ~11 years.
kernel = gpx.kernels.Matern52(lengthscale=0.2, variance=0.5)

prior = gpx.gps.Prior(mean_function=mean_f, kernel=kernel)

def model(x, y):
    # Register GPJax parameters (lengthscale, variance, mean_constant) with Numpyro
    gp = numpyro_extras.register_parameters(prior)

    # Sample the latent function f at the input locations x
    f = numpyro.sample("f", gp(x))

    # The intensity is exp(f)
    rate = jnp.exp(f)

    # Observation model: Poisson
    numpyro.sample("y", dist.Poisson(rate), obs=y)

# %%
# 3. Inference
rng_key, rng_key_ = random.split(key)

kernel_nuts = NUTS(model, target_accept_prob=0.9)
mcmc = MCMC(
    kernel_nuts,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True,
    jit_model_args=True,
)

# Run MCMC
# Note: We pass years_norm for stability, but we'll plot against original years
mcmc.run(rng_key_, x=years_norm, y=counts)

# %%
# 4. Analysis & Plotting
mcmc.print_summary()

# Extract samples
samples = mcmc.get_samples()
f_samples = samples["f"]
intensity_samples = jnp.exp(f_samples)

# Compute statistics
mean_intensity = jnp.mean(intensity_samples, axis=0)
lower_ci = jnp.percentile(intensity_samples, 2.5, axis=0)
upper_ci = jnp.percentile(intensity_samples, 97.5, axis=0)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(years.flatten(), counts, color="gray", alpha=0.5, label="Observed Counts", width=1.0)
plt.plot(years.flatten(), mean_intensity, color="C0", label="Posterior Mean Intensity", linewidth=2)
plt.fill_between(years.flatten(), lower_ci, upper_ci, color="C0", alpha=0.3, label="95% CI")

plt.xlabel("Year")
plt.ylabel("Number of Disasters")
plt.title("Coal Mining Disasters: Log-Gaussian Cox Process (GPJax + Numpyro)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lgcp_coal_mining.png")
# plt.show()

# Trace plot for diagnostics
az.plot_trace(mcmc, var_names=["kernel.lengthscale", "kernel.variance"])
plt.tight_layout()