import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, trace


def test_numpyro_sample_into_gpjax_conjugate_mll():
    """Integration test: numpyro.sample values flow through GPJax constructors
    and conjugate_mll evaluates to a finite scalar."""
    key = jr.key(0)
    X = jr.uniform(key, shape=(10, 1))
    y = jnp.sin(X)
    D = gpx.Dataset(X=X, y=y)

    mll_value = None

    def model_fn():
        nonlocal mll_value
        lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
        obs_noise = numpyro.sample("obs_noise", dist.LogNormal(0.0, 1.0))

        kernel = gpx.kernels.RBF(lengthscale=lengthscale, variance=variance)
        meanf = gpx.mean_functions.Constant()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_noise)
        posterior = prior * likelihood

        mll_value = gpx.objectives.conjugate_mll(posterior, D)
        numpyro.factor("gp_log_lik", mll_value)

    with seed(rng_seed=42):
        tr = trace(model_fn).get_trace()

    assert "lengthscale" in tr
    assert "variance" in tr
    assert "obs_noise" in tr
    assert mll_value is not None
    assert jnp.isfinite(mll_value)
