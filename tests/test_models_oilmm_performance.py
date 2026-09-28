"""OILMM structural correctness tests.

Wall-clock scaling assertions previously here have been moved to
``benchmarks/objectives.py::OilmmPredictSuite`` (tracked by ASV). What
remains here are tests that check the model has the expected structural
properties (e.g. that prediction can be JIT-compiled at all), not its
runtime performance.
"""

import gpjax as gpx
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr

config.update("jax_enable_x64", True)


def test_prediction_jit_compiles():
    """Verify prediction can be JIT compiled."""
    key = jr.key(123)
    model = gpx.models.create_oilmm(
        num_outputs=4,
        num_latent_gps=2,
        key=key,
    )

    N = 30
    X = jnp.linspace(0, 1, N).reshape(-1, 1)
    y = jr.normal(key, (N, 4))
    dataset = gpx.Dataset(X=X, y=y)
    posterior = model.condition(dataset)

    @jax.jit
    def predict_fn(X_test):
        pred = posterior.predict(X_test)
        return pred.loc, pred.covariance()

    X_test = jnp.linspace(0, 1, 10).reshape(-1, 1)
    mean, cov = predict_fn(X_test)

    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(cov))
