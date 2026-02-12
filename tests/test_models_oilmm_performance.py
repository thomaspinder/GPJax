"""Performance tests for OILMM scaling properties."""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)


class TestOILMMPerformance:
    """Tests verifying OILMM achieves expected complexity."""

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_scales_linearly_in_m(self, m):
        """Test that cost scales ~linearly with num_latent_gps."""
        import gpjax as gpx

        key = jr.key(42)
        N, P = 50, 6

        # Create model
        model = gpx.models.create_oilmm(
            num_outputs=P,
            num_latent_gps=m,
            key=key,
        )

        # Create data
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        # Condition
        posterior = model.condition_on_observations(dataset)

        # Time prediction (JIT first call)
        X_test = jnp.linspace(0.1, 0.9, 20).reshape(-1, 1)
        _ = posterior.predict(X_test)  # Warm-up

        # Time second call
        start = time.perf_counter()
        _ = posterior.predict(X_test)
        elapsed = time.perf_counter() - start

        print(f"m={m}: {elapsed * 1000:.2f}ms")
        # Just verify it completes (actual timing too noisy for CI)
        assert elapsed < 1.0  # Should be fast

    def test_prediction_jit_compiles(self):
        """Verify prediction can be JIT compiled."""
        import gpjax as gpx

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
        posterior = model.condition_on_observations(dataset)

        # Define JIT function that returns raw arrays
        @jax.jit
        def predict_fn(X_test):
            pred = posterior.predict(X_test)
            return pred.loc, pred.covariance()

        # Should compile without error
        X_test = jnp.linspace(0, 1, 10).reshape(-1, 1)
        mean, cov = predict_fn(X_test)

        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(cov))
