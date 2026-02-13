"""Tests for the Orthogonal Additive Kernel."""

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.kernels.additive.oak import _constrained_se_kernel


class TestConstrainedSEKernel:
    """Tests for the internal constrained SE kernel function."""

    def test_returns_scalar(self):
        """Constrained kernel eval on two scalars returns a scalar."""
        result = _constrained_se_kernel(
            x=jnp.array(0.5),
            y=jnp.array(-0.3),
            lengthscale=jnp.array(1.0),
            variance=jnp.array(1.0),
        )
        assert result.shape == ()

    def test_symmetric(self):
        """k_tilde(x, y) == k_tilde(y, x)."""
        x, y = jnp.array(0.5), jnp.array(-0.3)
        l, v = jnp.array(1.0), jnp.array(1.0)
        k_xy = _constrained_se_kernel(x, y, l, v)
        k_yx = _constrained_se_kernel(y, x, l, v)
        assert jnp.allclose(k_xy, k_yx, atol=1e-12)

    def test_orthogonality_constraint(self):
        """Integral of k_tilde(x, x') p(x) dx should be approx 0.

        We verify via Monte Carlo: sample x ~ N(0,1), fix x', compute
        mean of k_tilde(x_samples, x') -- should be near zero.
        """
        key = jr.PRNGKey(42)
        x_samples = jr.normal(key, shape=(50_000,))
        x_prime = jnp.array(0.7)
        l, v = jnp.array(1.5), jnp.array(1.0)

        k_vals = jax.vmap(
            lambda x: _constrained_se_kernel(x, x_prime, l, v)
        )(x_samples)
        mean_val = jnp.mean(k_vals)
        assert jnp.abs(mean_val) < 0.01, f"Expected ~0, got {mean_val}"

    def test_less_than_base_kernel(self):
        """Constrained kernel should be less than or equal to base SE kernel.

        Since we subtract a non-negative projection term, k_tilde(x,x) <= k(x,x).
        """
        x = jnp.array(0.0)
        l, v = jnp.array(1.0), jnp.array(1.0)
        k_tilde_xx = _constrained_se_kernel(x, x, l, v)
        k_xx = v * jnp.exp(-0.5 * jnp.square(x - x) / jnp.square(l))
        assert k_tilde_xx <= k_xx + 1e-10
