"""Tests for the Orthogonal Additive Kernel."""

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gpjax.kernels import RBF
from gpjax.kernels.additive.oak import (
    OrthogonalAdditiveKernel,
    _constrained_se_kernel,
    _newton_girard,
)
from gpjax.parameters import NonNegativeReal


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


class TestNewtonGirard:
    """Tests for Newton-Girard elementary symmetric polynomial computation."""

    def test_order_1_is_sum(self):
        """e_1(z1, z2, z3) = z1 + z2 + z3."""
        z = jnp.array([2.0, 3.0, 5.0])
        e = _newton_girard(z, max_order=1)
        # e[0] = 1 (e_0), e[1] = sum(z)
        assert jnp.allclose(e[0], 1.0)
        assert jnp.allclose(e[1], 10.0)

    def test_order_2_is_pairwise_products(self):
        """e_2(z1, z2, z3) = z1*z2 + z1*z3 + z2*z3."""
        z = jnp.array([2.0, 3.0, 5.0])
        e = _newton_girard(z, max_order=2)
        expected_e2 = 2.0 * 3.0 + 2.0 * 5.0 + 3.0 * 5.0  # 31.0
        assert jnp.allclose(e[2], expected_e2)

    def test_order_3_is_triple_product(self):
        """e_3(z1, z2, z3) = z1*z2*z3."""
        z = jnp.array([2.0, 3.0, 5.0])
        e = _newton_girard(z, max_order=3)
        assert jnp.allclose(e[3], 30.0)

    def test_full_order_4d(self):
        """Full check for D=4 against brute-force."""
        z = jnp.array([1.0, 2.0, 3.0, 4.0])
        e = _newton_girard(z, max_order=4)
        # e_1 = 1+2+3+4 = 10
        assert jnp.allclose(e[1], 10.0)
        # e_2 = 1*2+1*3+1*4+2*3+2*4+3*4 = 35
        assert jnp.allclose(e[2], 35.0)
        # e_3 = 1*2*3+1*2*4+1*3*4+2*3*4 = 50
        assert jnp.allclose(e[3], 50.0)
        # e_4 = 1*2*3*4 = 24
        assert jnp.allclose(e[4], 24.0)

    def test_truncated_order(self):
        """max_order < D returns only up to that order."""
        z = jnp.array([1.0, 2.0, 3.0, 4.0])
        e = _newton_girard(z, max_order=2)
        assert e.shape == (3,)  # e_0, e_1, e_2
        assert jnp.allclose(e[1], 10.0)
        assert jnp.allclose(e[2], 35.0)


class TestOrthogonalAdditiveKernelInit:
    """Tests for OAK construction."""

    def test_basic_construction(self):
        """Construct a 3D OAK with default settings."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        assert kernel.max_order == 3
        assert len(kernel.base_kernels) == 3
        assert kernel.order_variances[...].shape == (4,)  # includes sigma^2_0

    def test_custom_max_order(self):
        """max_order < D truncates interaction orders."""
        base_kernels = [RBF(active_dims=[i]) for i in range(5)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels, max_order=2)
        assert kernel.max_order == 2
        assert kernel.order_variances[...].shape == (3,)  # e_0, e_1, e_2

    def test_custom_order_variances(self):
        """User can provide initial order variances."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        ov = jnp.array([0.5, 1.0, 0.5, 0.1])
        kernel = OrthogonalAdditiveKernel(
            base_kernels=base_kernels, order_variances=ov
        )
        assert jnp.allclose(kernel.order_variances[...], ov)

    def test_order_variances_are_trainable(self):
        """Order variances should be NonNegativeReal parameters."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        assert isinstance(kernel.order_variances, NonNegativeReal)

    def test_max_order_exceeds_D_raises(self):
        """max_order > D is invalid."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        with pytest.raises(ValueError, match="max_order"):
            OrthogonalAdditiveKernel(base_kernels=base_kernels, max_order=5)

    def test_empty_base_kernels_raises(self):
        """Must provide at least one base kernel."""
        with pytest.raises(ValueError, match="at least one"):
            OrthogonalAdditiveKernel(base_kernels=[])
