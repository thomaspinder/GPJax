"""Tests for the Orthogonal Additive Kernel."""

from jax import config

config.update("jax_enable_x64", True)

from gpjax.kernels import RBF
from gpjax.kernels.additive.oak import (
    OrthogonalAdditiveKernel,
    _constrained_se_kernel,
    _newton_girard,
)
from gpjax.parameters import NonNegativeReal
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


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
        ls, v = jnp.array(1.0), jnp.array(1.0)
        k_xy = _constrained_se_kernel(x, y, ls, v)
        k_yx = _constrained_se_kernel(y, x, ls, v)
        assert jnp.allclose(k_xy, k_yx, atol=1e-12)

    def test_orthogonality_constraint(self):
        """Integral of k_tilde(x, x') p(x) dx should be approx 0.

        We verify via Monte Carlo: sample x ~ N(0,1), fix x', compute
        mean of k_tilde(x_samples, x') -- should be near zero.
        """
        key = jr.PRNGKey(42)
        x_samples = jr.normal(key, shape=(50_000,))
        x_prime = jnp.array(0.7)
        ls, v = jnp.array(1.5), jnp.array(1.0)

        k_vals = jax.vmap(lambda x: _constrained_se_kernel(x, x_prime, ls, v))(
            x_samples
        )
        mean_val = jnp.mean(k_vals)
        assert jnp.abs(mean_val) < 0.01, f"Expected ~0, got {mean_val}"

    def test_less_than_base_kernel(self):
        """Constrained kernel should be less than or equal to base SE kernel.

        Since we subtract a non-negative projection term, k_tilde(x,x) <= k(x,x).
        """
        x = jnp.array(0.0)
        ls, v = jnp.array(1.0), jnp.array(1.0)
        k_tilde_xx = _constrained_se_kernel(x, x, ls, v)
        k_xx = v * jnp.exp(-0.5 * jnp.square(x - x) / jnp.square(ls))
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
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels, order_variances=ov)
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


class TestOrthogonalAdditiveKernelCall:
    """Tests for OAK __call__ (scalar pair evaluation)."""

    def test_returns_scalar(self):
        """Calling the kernel on two points returns a scalar."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        x = jnp.array([0.5, -0.3, 0.1])
        y = jnp.array([0.2, 0.4, -0.6])
        result = kernel(x, y)
        assert result.shape == ()

    def test_symmetric(self):
        """k(x, y) == k(y, x)."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        x = jnp.array([0.5, -0.3, 0.1])
        y = jnp.array([0.2, 0.4, -0.6])
        assert jnp.allclose(kernel(x, y), kernel(y, x), atol=1e-12)

    def test_max_order_1_is_gam(self):
        """With max_order=1, kernel is offset + sum of 1D constrained kernels.

        k(x,y) = sigma^2_0 + sigma^2_1 * sum_d k_tilde_d(x_d, y_d)
        """
        base_kernels = [
            RBF(active_dims=[i], lengthscale=1.0, variance=1.0) for i in range(3)
        ]
        kernel = OrthogonalAdditiveKernel(
            base_kernels=base_kernels,
            max_order=1,
            order_variances=jnp.ones(2),
        )
        x = jnp.array([0.5, -0.3, 0.1])
        y = jnp.array([0.2, 0.4, -0.6])
        result = kernel(x, y)

        # Manually compute: sigma^2_0 * 1 + sigma^2_1 * sum(k_tilde_d)
        manual = 1.0  # sigma^2_0 * e_0
        for d in range(3):
            manual += _constrained_se_kernel(x[d], y[d], jnp.array(1.0), jnp.array(1.0))
        assert jnp.allclose(result, manual, atol=1e-10)

    def test_brute_force_d3_full_order(self):
        """D=3, max_order=3: compare against explicit enumeration of all subsets."""
        ls = [1.0, 1.5, 0.8]
        base_kernels = [
            RBF(active_dims=[i], lengthscale=ls[i], variance=1.0) for i in range(3)
        ]
        ov = jnp.array([0.5, 1.0, 0.8, 0.3])
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels, order_variances=ov)
        x = jnp.array([0.5, -0.3, 0.1])
        y = jnp.array([0.2, 0.4, -0.6])
        result = kernel(x, y)

        # Brute-force computation
        z = jnp.array(
            [
                _constrained_se_kernel(x[d], y[d], jnp.array(ls[d]), jnp.array(1.0))
                for d in range(3)
            ]
        )
        # e_0 = 1
        # e_1 = z0 + z1 + z2
        # e_2 = z0*z1 + z0*z2 + z1*z2
        # e_3 = z0*z1*z2
        e0 = 1.0
        e1 = z[0] + z[1] + z[2]
        e2 = z[0] * z[1] + z[0] * z[2] + z[1] * z[2]
        e3 = z[0] * z[1] * z[2]
        expected = ov[0] * e0 + ov[1] * e1 + ov[2] * e2 + ov[3] * e3
        assert jnp.allclose(result, expected, atol=1e-10)


class TestOrthogonalAdditiveKernelProperties:
    """Tests for gram matrix, PSD, JIT, gradients."""

    @pytest.fixture
    def kernel_and_data(self):
        """3D OAK kernel with small dataset."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        key = jr.PRNGKey(0)
        x = jr.normal(key, shape=(10, 3))
        return kernel, x

    def test_gram_shape(self, kernel_and_data):
        """Gram matrix has shape (N, N)."""
        kernel, x = kernel_and_data
        K = kernel.gram(x)
        assert K.to_dense().shape == (10, 10)

    def test_gram_symmetric(self, kernel_and_data):
        """Gram matrix is symmetric."""
        kernel, x = kernel_and_data
        K = kernel.gram(x).to_dense()
        assert jnp.allclose(K, K.T, atol=1e-10)

    def test_gram_psd(self, kernel_and_data):
        """Gram matrix eigenvalues are non-negative."""
        kernel, x = kernel_and_data
        K = kernel.gram(x).to_dense()
        eigvals = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigvals > -1e-6), f"Negative eigenvalue: {eigvals.min()}"

    def test_cross_covariance_shape(self, kernel_and_data):
        """Cross-covariance has shape (N, M)."""
        kernel, x = kernel_and_data
        key = jr.PRNGKey(1)
        y = jr.normal(key, shape=(7, 3))
        Kxy = kernel.cross_covariance(x, y)
        assert Kxy.shape == (10, 7)

    def test_cross_covariance_matches_gram_diagonal(self, kernel_and_data):
        """cross_covariance(x, x) diagonal matches gram diagonal."""
        kernel, x = kernel_and_data
        K_gram = kernel.gram(x).to_dense()
        K_cross = kernel.cross_covariance(x, x)
        assert jnp.allclose(jnp.diag(K_gram), jnp.diag(K_cross), atol=1e-10)

    def test_jit_compatible(self, kernel_and_data):
        """Kernel gram computation works under jax.jit."""
        kernel, x = kernel_and_data
        K_eager = kernel.gram(x).to_dense()
        K_jit = jax.jit(lambda xx: kernel.gram(xx).to_dense())(x)
        assert jnp.allclose(K_eager, K_jit, atol=1e-10)

    def test_gradient_flows(self):
        """Gradients w.r.t. kernel parameters are finite and non-zero."""
        from flax import nnx

        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        x = jr.normal(jr.PRNGKey(0), shape=(5, 3))

        graphdef, state = nnx.split(kernel)

        def loss_fn(state):
            k = nnx.merge(graphdef, state)
            K = k.gram(x).to_dense()
            return jnp.sum(K)

        grads = jax.grad(loss_fn)(state)
        # Check order_variances gradient exists and is finite
        ov_grad = grads.order_variances.value
        assert jnp.all(jnp.isfinite(ov_grad))
        assert not jnp.allclose(ov_grad, 0.0)


class TestRankFirstOrder:
    """Tests for rank_first_order decomposition utility."""

    @pytest.fixture
    def kernel_and_data(self):
        """3D OAK kernel with small dataset."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels, max_order=2)
        key = jr.PRNGKey(0)
        x = jr.normal(key, shape=(20, 3))
        y = jr.normal(jr.PRNGKey(1), shape=(20, 1))
        return kernel, x, y

    def test_shape(self, kernel_and_data):
        """Returns shape (D,)."""
        from gpjax.kernels.additive.decompose import rank_first_order

        kernel, x, y = kernel_and_data
        scores = rank_first_order(kernel, x, y, noise_variance=0.1)
        assert scores.shape == (3,)

    def test_non_negative(self, kernel_and_data):
        """All scores are non-negative."""
        from gpjax.kernels.additive.decompose import rank_first_order

        kernel, x, y = kernel_and_data
        scores = rank_first_order(kernel, x, y, noise_variance=0.1)
        assert jnp.all(scores >= -1e-10)

    def test_matches_manual(self, kernel_and_data):
        """Matches manual computation via _sobol_integral_matrix."""
        from gpjax.kernels.additive.decompose import rank_first_order
        from gpjax.kernels.additive.sobol import _sobol_integral_matrix

        kernel, x, y = kernel_and_data
        scores = rank_first_order(kernel, x, y, noise_variance=0.1)

        # Manual computation
        N = x.shape[0]
        K = kernel.gram(x).to_dense()
        K_noisy = K + 0.1 * jnp.eye(N)
        alpha = jnp.linalg.solve(K_noisy, y.squeeze())
        ov = kernel.order_variances[...]
        ls = kernel._lengthscales
        vs = kernel._variances
        M_stack = jax.vmap(_sobol_integral_matrix)(x.T, ls, vs)
        manual = jnp.square(ov[1]) * jax.vmap(lambda M: alpha @ M @ alpha)(M_stack)
        assert jnp.allclose(scores, manual, atol=1e-10)

    def test_jit_compatible(self, kernel_and_data):
        """Works under jax.jit."""
        from gpjax.kernels.additive.decompose import rank_first_order

        kernel, x, y = kernel_and_data
        eager = rank_first_order(kernel, x, y, noise_variance=0.1)
        jitted = jax.jit(lambda xx, yy: rank_first_order(kernel, xx, yy, 0.1))(x, y)
        assert jnp.allclose(eager, jitted, atol=1e-10)


class TestPredictFirstOrder:
    """Tests for predict_first_order decomposition utility."""

    @pytest.fixture
    def kernel_and_data(self):
        """3D OAK kernel with small dataset and grid."""
        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels, max_order=2)
        key = jr.PRNGKey(0)
        x = jr.normal(key, shape=(20, 3))
        y = jr.normal(jr.PRNGKey(1), shape=(20, 1))
        x_grid = jnp.linspace(-2.0, 2.0, 50)
        return kernel, x, y, x_grid

    def test_shapes(self, kernel_and_data):
        """Returns (M,) mean and (M,) variance."""
        from gpjax.kernels.additive.decompose import predict_first_order

        kernel, x, y, x_grid = kernel_and_data
        mean, var = predict_first_order(kernel, x, y, 0.1, dim=0, x_grid=x_grid)
        assert mean.shape == (50,)
        assert var.shape == (50,)

    def test_non_negative_variance(self, kernel_and_data):
        """Variance is non-negative."""
        from gpjax.kernels.additive.decompose import predict_first_order

        kernel, x, y, x_grid = kernel_and_data
        _, var = predict_first_order(kernel, x, y, 0.1, dim=0, x_grid=x_grid)
        assert jnp.all(var >= 0.0)

    def test_finite_mean(self, kernel_and_data):
        """Mean values are finite."""
        from gpjax.kernels.additive.decompose import predict_first_order

        kernel, x, y, x_grid = kernel_and_data
        mean, _ = predict_first_order(kernel, x, y, 0.1, dim=0, x_grid=x_grid)
        assert jnp.all(jnp.isfinite(mean))

    def test_matches_manual(self, kernel_and_data):
        """Matches manual computation."""
        from gpjax.kernels.additive.decompose import predict_first_order
        from gpjax.kernels.additive.oak import _constrained_se_kernel

        kernel, x, y, x_grid = kernel_and_data
        mean, _var = predict_first_order(kernel, x, y, 0.1, dim=0, x_grid=x_grid)

        # Manual
        N = x.shape[0]
        K = kernel.gram(x).to_dense()
        K_noisy = K + 0.1 * jnp.eye(N)
        alpha = jnp.linalg.solve(K_noisy, y.squeeze())
        ls_d = kernel._lengthscales[0]
        var_d = kernel._variances[0]
        ov = kernel.order_variances[...]

        K_star = jax.vmap(
            jax.vmap(
                lambda xg, xt: _constrained_se_kernel(xg, xt, ls_d, var_d),
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )(x_grid, x[:, 0])
        K_star = ov[1] * K_star
        manual_mean = K_star @ alpha
        assert jnp.allclose(mean, manual_mean, atol=1e-10)

    def test_jit_compatible(self, kernel_and_data):
        """Works under jax.jit."""
        from gpjax.kernels.additive.decompose import predict_first_order

        kernel, x, y, x_grid = kernel_and_data
        mean_e, var_e = predict_first_order(kernel, x, y, 0.1, dim=0, x_grid=x_grid)
        mean_j, var_j = jax.jit(
            lambda xx, yy, xg: predict_first_order(kernel, xx, yy, 0.1, 0, xg)
        )(x, y, x_grid)
        assert jnp.allclose(mean_e, mean_j, atol=1e-10)
        assert jnp.allclose(var_e, var_j, atol=1e-10)

    def test_different_dims_differ(self, kernel_and_data):
        """Different dimensions produce different predictions."""
        from gpjax.kernels.additive.decompose import predict_first_order

        kernel, x, y, x_grid = kernel_and_data
        mean0, _ = predict_first_order(kernel, x, y, 0.1, dim=0, x_grid=x_grid)
        mean1, _ = predict_first_order(kernel, x, y, 0.1, dim=1, x_grid=x_grid)
        assert not jnp.allclose(mean0, mean1, atol=1e-6)


class TestSobolIndices:
    """Tests for Sobol index computation."""

    def test_returns_correct_shape(self):
        """Sobol indices have shape (max_order,) -- one per interaction order."""
        from gpjax.kernels.additive.sobol import sobol_indices

        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        key = jr.PRNGKey(0)
        x = jr.normal(key, shape=(20, 3))
        y = jr.normal(jr.PRNGKey(1), shape=(20, 1))
        indices = sobol_indices(kernel, x, y, noise_variance=0.1)
        assert indices.shape == (3,)  # orders 1, 2, 3 (not offset)

    def test_normalized_to_sum_to_one(self):
        """Normalized Sobol indices sum to 1."""
        from gpjax.kernels.additive.sobol import sobol_indices

        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        key = jr.PRNGKey(0)
        x = jr.normal(key, shape=(20, 3))
        y = jr.normal(jr.PRNGKey(1), shape=(20, 1))
        indices = sobol_indices(kernel, x, y, noise_variance=0.1)
        assert jnp.allclose(jnp.sum(indices), 1.0, atol=1e-6)

    def test_non_negative(self):
        """All Sobol indices are non-negative."""
        from gpjax.kernels.additive.sobol import sobol_indices

        base_kernels = [RBF(active_dims=[i]) for i in range(3)]
        kernel = OrthogonalAdditiveKernel(base_kernels=base_kernels)
        key = jr.PRNGKey(0)
        x = jr.normal(key, shape=(20, 3))
        y = jr.normal(jr.PRNGKey(1), shape=(20, 1))
        indices = sobol_indices(kernel, x, y, noise_variance=0.1)
        assert jnp.all(indices >= -1e-10)
