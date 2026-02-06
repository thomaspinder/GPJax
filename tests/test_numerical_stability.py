"""Tests for numerical stability of kernel computations and GP predictions.

Parametrized across kernel types, these tests exercise edge cases that may
trigger NaN/Inf: very small lengthscales, very large variances, extreme input
ranges, and near-singular kernel matrices.
"""

import gpjax as gpx
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
)
from gpjax.linalg.utils import add_jitter
from gpjax.mean_functions import Zero
import jax.numpy as jnp
import jax.random as jr
import pytest

KERNEL_CLASSES = [RBF, Matern12, Matern32, Matern52, RationalQuadratic]


def _make_kernel(kernel_cls, lengthscale=1.0, variance=1.0):
    """Create a kernel instance with the given parameters."""
    return kernel_cls(lengthscale=lengthscale, variance=variance)


# ---------------------------------------------------------------------------
# Kernel gram matrix stability
# ---------------------------------------------------------------------------


class TestKernelGramStability:
    """Test that kernel gram matrices remain finite under extreme parameters."""

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_small_lengthscale(self, kernel_cls):
        """Very small lengthscale should produce finite gram matrix."""
        kernel = _make_kernel(kernel_cls, lengthscale=1e-6)
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        gram = kernel.gram(x).to_dense()
        assert jnp.all(jnp.isfinite(gram)), (
            f"Non-finite values in gram with small lengthscale for {kernel_cls.__name__}"
        )

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_large_variance(self, kernel_cls):
        """Large variance should produce finite gram matrix."""
        kernel = _make_kernel(kernel_cls, variance=1e6)
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        gram = kernel.gram(x).to_dense()
        assert jnp.all(jnp.isfinite(gram)), (
            f"Non-finite values in gram with large variance for {kernel_cls.__name__}"
        )

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_large_input_range(self, kernel_cls):
        """Extreme input ranges should produce finite gram matrix."""
        kernel = _make_kernel(kernel_cls)
        x = jnp.linspace(-1e4, 1e4, 10).reshape(-1, 1)
        gram = kernel.gram(x).to_dense()
        assert jnp.all(jnp.isfinite(gram)), (
            f"Non-finite values in gram with large inputs for {kernel_cls.__name__}"
        )

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_identical_points(self, kernel_cls):
        """Identical input points should produce finite gram matrix."""
        kernel = _make_kernel(kernel_cls)
        x = jnp.ones((5, 1))
        gram = kernel.gram(x).to_dense()
        assert jnp.all(jnp.isfinite(gram))

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_very_close_points(self, kernel_cls):
        """Very close but distinct points should produce finite gram matrix."""
        kernel = _make_kernel(kernel_cls)
        x = jnp.array([[0.0], [1e-10], [2e-10], [3e-10], [4e-10]])
        gram = kernel.gram(x).to_dense()
        assert jnp.all(jnp.isfinite(gram))


# ---------------------------------------------------------------------------
# Cholesky stability with jitter
# ---------------------------------------------------------------------------


class TestCholeskyStability:
    """Test that Cholesky decomposition succeeds with default jitter."""

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_cholesky_with_jitter(self, kernel_cls):
        """Cholesky should succeed on jittered kernel gram matrix."""
        kernel = _make_kernel(kernel_cls)
        x = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        gram = kernel.gram(x).to_dense()
        jittered = add_jitter(gram, jitter=1e-6)
        L = jnp.linalg.cholesky(jittered)
        assert jnp.all(jnp.isfinite(L)), f"Cholesky failed for {kernel_cls.__name__}"

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_cholesky_identical_points(self, kernel_cls):
        """Cholesky should succeed on near-singular gram (identical points + jitter)."""
        kernel = _make_kernel(kernel_cls)
        x = jnp.ones((10, 1))
        gram = kernel.gram(x).to_dense()
        jittered = add_jitter(gram, jitter=1e-6)
        L = jnp.linalg.cholesky(jittered)
        assert jnp.all(jnp.isfinite(L))


# ---------------------------------------------------------------------------
# GP posterior prediction stability
# ---------------------------------------------------------------------------


class TestPosteriorPredictionStability:
    """Test that full GP pipeline (prior -> posterior -> predict) stays finite."""

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_standard_predict_finite(self, kernel_cls):
        """Standard prediction should produce finite mean and covariance."""
        kernel = _make_kernel(kernel_cls)
        prior = gpx.gps.Prior(mean_function=Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=20)
        posterior = prior * likelihood

        x_train = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        y_train = jnp.sin(2 * jnp.pi * x_train)
        D = gpx.Dataset(X=x_train, y=y_train)

        x_test = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        pred_dist = posterior.predict(x_test, D)

        mean = pred_dist.mean
        cov = pred_dist.covariance()
        assert jnp.all(jnp.isfinite(mean)), f"Non-finite mean for {kernel_cls.__name__}"
        assert jnp.all(jnp.isfinite(cov)), (
            f"Non-finite covariance for {kernel_cls.__name__}"
        )

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_predict_with_noisy_data(self, kernel_cls):
        """Prediction with noisy data should stay finite."""
        kernel = _make_kernel(kernel_cls)
        prior = gpx.gps.Prior(mean_function=Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=30)
        posterior = prior * likelihood

        key = jr.key(42)
        x_train = jnp.linspace(-2.0, 2.0, 30).reshape(-1, 1)
        y_train = jnp.sin(x_train) + 0.5 * jr.normal(key, x_train.shape)
        D = gpx.Dataset(X=x_train, y=y_train)

        x_test = jnp.linspace(-3.0, 3.0, 15).reshape(-1, 1)
        pred_dist = posterior.predict(x_test, D)

        assert jnp.all(jnp.isfinite(pred_dist.mean))
        assert jnp.all(jnp.isfinite(pred_dist.covariance()))

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_predict_extrapolation(self, kernel_cls):
        """Predictions far from training data should stay finite."""
        kernel = _make_kernel(kernel_cls)
        prior = gpx.gps.Prior(mean_function=Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=10)
        posterior = prior * likelihood

        x_train = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        y_train = jnp.cos(x_train)
        D = gpx.Dataset(X=x_train, y=y_train)

        # Test inputs far from training range
        x_test = jnp.linspace(50.0, 100.0, 5).reshape(-1, 1)
        pred_dist = posterior.predict(x_test, D)

        assert jnp.all(jnp.isfinite(pred_dist.mean))
        assert jnp.all(jnp.isfinite(pred_dist.covariance()))

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_diagonal_predict_finite(self, kernel_cls):
        """Diagonal covariance prediction should stay finite."""
        kernel = _make_kernel(kernel_cls)
        prior = gpx.gps.Prior(mean_function=Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=20)
        posterior = prior * likelihood

        x_train = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        y_train = jnp.sin(x_train)
        D = gpx.Dataset(X=x_train, y=y_train)

        x_test = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        pred_dist = posterior.predict(x_test, D, return_covariance_type="diagonal")

        assert jnp.all(jnp.isfinite(pred_dist.mean))
        assert jnp.all(jnp.isfinite(pred_dist.covariance()))

    @pytest.mark.parametrize("kernel_cls", KERNEL_CLASSES)
    def test_predictive_variance_nonnegative(self, kernel_cls):
        """Diagonal of predictive covariance should be non-negative."""
        kernel = _make_kernel(kernel_cls)
        prior = gpx.gps.Prior(mean_function=Zero(), kernel=kernel)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=15)
        posterior = prior * likelihood

        x_train = jnp.linspace(0.0, 1.0, 15).reshape(-1, 1)
        y_train = x_train**2
        D = gpx.Dataset(X=x_train, y=y_train)

        x_test = jnp.linspace(-0.5, 1.5, 10).reshape(-1, 1)
        pred_dist = posterior.predict(x_test, D)

        var = jnp.diag(pred_dist.covariance())
        assert jnp.all(var >= -1e-6), (
            f"Negative variance for {kernel_cls.__name__}: {var}"
        )
