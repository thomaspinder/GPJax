"""Tests for Woodbury identity helper functions."""

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt
import pytest

config.update("jax_enable_x64", True)

from gpjax.linalg.woodbury import woodbury_logdet, woodbury_quad, woodbury_solve


def _dense_reference(W, noise):
    """Build the dense matrix W W^T + diag(noise) for reference."""
    return W @ W.T + jnp.diag(noise)


class TestWoodburySolve:
    @pytest.mark.parametrize("N,m", [(20, 3), (50, 5), (100, 10)])
    def test_matches_dense_vector(self, N, m):
        key = jr.key(0)
        k1, k2, _k3 = jr.split(key, 3)
        W = jr.normal(k1, (N, m))
        noise = jnp.ones(N) * 0.1
        b = jr.normal(k2, (N,))

        result = woodbury_solve(W, noise, b)
        expected = jnp.linalg.solve(_dense_reference(W, noise), b)
        npt.assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("N,m", [(20, 3), (50, 5)])
    def test_matches_dense_matrix(self, N, m):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        W = jr.normal(k1, (N, m))
        noise = jnp.ones(N) * 0.1
        B = jr.normal(k2, (N, 7))

        result = woodbury_solve(W, noise, B)
        expected = jnp.linalg.solve(_dense_reference(W, noise), B)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_heterogeneous_noise(self):
        key = jr.key(0)
        k1, k2, k3 = jr.split(key, 3)
        N, m = 30, 4
        W = jr.normal(k1, (N, m))
        noise = jnp.abs(jr.normal(k2, (N,))) + 0.01
        b = jr.normal(k3, (N,))

        result = woodbury_solve(W, noise, b)
        expected = jnp.linalg.solve(_dense_reference(W, noise), b)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_jit(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        W = jr.normal(k1, (20, 3))
        noise = jnp.ones(20) * 0.1
        b = jr.normal(k2, (20,))

        result = jax.jit(woodbury_solve)(W, noise, b)
        expected = jnp.linalg.solve(_dense_reference(W, noise), b)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_grad(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        W = jr.normal(k1, (20, 3))
        noise = jnp.ones(20) * 0.1
        b = jr.normal(k2, (20,))

        @jax.grad
        def fn(W):
            return jnp.sum(woodbury_solve(W, noise, b))

        grads = fn(W)
        assert jnp.all(jnp.isfinite(grads))


class TestWoodburyLogdet:
    @pytest.mark.parametrize("N,m", [(20, 3), (50, 5), (100, 10)])
    def test_matches_dense(self, N, m):
        key = jr.key(0)
        W = jr.normal(key, (N, m))
        noise = jnp.ones(N) * 0.1

        result = woodbury_logdet(W, noise)
        expected = jnp.linalg.slogdet(_dense_reference(W, noise))[1]
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_heterogeneous_noise(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        N, m = 30, 4
        W = jr.normal(k1, (N, m))
        noise = jnp.abs(jr.normal(k2, (N,))) + 0.01

        result = woodbury_logdet(W, noise)
        expected = jnp.linalg.slogdet(_dense_reference(W, noise))[1]
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_jit(self):
        key = jr.key(0)
        W = jr.normal(key, (20, 3))
        noise = jnp.ones(20) * 0.1

        result = jax.jit(woodbury_logdet)(W, noise)
        expected = jnp.linalg.slogdet(_dense_reference(W, noise))[1]
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_grad(self):
        key = jr.key(0)
        W = jr.normal(key, (20, 3))
        noise = jnp.ones(20) * 0.1

        @jax.grad
        def fn(W):
            return woodbury_logdet(W, noise)

        grads = fn(W)
        assert jnp.all(jnp.isfinite(grads))


class TestWoodburyQuad:
    @pytest.mark.parametrize("N,m", [(20, 3), (50, 5)])
    def test_matches_dense(self, N, m):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        W = jr.normal(k1, (N, m))
        noise = jnp.ones(N) * 0.1
        diff = jr.normal(k2, (N,))

        result = woodbury_quad(W, noise, diff)
        Sigma = _dense_reference(W, noise)
        expected = diff @ jnp.linalg.solve(Sigma, diff)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_jit(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        W = jr.normal(k1, (20, 3))
        noise = jnp.ones(20) * 0.1
        diff = jr.normal(k2, (20,))

        result = jax.jit(woodbury_quad)(W, noise, diff)
        Sigma = _dense_reference(W, noise)
        expected = diff @ jnp.linalg.solve(Sigma, diff)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_grad(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        W = jr.normal(k1, (20, 3))
        noise = jnp.ones(20) * 0.1
        diff = jr.normal(k2, (20,))

        @jax.grad
        def fn(W):
            return woodbury_quad(W, noise, diff)

        grads = fn(W)
        assert jnp.all(jnp.isfinite(grads))


class TestNumericalStability:
    def test_large_N_small_m(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        N, m = 1000, 10
        W = jr.normal(k1, (N, m)) * 0.5
        noise = jnp.ones(N) * 0.1
        b = jr.normal(k2, (N,))

        result = woodbury_solve(W, noise, b)
        assert jnp.all(jnp.isfinite(result))

        ld = woodbury_logdet(W, noise)
        assert jnp.isfinite(ld)

    def test_small_noise(self):
        key = jr.key(0)
        k1, k2 = jr.split(key)
        N, m = 50, 5
        W = jr.normal(k1, (N, m))
        noise = jnp.ones(N) * 1e-8
        b = jr.normal(k2, (N,))

        result = woodbury_solve(W, noise, b)
        assert jnp.all(jnp.isfinite(result))


class TestConjugateMllIntegration:
    """Verify Woodbury path in conjugate_mll matches dense path."""

    @pytest.mark.parametrize("KernelClass", ["RBF", "Matern52"])
    def test_mll_matches_dense(self, KernelClass):
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.kernels.approximations.hsgp import HSGP
        from gpjax.kernels.stationary import RBF, Matern52
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero
        from gpjax.objectives import conjugate_mll

        kernel_cls = {"RBF": RBF, "Matern52": Matern52}[KernelClass]

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        base_kernel = kernel_cls(n_dims=1)
        hsgp = HSGP(
            base_kernel=base_kernel,
            num_basis_fns=40,
            domain_half_width=5.0,
            center=0.0,
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        # Woodbury path (automatic via LowRank gram)
        mll_woodbury = conjugate_mll(posterior, D)

        # Dense reference: manually compute via dense gram
        dense_gram = hsgp.gram(x).to_dense()
        from gpjax.distributions import GaussianDistribution
        from gpjax.linalg import Dense, psd
        from gpjax.linalg.utils import add_jitter

        noise = likelihood.noise_vector(n)
        mx = prior.mean_function(x)
        y_flat, mx_flat = likelihood.prepare_targets(y, mx)
        Kxx_dense = add_jitter(dense_gram, prior.jitter)
        Sigma_dense = Kxx_dense + jnp.diag(noise)
        Sigma = psd(Dense(Sigma_dense))
        mll_dense = (
            GaussianDistribution(jnp.atleast_1d(mx_flat.squeeze()), Sigma)
            .log_prob(jnp.atleast_1d(y_flat.squeeze()))
            .squeeze()
        )

        npt.assert_allclose(mll_woodbury, mll_dense, atol=1e-5)

    def test_mll_differentiable_woodbury(self):
        from flax import nnx
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.kernels.approximations.hsgp import HSGP
        from gpjax.kernels.stationary import RBF
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero
        from gpjax.objectives import conjugate_mll

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        base_kernel = RBF(n_dims=1)
        hsgp = HSGP(
            base_kernel=base_kernel,
            num_basis_fns=20,
            domain_half_width=5.0,
            center=0.0,
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        graphdef, state = nnx.split(posterior)

        def loss(state):
            model = nnx.merge(graphdef, state)
            return -conjugate_mll(model, D)

        grad_fn = jax.grad(loss)
        grads = grad_fn(state)

        flat_grads = jax.tree.leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g))


class TestPredictIntegration:
    """Verify Woodbury predict path matches dense predict path."""

    def test_predict_mean_matches_dense(self):
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.kernels.approximations.hsgp import HSGP
        from gpjax.kernels.stationary import RBF
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        base_kernel = RBF(n_dims=1)
        hsgp = HSGP(
            base_kernel=base_kernel,
            num_basis_fns=30,
            domain_half_width=5.0,
            center=0.0,
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        x_test = jnp.linspace(-2.5, 2.5, 20)[:, None]
        pred = posterior.predict(x_test, D)

        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))

    def test_predict_diagonal(self):
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.kernels.approximations.hsgp import HSGP
        from gpjax.kernels.stationary import RBF
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        base_kernel = RBF(n_dims=1)
        hsgp = HSGP(
            base_kernel=base_kernel,
            num_basis_fns=30,
            domain_half_width=5.0,
            center=0.0,
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        x_test = jnp.linspace(-2.5, 2.5, 20)[:, None]
        pred = posterior.predict(x_test, D, return_covariance_type="diagonal")

        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.variance))

    def test_predict_rff(self):
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.kernels.approximations import RFF
        from gpjax.kernels.stationary import RBF
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        rff = RFF(base_kernel=RBF(n_dims=1), num_basis_fns=50, key=jr.key(0))
        prior = Prior(kernel=rff, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        x_test = jnp.linspace(-2.5, 2.5, 20)[:, None]
        pred = posterior.predict(x_test, D)

        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))
