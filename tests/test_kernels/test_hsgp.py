"""Tests for the Hilbert Space Gaussian Process (HSGP) kernel approximation."""

from gpjax.kernels.approximations.hsgp import HSGP
from gpjax.kernels.stationary import RBF, Matern12, Matern32, Matern52
from gpjax.linalg.operators import Dense
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt
import pytest

config.update("jax_enable_x64", True)


class TestEigenvalues:
    def test_eigenvalue_count(self):
        kernel = HSGP(
            base_kernel=RBF(n_dims=1), num_basis_fns=10, domain_half_width=3.0
        )
        evals = kernel.eigenvalues()
        assert evals.shape == (10,)

    def test_eigenvalue_formula(self):
        """sqrt(lambda_j) = j * pi / (2 * L)."""
        L = 5.0
        m = 4
        kernel = HSGP(base_kernel=RBF(n_dims=1), num_basis_fns=m, domain_half_width=L)
        evals = kernel.eigenvalues()
        expected = jnp.arange(1, m + 1) * jnp.pi / (2.0 * L)
        npt.assert_allclose(evals, expected)

    def test_eigenvalues_increase(self):
        kernel = HSGP(
            base_kernel=RBF(n_dims=1), num_basis_fns=20, domain_half_width=3.0
        )
        evals = kernel.eigenvalues()
        assert jnp.all(jnp.diff(evals) > 0)


class TestEigenfunctions:
    def test_shape(self):
        kernel = HSGP(
            base_kernel=RBF(n_dims=1),
            num_basis_fns=10,
            domain_half_width=3.0,
            center=0.0,
        )
        x = jnp.linspace(-2, 2, 50)[:, None]
        phi = kernel.eigenfunctions(x)
        assert phi.shape == (50, 10)

    def test_orthonormality(self):
        """Eigenfunctions should be approximately orthonormal over [-L, L]."""
        L = 3.0
        m = 5
        kernel = HSGP(
            base_kernel=RBF(n_dims=1), num_basis_fns=m, domain_half_width=L, center=0.0
        )
        # Dense grid for numerical integration
        n_quad = 10000
        x = jnp.linspace(-L, L, n_quad)[:, None]
        phi = kernel.eigenfunctions(x)
        dx = 2.0 * L / n_quad
        gram = phi.T @ phi * dx  # Approximate integral
        npt.assert_allclose(gram, jnp.eye(m), atol=1e-2)

    def test_zero_at_boundaries(self):
        """Eigenfunctions should be zero at x = -L and x = L."""
        L = 3.0
        kernel = HSGP(
            base_kernel=RBF(n_dims=1), num_basis_fns=5, domain_half_width=L, center=0.0
        )
        boundaries = jnp.array([[-L], [L]])
        phi = kernel.eigenfunctions(boundaries)
        npt.assert_allclose(phi, 0.0, atol=1e-12)


class TestCentering:
    def test_explicit_center(self):
        kernel = HSGP(
            base_kernel=RBF(n_dims=1),
            num_basis_fns=5,
            domain_half_width=3.0,
            center=1.0,
        )
        assert kernel._center == 1.0

    def test_auto_center(self):
        kernel = HSGP(base_kernel=RBF(n_dims=1), num_basis_fns=5, domain_half_width=3.0)
        x = jnp.linspace(2.0, 8.0, 50)[:, None]
        _ = kernel.eigenfunctions(x)
        assert kernel._center == pytest.approx(5.0)

    def test_auto_center_persists(self):
        """Once set, auto-center should not change on subsequent calls."""
        kernel = HSGP(base_kernel=RBF(n_dims=1), num_basis_fns=5, domain_half_width=3.0)
        x1 = jnp.linspace(0, 10, 50)[:, None]
        x2 = jnp.linspace(-5, 5, 50)[:, None]
        _ = kernel.eigenfunctions(x1)
        center_after_first = kernel._center
        _ = kernel.eigenfunctions(x2)
        assert kernel._center == center_after_first


class TestComputeBasis:
    def test_returns_tuple(self):
        kernel = HSGP(
            base_kernel=RBF(n_dims=1),
            num_basis_fns=10,
            domain_half_width=3.0,
            center=0.0,
        )
        x = jnp.linspace(-2, 2, 50)[:, None]
        phi, sqrt_psd = kernel.compute_basis(x)
        assert phi.shape == (50, 10)
        assert sqrt_psd.shape == (10,)

    def test_sqrt_psd_positive(self):
        kernel = HSGP(
            base_kernel=RBF(n_dims=1),
            num_basis_fns=10,
            domain_half_width=3.0,
            center=0.0,
        )
        x = jnp.linspace(-2, 2, 50)[:, None]
        _, sqrt_psd = kernel.compute_basis(x)
        assert jnp.all(sqrt_psd > 0)


class TestGram:
    @pytest.mark.parametrize("KernelClass", [RBF, Matern12, Matern32, Matern52])
    def test_gram_shape_and_psd(self, KernelClass):
        base = KernelClass(n_dims=1)
        hsgp = HSGP(
            base_kernel=base, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        x = jnp.linspace(-3, 3, 30)[:, None]
        linop = hsgp.gram(x)
        assert isinstance(linop, Dense)
        K = linop.to_dense()
        assert K.shape == (30, 30)
        # PSD check
        evals, _ = jnp.linalg.eigh(K + 1e-6 * jnp.eye(30))
        assert jnp.all(evals > 0)

    @pytest.mark.parametrize("KernelClass", [RBF, Matern12, Matern32, Matern52])
    def test_cross_covariance_shape(self, KernelClass):
        base = KernelClass(n_dims=1)
        hsgp = HSGP(
            base_kernel=base, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        x1 = jnp.linspace(-3, 3, 30)[:, None]
        x2 = jnp.linspace(-2, 2, 20)[:, None]
        Kxy = hsgp.cross_covariance(x1, x2)
        assert Kxy.shape == (30, 20)

    def test_gram_symmetric(self):
        base = RBF(n_dims=1)
        hsgp = HSGP(
            base_kernel=base, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        x = jnp.linspace(-3, 3, 30)[:, None]
        K = hsgp.gram(x).to_dense()
        npt.assert_allclose(K, K.T, atol=1e-12)

    @pytest.mark.parametrize("KernelClass", [RBF, Matern12, Matern32, Matern52])
    def test_diagonal(self, KernelClass):
        base = KernelClass(n_dims=1)
        hsgp = HSGP(
            base_kernel=base, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        x = jnp.linspace(-3, 3, 30)[:, None]
        diag = hsgp.diagonal(x)
        K = hsgp.gram(x).to_dense()
        npt.assert_allclose(jnp.diag(diag.to_dense()), jnp.diag(K), atol=1e-10)


class TestConvergence:
    @pytest.mark.parametrize("KernelClass", [RBF, Matern32, Matern52])
    def test_gram_converges_to_exact(self, KernelClass):
        """With large m and appropriate L, HSGP Gram should converge to exact."""
        base = KernelClass(n_dims=1)
        x = jnp.linspace(-1, 1, 30)[:, None]
        exact = base.gram(x).to_dense()

        hsgp_coarse = HSGP(
            base_kernel=base, num_basis_fns=10, domain_half_width=5.0, center=0.0
        )
        hsgp_fine = HSGP(
            base_kernel=base, num_basis_fns=80, domain_half_width=5.0, center=0.0
        )

        err_coarse = jnp.linalg.norm(exact - hsgp_coarse.gram(x).to_dense(), ord="fro")
        err_fine = jnp.linalg.norm(exact - hsgp_fine.gram(x).to_dense(), ord="fro")

        # Finer approximation should have smaller error
        assert err_fine < err_coarse

    def test_rbf_close_to_exact(self):
        """RBF with large m should be very close to exact."""
        base = RBF(n_dims=1)
        x = jnp.linspace(-1, 1, 20)[:, None]
        exact = base.gram(x).to_dense()

        hsgp = HSGP(
            base_kernel=base, num_basis_fns=100, domain_half_width=5.0, center=0.0
        )
        approx = hsgp.gram(x).to_dense()
        max_err = jnp.max(jnp.abs(exact - approx))
        assert max_err < 0.01


class TestValidation:
    def test_nonstationary_kernel_rejected(self):
        from gpjax.kernels.nonstationary import Linear

        with pytest.raises(TypeError):
            HSGP(base_kernel=Linear(1), num_basis_fns=10, domain_half_width=3.0)

    def test_pointwise_call_raises(self):
        hsgp = HSGP(base_kernel=RBF(n_dims=1), num_basis_fns=10, domain_half_width=3.0)
        with pytest.raises(RuntimeError):
            hsgp(jnp.array([1.0]), jnp.array([2.0]))


class TestIntegration:
    def test_prior_posterior_pipeline(self):
        """HSGP should work as a drop-in kernel in the Prior/Posterior pipeline."""
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
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
            base_kernel=base_kernel, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        # MLL should return a finite scalar
        mll = conjugate_mll(posterior, D)
        assert jnp.isfinite(mll)

    def test_predict(self):
        """HSGP posterior should produce finite mean and variance."""
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        base_kernel = RBF(n_dims=1)
        hsgp = HSGP(
            base_kernel=base_kernel, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        x_test = jnp.linspace(-2.5, 2.5, 30)[:, None]
        pred = posterior.predict(x_test, D)

        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))

    def test_mll_differentiable(self):
        """conjugate_mll with HSGP must be differentiable w.r.t. kernel params."""
        from flax import nnx
        from gpjax.dataset import Dataset
        from gpjax.gps import Prior
        from gpjax.likelihoods import Gaussian
        from gpjax.mean_functions import Zero
        from gpjax.objectives import conjugate_mll
        import jax

        key = jr.key(42)
        n = 50
        x = jnp.linspace(-3, 3, n)[:, None]
        y = jnp.sin(x) + 0.1 * jr.normal(key, (n, 1))
        D = Dataset(X=x, y=y)

        base_kernel = RBF(n_dims=1)
        hsgp = HSGP(
            base_kernel=base_kernel, num_basis_fns=20, domain_half_width=5.0, center=0.0
        )
        prior = Prior(kernel=hsgp, mean_function=Zero())
        likelihood = Gaussian(num_datapoints=n)
        posterior = prior * likelihood

        # Split into graphdef and state
        graphdef, state = nnx.split(posterior)

        def loss(state):
            model = nnx.merge(graphdef, state)
            return -conjugate_mll(model, D)

        grad_fn = jax.grad(loss)
        grads = grad_fn(state)

        # Gradients should be finite
        flat_grads = jax.tree.leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient: {g}"
