from gpjax.kernels.multioutput.icm import ICMKernel
from gpjax.kernels.stationary import RBF
from gpjax.parameters import CoregionalizationMatrix
import jax
import jax.numpy as jnp
import pytest


class TestMultiOutputKernel:
    def test_point_pair_raises(self):
        """MultiOutputKernel.__call__ raises NotImplementedError."""
        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=2, rank=1, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        x = jnp.array([1.0])
        with pytest.raises(NotImplementedError, match="point-pair"):
            kernel(x, x)


class TestICMKernel:
    def test_num_outputs(self):
        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=2, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        assert kernel.num_outputs == 3

    def test_num_latent_gps(self):
        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=3, rank=2, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        assert kernel.num_latent_gps == 1

    def test_latent_kernels(self):
        key = jax.random.PRNGKey(0)
        base = RBF()
        coreg = CoregionalizationMatrix(num_outputs=3, rank=2, key=key)
        kernel = ICMKernel(base_kernel=base, coregionalization_matrix=coreg)
        assert kernel.latent_kernels == (base,)

    def test_is_abstract_kernel(self):
        key = jax.random.PRNGKey(0)
        coreg = CoregionalizationMatrix(num_outputs=2, rank=1, key=key)
        kernel = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg)
        from gpjax.kernels.base import AbstractKernel

        assert isinstance(kernel, AbstractKernel)


class TestMultiOutputKernelComputation:
    @pytest.fixture
    def icm_setup(self):
        key = jax.random.PRNGKey(0)
        N, D, P = 10, 2, 3
        X = jax.random.normal(key, (N, D))
        coreg = CoregionalizationMatrix(num_outputs=P, rank=2, key=key)
        base = RBF(n_dims=D)
        kernel = ICMKernel(base_kernel=base, coregionalization_matrix=coreg)
        return kernel, X, N, P

    def test_gram_shape(self, icm_setup):
        """gram() returns [NP, NP] operator."""
        kernel, X, N, P = icm_setup
        K = kernel.gram(X)
        assert K.shape == (N * P, N * P)

    def test_gram_is_kronecker(self, icm_setup):
        """gram() returns a Kronecker operator."""
        kernel, X, _N, _P = icm_setup
        from gpjax.linalg import Kronecker

        K = kernel.gram(X)
        assert isinstance(K, Kronecker)

    def test_gram_equals_manual_kronecker(self, icm_setup):
        """gram() matches manual kron(B, K_input)."""
        kernel, X, _N, _P = icm_setup
        K_input = kernel.base_kernel.gram(X).to_dense()
        B = kernel.coregionalization_matrix.B
        expected = jnp.kron(B, K_input)
        actual = kernel.gram(X).to_dense()
        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_gram_psd(self, icm_setup):
        """gram() is positive semi-definite."""
        kernel, X, _N, _P = icm_setup
        K = kernel.gram(X).to_dense()
        eigvals = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigvals >= -1e-6)

    def test_cross_covariance_shape(self, icm_setup):
        """cross_covariance() returns [NP, MP]."""
        kernel, X, N, P = icm_setup
        key = jax.random.PRNGKey(1)
        M = 5
        Y = jax.random.normal(key, (M, X.shape[1]))
        Kxy = kernel.cross_covariance(X, Y)
        assert Kxy.shape == (N * P, M * P)

    def test_cross_covariance_equals_manual(self, icm_setup):
        """cross_covariance() matches manual kron(B, K_xy)."""
        kernel, X, _N, _P = icm_setup
        key = jax.random.PRNGKey(1)
        M = 5
        Y = jax.random.normal(key, (M, X.shape[1]))
        K_xy = kernel.base_kernel.cross_covariance(X, Y)
        B = kernel.coregionalization_matrix.B
        expected = jnp.kron(B, K_xy)
        actual = kernel.cross_covariance(X, Y)
        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_diagonal_shape(self, icm_setup):
        """diagonal() returns Diagonal operator with NP entries."""
        kernel, X, N, P = icm_setup
        diag_op = kernel.diagonal(X)
        assert diag_op.shape == (N * P, N * P)

    def test_diagonal_matches_gram_diagonal(self, icm_setup):
        """diagonal() matches the diagonal of the full gram matrix."""
        kernel, X, _N, _P = icm_setup
        gram_diag = jnp.diag(kernel.gram(X).to_dense())
        diag_op = kernel.diagonal(X)
        assert jnp.allclose(diag_op.diagonal, gram_diag, atol=1e-6)


def test_public_imports():
    """Multi-output classes are importable from gpjax and gpjax.kernels."""
    import gpjax as gpx

    assert hasattr(gpx.kernels, "ICMKernel")
    assert hasattr(gpx.kernels, "MultiOutputKernel")
    assert hasattr(gpx.kernels, "MultiOutputKernelComputation")
    assert hasattr(gpx.parameters, "CoregionalizationMatrix")
    assert hasattr(gpx.likelihoods, "MultiOutputGaussian")
