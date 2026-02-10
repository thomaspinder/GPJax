from gpjax.kernels.multioutput.icm import ICMKernel
from gpjax.kernels.multioutput.lcm import LCMKernel
from gpjax.kernels.stationary import RBF, Matern52
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


class TestLCMKernel:
    def test_num_outputs(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=3, rank=2, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=3, rank=1, key=k2)
        kernel = LCMKernel(
            kernels=[RBF(), Matern52()],
            coregionalization_matrices=[coreg1, coreg2],
        )
        assert kernel.num_outputs == 3

    def test_num_latent_gps(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k2)
        kernel = LCMKernel(
            kernels=[RBF(), Matern52()],
            coregionalization_matrices=[coreg1, coreg2],
        )
        assert kernel.num_latent_gps == 2

    def test_latent_kernels(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        base1, base2 = RBF(), Matern52()
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k2)
        kernel = LCMKernel(
            kernels=[base1, base2],
            coregionalization_matrices=[coreg1, coreg2],
        )
        assert kernel.latent_kernels == (base1, base2)

    def test_is_multi_output_kernel(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k2)
        kernel = LCMKernel(
            kernels=[RBF(), Matern52()],
            coregionalization_matrices=[coreg1, coreg2],
        )
        from gpjax.kernels.multioutput.base import MultiOutputKernel

        assert isinstance(kernel, MultiOutputKernel)

    def test_point_pair_raises(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k2)
        kernel = LCMKernel(
            kernels=[RBF(), Matern52()],
            coregionalization_matrices=[coreg1, coreg2],
        )
        x = jnp.array([1.0])
        with pytest.raises(NotImplementedError, match="point-pair"):
            kernel(x, x)

    def test_mismatched_lengths_raises(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k2)
        with pytest.raises(ValueError, match="same length"):
            LCMKernel(kernels=[RBF()], coregionalization_matrices=[coreg1, coreg2])

    def test_mismatched_num_outputs_raises(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=3, rank=1, key=k2)
        with pytest.raises(ValueError, match="num_outputs"):
            LCMKernel(
                kernels=[RBF(), Matern52()], coregionalization_matrices=[coreg1, coreg2]
            )

    def test_from_icm_components(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        coreg1 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k1)
        coreg2 = CoregionalizationMatrix(num_outputs=2, rank=1, key=k2)
        icm1 = ICMKernel(base_kernel=RBF(), coregionalization_matrix=coreg1)
        icm2 = ICMKernel(base_kernel=Matern52(), coregionalization_matrix=coreg2)
        kernel = LCMKernel.from_icm_components([icm1, icm2])
        assert kernel.num_outputs == 2
        assert kernel.num_latent_gps == 2
        assert kernel.latent_kernels[0] is icm1.base_kernel
        assert kernel.coregionalization_matrices[0] is coreg1


class TestLCMKernelComputation:
    @pytest.fixture
    def lcm_setup(self):
        """LCM with Q=2 components, P=3 outputs, N=10 points."""
        key = jax.random.PRNGKey(0)
        N, D, P = 10, 2, 3
        k1, k2, k3 = jax.random.split(key, 3)
        X = jax.random.normal(k1, (N, D))
        coreg1 = CoregionalizationMatrix(num_outputs=P, rank=2, key=k2)
        coreg2 = CoregionalizationMatrix(num_outputs=P, rank=1, key=k3)
        kernel = LCMKernel(
            kernels=[RBF(n_dims=D), Matern52(n_dims=D)],
            coregionalization_matrices=[coreg1, coreg2],
        )
        return kernel, X, N, P

    @pytest.fixture
    def lcm_single_setup(self):
        """LCM with Q=1 component (should behave like ICM)."""
        key = jax.random.PRNGKey(0)
        N, D, P = 10, 2, 3
        k1, k2 = jax.random.split(key)
        X = jax.random.normal(k1, (N, D))
        coreg = CoregionalizationMatrix(num_outputs=P, rank=2, key=k2)
        kernel = LCMKernel(
            kernels=[RBF(n_dims=D)],
            coregionalization_matrices=[coreg],
        )
        return kernel, X, N, P, coreg

    def test_gram_shape(self, lcm_setup):
        kernel, X, N, P = lcm_setup
        K = kernel.gram(X)
        assert K.shape == (N * P, N * P)

    def test_gram_equals_manual_sum(self, lcm_setup):
        """gram() matches manual Σ_q kron(B_q, K_q)."""
        kernel, X, _N, _P = lcm_setup
        expected = sum(
            jnp.kron(cm.B, k.gram(X).to_dense())
            for cm, k in zip(
                kernel.coregionalization_matrices, kernel.latent_kernels, strict=True
            )
        )
        actual = kernel.gram(X).to_dense()
        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_gram_psd(self, lcm_setup):
        kernel, X, _N, _P = lcm_setup
        K = kernel.gram(X).to_dense()
        eigvals = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigvals >= -1e-6)

    def test_gram_q1_is_kronecker(self, lcm_single_setup):
        """Q=1 LCM returns Kronecker operator (ICM efficiency)."""
        from gpjax.linalg import Kronecker

        kernel, X, _N, _P, _coreg = lcm_single_setup
        K = kernel.gram(X)
        assert isinstance(K, Kronecker)

    def test_gram_q2_is_dense(self, lcm_setup):
        """Q>1 LCM returns Dense operator."""
        from gpjax.linalg import Dense

        kernel, X, _N, _P = lcm_setup
        K = kernel.gram(X)
        assert isinstance(K, Dense)

    def test_cross_covariance_shape(self, lcm_setup):
        kernel, X, N, P = lcm_setup
        M = 5
        Y = jax.random.normal(jax.random.PRNGKey(1), (M, X.shape[1]))
        Kxy = kernel.cross_covariance(X, Y)
        assert Kxy.shape == (N * P, M * P)

    def test_cross_covariance_equals_manual(self, lcm_setup):
        kernel, X, _N, _P = lcm_setup
        M = 5
        Y = jax.random.normal(jax.random.PRNGKey(1), (M, X.shape[1]))
        expected = sum(
            jnp.kron(cm.B, k.cross_covariance(X, Y))
            for cm, k in zip(
                kernel.coregionalization_matrices, kernel.latent_kernels, strict=True
            )
        )
        actual = kernel.cross_covariance(X, Y)
        assert jnp.allclose(actual, expected, atol=1e-6)

    def test_diagonal_shape(self, lcm_setup):
        kernel, X, N, P = lcm_setup
        diag_op = kernel.diagonal(X)
        assert diag_op.shape == (N * P, N * P)

    def test_diagonal_matches_gram_diagonal(self, lcm_setup):
        kernel, X, _N, _P = lcm_setup
        gram_diag = jnp.diag(kernel.gram(X).to_dense())
        diag_op = kernel.diagonal(X)
        assert jnp.allclose(diag_op.diagonal, gram_diag, atol=1e-6)


def test_public_imports():
    """Multi-output classes are importable from gpjax and gpjax.kernels."""
    import gpjax as gpx

    assert hasattr(gpx.kernels, "ICMKernel")
    assert hasattr(gpx.kernels, "LCMKernel")
    assert hasattr(gpx.kernels, "MultiOutputKernel")
    assert hasattr(gpx.kernels, "MultiOutputKernelComputation")
    assert hasattr(gpx.parameters, "CoregionalizationMatrix")
    assert hasattr(gpx.likelihoods, "MultiOutputGaussian")
