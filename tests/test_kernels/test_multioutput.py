import jax
import jax.numpy as jnp
import pytest
from gpjax.kernels.multioutput.base import MultiOutputKernel
from gpjax.kernels.multioutput.icm import ICMKernel
from gpjax.kernels.stationary import RBF
from gpjax.parameters import CoregionalizationMatrix


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
