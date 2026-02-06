import typing as tp

import jax.numpy as jnp
from jaxtyping import Float

import gpjax
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import (
    Diagonal,
)
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.approximations.RFF")

# TODO: Use low rank linear operator!


class BasisFunctionComputation(AbstractKernelComputation):
    r"""Compute engine class for finite basis function approximations to a kernel."""

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        z1 = self.compute_features(kernel, x)
        z2 = self.compute_features(kernel, y)
        return self.scaling(kernel) * jnp.matmul(z1, z2.T)

    def _gram(self, kernel: K, inputs: Float[Array, "N D"]) -> Float[Array, "N N"]:
        z1 = self.compute_features(kernel, inputs)
        return self.scaling(kernel) * jnp.matmul(z1, z1.T)

    def diagonal(self, kernel: K, inputs: Float[Array, "N D"]) -> Diagonal:
        r"""For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): the kernel function.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns
        -------
            Diagonal: The computed diagonal variance entries.
        """
        return super().diagonal(kernel.base_kernel, inputs)

    def compute_features(
        self, kernel: K, x: Float[Array, "N D"]
    ) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            kernel: the kernel function.
            x: the inputs to the kernel function of shape `(N, D)`.

        Returns:
            A matrix of shape $N \times L$ representing the random fourier features where $L = 2M$.
        """
        frequencies = kernel.frequencies
        scaling_factor = kernel.base_kernel.lengthscale[...]
        z = jnp.matmul(x, (frequencies / scaling_factor).T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z

    def scaling(self, kernel: K) -> Float[Array, ""]:
        r"""Compute the scaling factor for the covariance matrix.

        Args:
            kernel: the kernel function.

        Returns:
            A scalar array representing the scaling factor.
        """
        return kernel.base_kernel.variance[...] / kernel.num_basis_fns
