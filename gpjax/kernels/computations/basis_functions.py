import typing as tp

import jax.numpy as jnp
from jaxtyping import Float

import gpjax
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import (
    Diagonal,
    LowRank,
)
from gpjax.linalg.utils import psd
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.approximations.RFF")


class BasisFunctionComputation(AbstractKernelComputation):
    r"""Compute engine for finite basis function approximations (RFF)."""

    def gram(self, kernel: K, x: Float[Array, "N D"]) -> LowRank:
        features = self.compute_features(kernel, x)
        weighted_features = features * jnp.sqrt(self.scaling(kernel))
        return psd(LowRank(weighted_features))

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        features_x = self.compute_features(kernel, x)
        features_y = self.compute_features(kernel, y)
        return self.scaling(kernel) * jnp.matmul(features_x, features_y.T)

    def _gram(self, kernel: K, inputs: Float[Array, "N D"]) -> Float[Array, "N N"]:
        features = self.compute_features(kernel, inputs)
        return self.scaling(kernel) * jnp.matmul(features, features.T)

    def diagonal(self, kernel: K, inputs: Float[Array, "N D"]) -> Diagonal:
        r"""Diagonal of the approximate Gram matrix.

        Args:
            kernel: The RFF kernel.
            inputs: Input matrix of shape ``(N, D)``.

        Returns:
            Diagonal variance entries.
        """
        return super().diagonal(kernel.base_kernel, inputs)

    def compute_features(
        self, kernel: K, x: Float[Array, "N D"]
    ) -> Float[Array, "N L"]:
        r"""Compute random Fourier features :math:`[\cos(z), \sin(z)]`.

        Args:
            kernel: The RFF kernel.
            x: Inputs of shape ``(N, D)``.

        Returns:
            Feature matrix of shape ``(N, 2M)`` where ``M = num_basis_fns``.
        """
        frequencies = kernel.frequencies
        lengthscale = kernel.base_kernel.lengthscale[...]
        projected = jnp.matmul(x, (frequencies / lengthscale).T)
        return jnp.concatenate([jnp.cos(projected), jnp.sin(projected)], axis=-1)

    def scaling(self, kernel: K) -> Float[Array, ""]:
        r"""Variance scaling factor: :math:`\sigma^2 / M`.

        Args:
            kernel: The RFF kernel.

        Returns:
            Scalar scaling factor.
        """
        return kernel.base_kernel.variance[...] / kernel.num_basis_fns
