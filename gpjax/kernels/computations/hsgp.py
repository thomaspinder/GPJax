"""Compute engine for the Hilbert Space GP approximation."""

import typing as tp

import jax.numpy as jnp
from jaxtyping import Float

import gpjax
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import Diagonal, LowRank
from gpjax.linalg.utils import psd
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.approximations.HSGP")


class HSGPComputation(AbstractKernelComputation):
    r"""Compute engine for the HSGP kernel approximation.

    Builds the Gram matrix from the low-rank decomposition
    :math:`\tilde{K} = \Phi \Lambda \Phi^\top` where
    :math:`\Phi` contains Laplacian eigenfunctions and
    :math:`\Lambda = \mathrm{diag}(S(\sqrt{\lambda_1}), \ldots,
    S(\sqrt{\lambda_m}))`.
    """

    def _weighted_basis(
        self, kernel: K, x: Float[Array, "N D"]
    ) -> tuple[Float[Array, "N m"], Float[Array, "N m"]]:
        r"""Return ``(phi, weighted_phi)`` where ``weighted_phi = phi * sqrt(S)``."""
        phi, sqrt_spectral_weights = kernel.compute_basis(x)
        weighted_phi = phi * sqrt_spectral_weights[None, :]
        return phi, weighted_phi

    def gram(self, kernel: K, x: Float[Array, "N D"]) -> LowRank:
        _, weighted_phi = self._weighted_basis(kernel, x)
        return psd(LowRank(weighted_phi))

    def _gram(self, kernel: K, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        _, weighted_phi = self._weighted_basis(kernel, x)
        return weighted_phi @ weighted_phi.T

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        _, weighted_phi_x = self._weighted_basis(kernel, x)
        _, weighted_phi_y = self._weighted_basis(kernel, y)
        return weighted_phi_x @ weighted_phi_y.T

    def diagonal(self, kernel: K, x: Float[Array, "N D"]) -> Diagonal:
        _, weighted_phi = self._weighted_basis(kernel, x)
        return psd(Diagonal(jnp.sum(weighted_phi**2, axis=1)))
