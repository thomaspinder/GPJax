"""Compute engine for the Hilbert Space GP approximation."""

import typing as tp

import jax.numpy as jnp
from jaxtyping import Float

import gpjax
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import Dense, Diagonal
from gpjax.linalg.utils import psd
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.approximations.HSGP")


class HSGPComputation(AbstractKernelComputation):
    r"""Compute engine for the HSGP kernel approximation.

    Computes the Gram matrix via the low-rank decomposition:

    .. math::
        \tilde{K} = \Phi \Lambda \Phi^\top

    where :math:`\Phi` is the matrix of Laplacian eigenfunctions and
    :math:`\Lambda = \mathrm{diag}(S(\sqrt{\lambda_1}), \ldots, S(\sqrt{\lambda_m}))`
    contains the spectral density evaluated at the eigenvalue square roots.
    """

    def gram(self, kernel: K, x: Float[Array, "N D"]) -> Dense:
        Kxx = self._gram(kernel, x)
        return psd(Dense(Kxx))

    def _gram(self, kernel: K, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        phi, sqrt_psd = kernel.compute_basis(x)
        weighted = phi * sqrt_psd[None, :]
        return weighted @ weighted.T

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        phi_x, sqrt_psd = kernel.compute_basis(x)
        phi_y, _ = kernel.compute_basis(y)
        return (phi_x * sqrt_psd[None, :]) @ (phi_y * sqrt_psd[None, :]).T

    def diagonal(self, kernel: K, x: Float[Array, "N D"]) -> Diagonal:
        phi, sqrt_psd = kernel.compute_basis(x)
        weighted = phi * sqrt_psd[None, :]
        return psd(Diagonal(jnp.sum(weighted**2, axis=1)))
