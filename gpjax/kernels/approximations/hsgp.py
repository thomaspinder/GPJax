"""Hilbert Space Gaussian Process (HSGP) kernel approximation.

Reference:
    Solin & Sarkka (2019). "Hilbert Space Methods for Reduced-Rank
    Gaussian Process Regression." Statistics and Computing.
"""

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations.hsgp import HSGPComputation
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.typing import Array


class HSGP(AbstractKernel):
    r"""Hilbert Space Gaussian Process approximation (1D).

    Approximates a stationary kernel by projecting onto the eigenbasis of the
    Laplacian operator with Dirichlet boundary conditions on :math:`[-L, L]`.
    The approximate covariance function is:

    .. math::
        \tilde{k}(x, x') = \sum_{j=1}^{m} S(\sqrt{\lambda_j})\,
        \phi_j(x)\,\phi_j(x')

    where :math:`\lambda_j = (j\pi / 2L)^2` are the eigenvalues,
    :math:`\phi_j(x) = L^{-1/2}\sin(j\pi(x + L) / 2L)` are the
    eigenfunctions, and :math:`S` is the spectral density of the base
    kernel.

    The linearised form decomposes the GP function as:

    .. math::
        f(x) \approx \Phi(x)\,\mathrm{diag}(\sqrt{S})\,\beta,
        \quad \beta \sim \mathcal{N}(0, I_m)

    which is a Bayesian linear model with :math:`m` basis functions.

    Args:
        base_kernel: Stationary kernel to approximate.  Must provide a
            ``spectral_density`` property returning a
            :class:`~gpjax.kernels.stationary.utils.SpectralDensity`.
        num_basis_fns: Number of basis functions :math:`m`.
        domain_half_width: Half-width :math:`L` of the approximation domain.
            Inputs should lie well inside :math:`[-L, L]` (after centering).
        center: Center of the data domain.  If ``None``, it is set
            automatically on the first call to :meth:`eigenfunctions` as the
            midpoint of the observed input range.
        compute_engine: Computation engine (default
            :class:`~gpjax.kernels.computations.hsgp.HSGPComputation`).

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>> X = jnp.linspace(-1, 1, 50)[:, None]
        >>> base = gpx.kernels.Matern52(n_dims=1)
        >>> hsgp = gpx.kernels.HSGP(base, num_basis_fns=20, domain_half_width=5.0)
        >>> K = hsgp.gram(X)  # approximate Gram matrix
    """

    compute_engine: HSGPComputation

    def __init__(
        self,
        base_kernel: StationaryKernel,
        num_basis_fns: int,
        domain_half_width: float,
        center: tp.Union[float, None] = None,
        compute_engine: HSGPComputation = HSGPComputation(),
    ):
        if not isinstance(base_kernel, StationaryKernel):
            raise TypeError(
                "HSGP can only approximate stationary kernels. "
                f"Got {type(base_kernel).__name__}."
            )
        _ = base_kernel.spectral_density

        super().__init__(
            active_dims=base_kernel.active_dims,
            n_dims=base_kernel.n_dims,
            compute_engine=compute_engine,
        )
        self.base_kernel = base_kernel
        self.num_basis_fns = num_basis_fns
        self.domain_half_width = domain_half_width
        self._center = center
        self.name = f"{self.base_kernel.name} (HSGP)"

    def eigenvalues(self) -> Float[Array, " m"]:
        r"""Square roots of the Laplacian eigenvalues.

        .. math::
            \sqrt{\lambda_j} = \frac{j\pi}{2L}, \quad j = 1, \ldots, m

        Returns:
            Array of shape ``(m,)``.
        """
        indices = jnp.arange(1, self.num_basis_fns + 1)
        return indices * jnp.pi / (2.0 * self.domain_half_width)

    def eigenfunctions(self, x: Float[Array, "N 1"]) -> Float[Array, "N m"]:
        r"""Laplacian eigenfunctions evaluated at *x*.

        .. math::
            \phi_j(x) = \frac{1}{\sqrt{L}}
            \sin\!\Bigl(\frac{j\pi\,(x + L)}{2L}\Bigr)

        where *x* is first shifted by the stored center.

        Args:
            x: Input locations of shape ``(N, 1)``.

        Returns:
            Basis matrix :math:`\Phi` of shape ``(N, m)``.
        """
        if self._center is None:
            self._center = float((x.max() + x.min()) / 2.0)

        half_width = self.domain_half_width
        x_centered = x - self._center
        sqrt_eigenvalues = self.eigenvalues()
        return jnp.sin((x_centered + half_width) * sqrt_eigenvalues) / jnp.sqrt(
            half_width
        )

    def spectral_weights(self) -> Float[Array, " m"]:
        r"""Spectral density evaluated at the eigenvalue square roots.

        .. math::
            S(\sqrt{\lambda_j})

        Returns:
            Array of shape ``(m,)`` with the spectral weights.
        """
        omega = self.eigenvalues()
        return self.base_kernel.spectral_density(
            omega,
            self.base_kernel.variance[...],
            self.base_kernel.lengthscale[...],
        )

    def compute_basis(
        self, x: Float[Array, "N 1"]
    ) -> tuple[Float[Array, "N m"], Float[Array, " m"]]:
        r"""Linearised HSGP decomposition.

        Returns :math:`(\Phi, \sqrt{S})` such that

        .. math::
            f(x) \approx \Phi\,\mathrm{diag}(\sqrt{S})\,\beta,
            \quad \beta \sim \mathcal{N}(0, I_m).

        Args:
            x: Input locations of shape ``(N, 1)``.

        Returns:
            Tuple ``(phi, sqrt_spectral_weights)`` where ``phi`` has shape
            ``(N, m)`` and ``sqrt_spectral_weights`` has shape ``(m,)``.
        """
        phi = self.eigenfunctions(x)
        sqrt_spectral_weights = jnp.sqrt(self.spectral_weights())
        return phi, sqrt_spectral_weights

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> None:
        raise RuntimeError(
            "HSGP does not support pointwise kernel evaluation. "
            "Use gram(), cross_covariance(), or compute_basis() instead."
        )
