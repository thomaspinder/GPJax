from abc import abstractmethod

from gpjax.kernels.base import AbstractKernel


class MultiOutputKernel(AbstractKernel):
    """Base class for multi-output kernels.

    Multi-output kernels produce structured covariance matrices (Kronecker,
    BlockDiag, etc.) over the joint input-output space. They do not support
    point-pair evaluation via __call__; all computation goes through the
    compute engine's gram() and cross_covariance() methods.
    """

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """Number of output dimensions."""
        ...

    @property
    @abstractmethod
    def num_latent_gps(self) -> int:
        """Number of latent GP functions."""
        ...

    @property
    @abstractmethod
    def latent_kernels(self) -> tuple[AbstractKernel, ...]:
        """Tuple of latent kernels."""
        ...

    def __call__(self, x, y):
        raise NotImplementedError(
            "Multi-output kernels do not support point-pair evaluation. "
            "Use kernel.gram(x) or kernel.cross_covariance(x, y) instead."
        )
