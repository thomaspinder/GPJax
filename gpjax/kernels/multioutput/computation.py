import jax.numpy as jnp
from jaxtyping import Float, Num

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import Dense, Diagonal, Kronecker
from gpjax.linalg.utils import psd
from gpjax.typing import Array


class MultiOutputKernelComputation(AbstractKernelComputation):
    """Compute engine for multi-output kernels.

    Dispatches on kernel type to build structured covariance matrices.
    Currently supports ICMKernel (Kronecker structure).
    """

    def gram(self, kernel, x: Num[Array, "N D"]) -> Kronecker:
        from gpjax.kernels.multioutput.icm import ICMKernel

        if isinstance(kernel, ICMKernel):
            K_input = kernel.base_kernel.gram(x)
            B = Dense(kernel.coregionalization_matrix.B)
            return psd(Kronecker([B, K_input]))
        raise NotImplementedError(f"No gram implementation for {type(kernel).__name__}")

    def cross_covariance(
        self, kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "..."]:
        """Override to bypass [N, M] return type annotation for multi-output."""
        return self._cross_covariance(kernel, x, y)

    def _cross_covariance(
        self, kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "..."]:
        from gpjax.kernels.multioutput.icm import ICMKernel

        if isinstance(kernel, ICMKernel):
            Kxy = kernel.base_kernel.cross_covariance(x, y)
            B = kernel.coregionalization_matrix.B
            return jnp.kron(B, Kxy)
        raise NotImplementedError(
            f"No cross_covariance implementation for {type(kernel).__name__}"
        )

    def diagonal(self, kernel, inputs: Num[Array, "N D"]) -> Diagonal:
        from gpjax.kernels.multioutput.icm import ICMKernel

        if isinstance(kernel, ICMKernel):
            k_diag = kernel.base_kernel.diagonal(inputs).diagonal
            b_diag = jnp.diag(kernel.coregionalization_matrix.B)
            return psd(Diagonal(jnp.kron(b_diag, k_diag)))
        raise NotImplementedError(
            f"No diagonal implementation for {type(kernel).__name__}"
        )
