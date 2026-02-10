import jax.numpy as jnp
from jaxtyping import Float, Num

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import Dense, Diagonal, Kronecker
from gpjax.linalg.operators import LinearOperator
from gpjax.linalg.utils import psd
from gpjax.typing import Array


class MultiOutputKernelComputation(AbstractKernelComputation):
    """Compute engine for multi-output kernels.

    Dispatches on kernel type to build structured covariance matrices.
    Supports ICMKernel (Kronecker) and LCMKernel (Kronecker for Q=1, Dense for Q>1).
    """

    def gram(self, kernel, x: Num[Array, "N D"]) -> LinearOperator:
        from gpjax.kernels.multioutput.icm import ICMKernel
        from gpjax.kernels.multioutput.lcm import LCMKernel

        if isinstance(kernel, ICMKernel):
            K_input = kernel.base_kernel.gram(x)
            B = Dense(kernel.coregionalization_matrix.B)
            return psd(Kronecker([B, K_input]))
        if isinstance(kernel, LCMKernel):
            if kernel.num_latent_gps == 1:
                K_input = kernel.latent_kernels[0].gram(x)
                B = Dense(kernel.coregionalization_matrices[0].B)
                return psd(Kronecker([B, K_input]))
            K = sum(
                jnp.kron(cm.B, k.gram(x).to_dense())
                for cm, k in zip(
                    kernel.coregionalization_matrices, kernel.latent_kernels
                )
            )
            return psd(Dense(K))
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
        from gpjax.kernels.multioutput.lcm import LCMKernel

        if isinstance(kernel, ICMKernel):
            Kxy = kernel.base_kernel.cross_covariance(x, y)
            B = kernel.coregionalization_matrix.B
            return jnp.kron(B, Kxy)
        if isinstance(kernel, LCMKernel):
            return sum(
                jnp.kron(cm.B, k.cross_covariance(x, y))
                for cm, k in zip(
                    kernel.coregionalization_matrices, kernel.latent_kernels
                )
            )
        raise NotImplementedError(
            f"No cross_covariance implementation for {type(kernel).__name__}"
        )

    def diagonal(self, kernel, inputs: Num[Array, "N D"]) -> Diagonal:
        from gpjax.kernels.multioutput.icm import ICMKernel
        from gpjax.kernels.multioutput.lcm import LCMKernel

        if isinstance(kernel, ICMKernel):
            k_diag = kernel.base_kernel.diagonal(inputs).diagonal
            b_diag = jnp.diag(kernel.coregionalization_matrix.B)
            return psd(Diagonal(jnp.kron(b_diag, k_diag)))
        if isinstance(kernel, LCMKernel):
            diag_sum = sum(
                jnp.kron(jnp.diag(cm.B), k.diagonal(inputs).diagonal)
                for cm, k in zip(
                    kernel.coregionalization_matrices, kernel.latent_kernels
                )
            )
            return psd(Diagonal(diag_sum))
        raise NotImplementedError(
            f"No diagonal implementation for {type(kernel).__name__}"
        )
