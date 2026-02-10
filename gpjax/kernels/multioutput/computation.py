import jax.numpy as jnp
from jaxtyping import Float, Num

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import Dense, Diagonal, Kronecker
from gpjax.linalg.operators import LinearOperator
from gpjax.linalg.utils import psd
from gpjax.typing import Array


class MultiOutputKernelComputation(AbstractKernelComputation):
    """Compute engine for multi-output kernels.

    Iterates over kernel.components — a sequence of (CoregionalizationMatrix,
    kernel) pairs — to build structured covariance matrices.  Single-component
    kernels (ICM) retain Kronecker structure; multi-component kernels (LCM)
    materialise the sum to Dense.
    """

    def gram(self, kernel, x: Num[Array, "N D"]) -> LinearOperator:
        components = kernel.components
        if len(components) == 1:
            cm, k = components[0]
            K_input = k.gram(x)
            B = Dense(cm.B)
            return psd(Kronecker([B, K_input]))
        K = sum(jnp.kron(cm.B, k.gram(x).to_dense()) for cm, k in components)
        return psd(Dense(K))

    def cross_covariance(
        self, kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "..."]:
        """Override to bypass [N, M] return type annotation for multi-output."""
        return self._cross_covariance(kernel, x, y)

    def _cross_covariance(
        self, kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "..."]:
        return sum(
            jnp.kron(cm.B, k.cross_covariance(x, y)) for cm, k in kernel.components
        )

    def diagonal(self, kernel, inputs: Num[Array, "N D"]) -> Diagonal:
        diag_sum = sum(
            jnp.kron(jnp.diag(cm.B), k.diagonal(inputs).diagonal)
            for cm, k in kernel.components
        )
        return psd(Diagonal(diag_sum))
