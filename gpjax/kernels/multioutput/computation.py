import jax.numpy as jnp
from jaxtyping import Float, Num
import lineax as lx

from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg.custom_operators import Kronecker
from gpjax.typing import Array


class MultiOutputKernelComputation(AbstractKernelComputation):
    """Compute engine for multi-output kernels.

    Iterates over kernel.components — a sequence of (CoregionalizationMatrix,
    kernel) pairs — to build structured covariance matrices.  Single-component
    kernels (ICM) retain Kronecker structure; multi-component kernels (LCM)
    materialise the sum to Dense.
    """

    def gram(self, kernel, x: Num[Array, "N D"]) -> lx.AbstractLinearOperator:
        components = kernel.components
        if len(components) == 1:
            cm, k = components[0]
            K_input = k.gram(x)
            B = lx.MatrixLinearOperator(cm.B)
            return lx.TaggedLinearOperator(
                Kronecker(A=B, B=K_input), lx.positive_semidefinite_tag
            )
        K = sum(jnp.kron(cm.B, k.gram(x).as_matrix()) for cm, k in components)
        return lx.TaggedLinearOperator(
            lx.MatrixLinearOperator(K), lx.positive_semidefinite_tag
        )

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

    def diagonal(self, kernel, inputs: Num[Array, "N D"]) -> lx.AbstractLinearOperator:
        diag_sum = sum(
            jnp.kron(jnp.diag(cm.B), lx.diagonal(k.diagonal(inputs)))
            for cm, k in kernel.components
        )
        return lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(diag_sum), lx.positive_semidefinite_tag
        )
