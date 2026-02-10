from flax import nnx

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.multioutput.base import MultiOutputKernel
from gpjax.kernels.multioutput.computation import MultiOutputKernelComputation
from gpjax.parameters import CoregionalizationMatrix


class LCMKernel(MultiOutputKernel):
    """Linear Model of Coregionalization kernel.

    Generalises ICM by combining Q independent latent kernels, each with its
    own coregionalization matrix:  K = Σ_q B^(q) ⊗ k_q(X, X).

    When Q=1 this is equivalent to an ICM and retains Kronecker structure.
    When Q>1 the sum is materialised to a dense matrix.

    Args:
        kernels: List of Q latent kernels.
        coregionalization_matrices: List of Q coregionalization matrices.
            All must have the same num_outputs.
    """

    def __init__(
        self,
        kernels: list[AbstractKernel],
        coregionalization_matrices: list[CoregionalizationMatrix],
    ):
        if len(kernels) != len(coregionalization_matrices):
            raise ValueError(
                f"kernels and coregionalization_matrices must have the same length, "
                f"got {len(kernels)} and {len(coregionalization_matrices)}."
            )
        num_outputs_set = {cm.num_outputs for cm in coregionalization_matrices}
        if len(num_outputs_set) != 1:
            raise ValueError(
                f"All coregionalization matrices must have the same num_outputs, "
                f"got {num_outputs_set}."
            )
        super().__init__(compute_engine=MultiOutputKernelComputation())
        self.base_kernels = nnx.List(kernels)
        self.coregionalization_matrices = nnx.List(coregionalization_matrices)

    @property
    def num_outputs(self) -> int:
        return self.coregionalization_matrices[0].num_outputs

    @property
    def num_latent_gps(self) -> int:
        return len(self.base_kernels)

    @property
    def latent_kernels(self) -> tuple[AbstractKernel, ...]:
        return tuple(self.base_kernels)

    @property
    def components(self):
        return tuple(
            zip(self.coregionalization_matrices, self.base_kernels, strict=True)
        )

    @classmethod
    def from_icm_components(cls, icm_kernels: list) -> "LCMKernel":
        """Build an LCM from a list of ICMKernel instances.

        Args:
            icm_kernels: List of ICMKernel objects. Each contributes its
                base_kernel and coregionalization_matrix as one LCM component.

        Returns:
            An LCMKernel combining all components.
        """
        kernels = [icm.base_kernel for icm in icm_kernels]
        matrices = [icm.coregionalization_matrix for icm in icm_kernels]
        return cls(kernels=kernels, coregionalization_matrices=matrices)
