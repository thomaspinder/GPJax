from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.multioutput.base import MultiOutputKernel
from gpjax.kernels.multioutput.computation import MultiOutputKernelComputation
from gpjax.parameters import CoregionalizationMatrix


class ICMKernel(MultiOutputKernel):
    """Intrinsic Coregionalization Model kernel.

    Combines a single shared input kernel with a coregionalization matrix
    to produce a Kronecker-structured covariance: K = B (x) K_input.

    Args:
        base_kernel: The shared input-space kernel.
        coregionalization_matrix: The output-space coregionalization.
    """

    def __init__(
        self,
        base_kernel: AbstractKernel,
        coregionalization_matrix: CoregionalizationMatrix,
    ):
        super().__init__(compute_engine=MultiOutputKernelComputation())
        self.base_kernel = base_kernel
        self.coregionalization_matrix = coregionalization_matrix

    @property
    def num_outputs(self) -> int:
        return self.coregionalization_matrix.num_outputs

    @property
    def num_latent_gps(self) -> int:
        return 1

    @property
    def latent_kernels(self) -> tuple[AbstractKernel, ...]:
        return (self.base_kernel,)

    @property
    def components(self):
        return ((self.coregionalization_matrix, self.base_kernel),)
