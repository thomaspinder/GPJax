from gpjax.kernels.computations.base import AbstractKernelComputation


class MultiOutputKernelComputation(AbstractKernelComputation):
    """Compute engine for multi-output kernels.

    Placeholder — gram/cross_covariance/diagonal implemented in Task 4.
    """

    def _cross_covariance(self, kernel, x, y):
        raise NotImplementedError("Implemented in Task 4.")
