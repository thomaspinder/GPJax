# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Int,
    Num,
)

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    EigenKernelComputation,
)
from gpjax.kernels.non_euclidean.utils import (
    calculate_heat_semigroup,
    jax_gather_nd,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import (
    Parameter,
    PositiveReal,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
    ScalarInt,
)


class GraphKernel(StationaryKernel):
    r"""The Matérn graph kernel defined on the vertex set of a graph.

    A Matérn graph kernel defined on the vertices of a graph.

    Computes the covariance for pairs of vertices $(v_i, v_j)$ with variance $\sigma^2$:
    $$
    k(v_i, v_j) = \sigma^2 \exp\Bigg(-\frac{\lVert v_i - v_j \rVert^2_2}{2\ell^2}\Bigg)
    $$
    where $\ell$ is the lengthscale parameter and $\sigma^2$ is the variance.

    The key reference for this object is Borovitskiy et. al., (2020).

    """

    num_vertex: tp.Union[ScalarInt, None]
    laplacian: Float[Array, "N N"]
    eigenvalues: Float[Array, "N 1"]
    eigenvectors: Float[Array, "N N"]
    name: str = "Graph Matérn"

    def __init__(
        self,
        laplacian: Num[Array, "N N"],
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[ScalarFloat, Float[Array, " D"], Parameter] = 1.0,
        variance: tp.Union[ScalarFloat, Parameter] = 1.0,
        smoothness: ScalarFloat = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = EigenKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            laplacian: the Laplacian matrix of the graph.
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            smoothness: the smoothness parameter of the Matérn kernel.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        if isinstance(smoothness, Parameter):
            self.smoothness = smoothness
        else:
            self.smoothness = PositiveReal(smoothness)

        self.laplacian = laplacian
        evals, eigenvectors = jnp.linalg.eigh(self.laplacian)
        self.eigenvectors = eigenvectors
        self.eigenvalues = evals.reshape(-1, 1)
        self.num_vertex = self.eigenvalues.shape[0]

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self,
        x: Int[Array, "N 1"],
        y: Int[Array, "M 1"],
    ):
        S = calculate_heat_semigroup(self)
        Kxx = (jax_gather_nd(self.eigenvectors, x) * S.squeeze()) @ jnp.transpose(
            jax_gather_nd(self.eigenvectors, y)
        )  # shape (n,n)
        return Kxx.squeeze()
