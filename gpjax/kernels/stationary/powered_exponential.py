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
from flax import nnx
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import euclidean_distance
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class PoweredExponential(StationaryKernel):
    r"""The powered exponential family of kernels.

    Computes the covariance for pairs of inputs $(x, y)$ with length-scale parameter
    $\ell$, $\sigma$ and power $\kappa$.
    $$
    k(x, y)=\sigma^2\exp\Bigg(-\Big(\frac{\lVert x-y\rVert^2}{\ell^2}\Big)^\kappa\Bigg)
    $$

    This also equivalent to the symmetric generalized normal distribution.
    See Diggle and Ribeiro (2007) - "Model-based Geostatistics".
    and
    https://en.wikipedia.org/wiki/Generalized_normal_distribution#Symmetric_version
    """

    name: str = "Powered Exponential"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]] = 1.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        power: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: the indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            power: the power of the kernel κ.
            n_dims: the number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: the computation engine that the kernel uses to compute the
                covariance matrix.
        """
        self.power = power

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        power_val = self.power.value if hasattr(self.power, "value") else self.power
        K = self.variance.value * jnp.exp(-(euclidean_distance(x, y) ** power_val))
        return K.squeeze()
