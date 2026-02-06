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

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import (
    NonNegativeReal,
)
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

WeightVariance = tp.Union[Float[Array, "D"], ScalarArray]
WeightVarianceCompatible = tp.Union[ScalarFloat, list[float], WeightVariance]


class ArcCosine(AbstractKernel):
    r"""The ArCosine kernel.

    This kernel is non-stationary and resembles the behavior of neural networks.
    See Section 3.1 of
    [Cho and Saul (2011)](https://arxiv.org/abs/1112.3712) for
    additional details.
    """

    variance: nnx.Variable[ScalarArray]
    weight_variance: nnx.Variable[WeightVariance]
    bias_variance: nnx.Variable[ScalarArray]
    name = "ArcCosine"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        order: tp.Literal[0, 1, 2] = 0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        weight_variance: tp.Union[
            WeightVarianceCompatible, nnx.Variable[WeightVariance]
        ] = 1.0,
        bias_variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            order: The order of the kernel. Must be 0, 1 or 2.
            variance: The variance of the kernel σ.
            weight_variance: The weight variance of the kernel.
            bias_variance: The bias variance of the kernel.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        if order not in [0, 1, 2]:
            raise ValueError("ArcCosine kernel only implemented for orders 0, 1 and 2.")

        self.order = order

        self.weight_variance = weight_variance

        if isinstance(variance, nnx.Variable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)
        self.bias_variance = bias_variance

        self.name = f"ArcCosine (order {self.order})"

        super().__init__(active_dims, n_dims, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarArray:
        x = self.slice_input(x)
        y = self.slice_input(y)

        x_x = self._weighted_prod(x, x)
        x_y = self._weighted_prod(x, y)
        y_y = self._weighted_prod(y, y)

        cos_theta = x_y / jnp.sqrt(x_x * y_y)
        jitter = 1e-15  # improve numerical stability
        theta = jnp.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        K = self._J(theta)
        K *= jnp.sqrt(x_x) ** self.order
        K *= jnp.sqrt(y_y) ** self.order
        K *= self.variance[...] / jnp.pi

        return K.squeeze()

    def _weighted_prod(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> ScalarFloat:
        r"""Calculate the weighted product between two arguments.

        Args:
            x (Float[Array, "D"]): The left hand argument.
            y (Float[Array, "D"]): The right hand argument.
        Returns:
            ScalarFloat: The value of the weighted product between the two arguments``.
        """
        weight_var = (
            self.weight_variance[...]
            if isinstance(self.weight_variance, nnx.Variable)
            else self.weight_variance
        )
        bias_var = (
            self.bias_variance[...]
            if isinstance(self.bias_variance, nnx.Variable)
            else self.bias_variance
        )
        return jnp.inner(weight_var * x, y) + bias_var

    def _J(self, theta: ScalarFloat) -> ScalarFloat:
        r"""Evaluate the angular dependency function corresponding to the desired order.

        Args:
            theta (Float[Array, "1"]): The weighted angle between inputs.

        Returns:
            Float[Array, "1"]: The value of the angular dependency function`.
        """

        if self.order == 0:
            return jnp.pi - theta
        elif self.order == 1:
            return jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)
        else:
            return 3.0 * jnp.sin(theta) * jnp.cos(theta) + (jnp.pi - theta) * (
                1.0 + 2.0 * jnp.cos(theta) ** 2
            )
