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
    PositiveReal,
)
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)


class Polynomial(AbstractKernel):
    r"""The Polynomial kernel with variable degree.

    Computes the covariance for pairs of inputs $(x, y)$ with variance $\sigma^2$:
    $$
    k(x, y) = (\alpha + \sigma^2 x y)^d
    $$
    where $\sigma^\in \mathbb{R}_{>0}$ is the kernel's variance parameter, shift
    parameter $\alpha$ and integer degree $d$.
    """

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        degree: int = 2,
        shift: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            degree: The degree of the polynomial.
            shift: The shift parameter of the kernel.
            variance: The variance of the kernel.
            n_dims: The number of input dimensions.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        super().__init__(active_dims, n_dims, compute_engine)

        self.degree = degree

        self.shift = shift
        if tp.TYPE_CHECKING and not isinstance(shift, nnx.Variable):
            self.shift = tp.cast(PositiveReal[ScalarArray], self.shift)

        if isinstance(variance, nnx.Variable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)
            if tp.TYPE_CHECKING:
                self.variance = tp.cast(NonNegativeReal[ScalarArray], self.variance)

        self.name = f"Polynomial (degree {self.degree})"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        shift_val = (
            self.shift[...] if isinstance(self.shift, nnx.Variable) else self.shift
        )
        variance_val = (
            self.variance[...]
            if isinstance(self.variance, nnx.Variable)
            else self.variance
        )
        K = jnp.power(shift_val + variance_val * jnp.dot(x, y), self.degree)
        return K.squeeze()
