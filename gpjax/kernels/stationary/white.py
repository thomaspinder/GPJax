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
from typing import ClassVar

import jax.numpy as jnp
from jaxtyping import Float
from paramax import AbstractUnwrappable

from gpjax.kernels.base import AbstractKernel, _val
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    ConstantDiagonalKernelComputation,
)
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import NonNegativeReal
from gpjax.typing import (
    Array,
    ScalarFloat,
)


class White(StationaryKernel):
    r"""The White noise kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with variance $\sigma^2$:
    $$
    k(x, y) = \sigma^2 \delta(x-y)
    $$
    """

    name: ClassVar[str] = "White"
    # White noise has no lengthscale: __call__ never reads it, so it is
    # overridden to a ClassVar here rather than inherited as a real field,
    # which would otherwise manufacture a phantom, trainable pytree leaf
    # (see issue #695).
    lengthscale: ClassVar[None] = None

    def __init__(
        self,
        active_dims: list[int] | slice | None = None,
        variance: ScalarFloat | AbstractUnwrappable = 1.0,
        n_dims: int | None = None,
        compute_engine: AbstractKernelComputation = ConstantDiagonalKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            variance: the variance of the kernel σ.
            n_dims: The number of input dimensions.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix
        """
        # Bypass StationaryKernel.__init__ (which would set a lengthscale)
        # and go straight to AbstractKernel.__init__.
        AbstractKernel.__init__(
            self,
            active_dims=active_dims,
            n_dims=n_dims,
            compute_engine=compute_engine,
        )
        if isinstance(variance, AbstractUnwrappable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        K = jnp.all(jnp.equal(x, y)) * _val(self.variance)
        return K.squeeze()
