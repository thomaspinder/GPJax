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

import typing as tp

from jax import vmap
import jax.numpy as jnp
from jaxtyping import Float

import gpjax
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.linalg import (
    Diagonal,
    psd,
)
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821
ConstantDiagonalType = Diagonal


class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    r"""Computation engine for constant diagonal kernels."""

    def gram(self, kernel: K, x: Float[Array, "N D"]) -> Diagonal:
        value = kernel(x[0], x[0])
        # Create a diagonal matrix with constant values
        diag = jnp.full(x.shape[0], value)
        return psd(Diagonal(diag))

    def _diagonal(self, kernel: K, inputs: Float[Array, "N D"]) -> Diagonal:
        diag = vmap(lambda x: kernel(x, x))(inputs)
        return psd(Diagonal(diag))

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # TODO: This is currently a dense implementation. We should implement
        # a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
