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

import abc
import typing as tp

from jax import vmap
from jaxtyping import (
    Float,
    Num,
)

import gpjax
from gpjax.linalg import (
    Dense,
    Diagonal,
    psd,
)
from gpjax.typing import Array

K = tp.TypeVar("K", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


class AbstractKernelComputation:
    r"""Abstract class for kernel computations.

    This class defines the interface for computing the covariance matrix of a kernel
    function. It is used to compute the Gram matrix, cross-covariance, and diagonal
    variance of a kernel function. Each computation engine implements the computation
    of these quantities in a different way. Subclasses implement computations as private
    methods. If a non-standard interface is required, the subclass should override the
    public methods of this class.

    """

    def _gram(
        self,
        kernel: K,
        x: Num[Array, "N D"],
    ) -> Float[Array, "N N"]:
        Kxx = self.cross_covariance(kernel, x, x)
        return Kxx

    def gram(
        self,
        kernel: K,
        x: Num[Array, "N D"],
    ) -> Dense:
        r"""For a given kernel, compute Gram covariance operator of the kernel function
        on an input matrix of shape `(N, D)`.

        Args:
            kernel: the kernel function.
            x: the inputs to the kernel function of shape `(N, D)`.

        Returns:
            The Gram covariance of the kernel function as a linear operator.
        """
        Kxx = self.cross_covariance(kernel, x, x)
        return psd(Dense(Kxx))

    @abc.abstractmethod
    def _cross_covariance(
        self, kernel: K, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]: ...

    def cross_covariance(
        self, kernel: K, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]:
        r"""For a given kernel, compute the cross-covariance matrix on an a pair
        of input matrices with shape `(N, D)` and `(M, D)`.

        Args:
            kernel: the kernel function.
            x: the first input matrix of shape `(N, D)`.
            y: the second input matrix of shape `(M, D)`.

        Returns:
            The computed cross-covariance of shape `(N, M)`.
        """
        return self._cross_covariance(kernel, x, y)

    def _diagonal(self, kernel: K, inputs: Num[Array, "N D"]) -> Diagonal:
        return psd(Diagonal(vmap(lambda x: kernel(x, x))(inputs)))

    def diagonal(self, kernel: K, inputs: Num[Array, "N D"]) -> Diagonal:
        r"""For a given kernel, compute the elementwise diagonal of the
        NxN gram matrix on an input matrix of shape `(N, D)`.

        Args:
            kernel: the kernel function.
            inputs: the input matrix of shape `(N, D)`.

        Returns:
            The computed diagonal variance as a `Diagonal` linear operator.
        """
        return self._diagonal(kernel, inputs)
