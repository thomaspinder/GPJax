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
import numpyro.distributions as npd

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

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class StationaryKernel(AbstractKernel):
    """Base class for stationary kernels.

    Stationary kernels are a class of kernels that are invariant to translations
    in the input space. They can be isotropic or anisotropic, meaning that they
    can have a single lengthscale for all input dimensions or a different lengthscale
    for each input dimension.
    """

    lengthscale: nnx.Variable[Lengthscale]
    variance: nnx.Variable[ScalarArray]

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]] = 1.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        super().__init__(active_dims, n_dims, compute_engine)
        self.n_dims = _validate_lengthscale(lengthscale, self.n_dims)
        if isinstance(lengthscale, nnx.Variable):
            self.lengthscale = lengthscale
        else:
            self.lengthscale = PositiveReal(lengthscale)

            # static typing
            if tp.TYPE_CHECKING:
                self.lengthscale = tp.cast(PositiveReal[Lengthscale], self.lengthscale)

        if isinstance(variance, nnx.Variable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)

            # static typing
            if tp.TYPE_CHECKING:
                self.variance = tp.cast(NonNegativeReal[ScalarFloat], self.variance)

    @property
    def spectral_density(self) -> npd.Normal | npd.StudentT:
        r"""The spectral density of the kernel.

        Returns:
            Callable[[Float[Array, "D"]], Float[Array, "D"]]: The spectral density function.
        """
        raise NotImplementedError(
            f"Kernel {self.name} does not have a spectral density."
        )


def _validate_lengthscale(
    lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]],
    n_dims: tp.Union[int, None],
):
    # Check that the lengthscale is a valid value.
    _check_lengthscale(lengthscale)

    n_dims = _check_lengthscale_dims_compat(lengthscale, n_dims)
    return n_dims


def _check_lengthscale_dims_compat(
    lengthscale: tp.Union[LengthscaleCompatible, nnx.Variable[Lengthscale]],
    n_dims: tp.Union[int, None],
):
    r"""Check that the lengthscale is compatible with n_dims.

    If possible, infer the number of input dimensions from the lengthscale.
    """

    if isinstance(lengthscale, nnx.Variable):
        return _check_lengthscale_dims_compat(lengthscale.value, n_dims)

    lengthscale = jnp.asarray(lengthscale)
    ls_shape = jnp.shape(lengthscale)

    if ls_shape == ():
        return n_dims
    elif ls_shape != () and n_dims is None:
        return ls_shape[0]
    elif ls_shape != () and n_dims is not None:
        if ls_shape != (n_dims,):
            raise ValueError(
                "Expected `lengthscale` to be compatible with the number "
                f"of input dimensions. Got `lengthscale` with shape {ls_shape}, "
                f"but the number of input dimensions is {n_dims}."
            )
        return n_dims


def _check_lengthscale(lengthscale: tp.Any):
    """Check that the lengthscale is a valid value."""

    if isinstance(lengthscale, nnx.Variable):
        _check_lengthscale(lengthscale.value)
        return

    if not isinstance(lengthscale, (int, float, jnp.ndarray, list, tuple)):
        raise TypeError(
            f"Expected `lengthscale` to be a array-like. Got {lengthscale}."
        )

    if isinstance(lengthscale, (jnp.ndarray, list)):
        ls_shape = jnp.shape(jnp.asarray(lengthscale))

        if len(ls_shape) > 1:
            raise ValueError(
                f"Expected `lengthscale` to be a scalar or 1D array. "
                f"Got `lengthscale` with shape {ls_shape}."
            )
