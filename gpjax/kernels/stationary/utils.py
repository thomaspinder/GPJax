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
from jaxtyping import Float
import numpyro.distributions as npd

from gpjax.typing import (
    Array,
    ScalarFloat,
)


def build_student_t_distribution(nu: int) -> npd.StudentT:
    r"""Student's t distribution for Matern spectral densities.

    Args:
        nu: Degrees of freedom (equals the Matern smoothness parameter
            :math:`\nu` mapped to the nearest integer: 1, 3, or 5).

    Returns:
        A standard Student's t distribution with ``df=nu``.
    """
    return npd.StudentT(df=nu, loc=0.0, scale=1.0)


class SpectralDensity:
    r"""Spectral density :math:`S(\omega)` of a stationary kernel.

    This class serves two roles:

    1. **Sampling** (for RFF): delegates to a wrapped NumPyro distribution
       via :meth:`sample`, drawing frequency samples :math:`\omega`.
    2. **Evaluation** (for HSGP): computes :math:`S(\omega)` at arbitrary
       frequencies via :meth:`__call__`, incorporating kernel variance and
       lengthscale.

    Args:
        distribution: NumPyro distribution to sample from (used by RFF).
        evaluate_fn: Callable ``(omega, variance, lengthscale) -> S(omega)``
            that evaluates the spectral density at given frequencies.
    """

    def __init__(
        self,
        distribution: npd.Distribution,
        evaluate_fn: tp.Callable[
            [Float[Array, " M"], ScalarFloat, ScalarFloat], Float[Array, " M"]
        ],
    ):
        self._distribution = distribution
        self._evaluate_fn = evaluate_fn

    def sample(self, key: Array, sample_shape: tuple[int, ...]) -> Float[Array, "..."]:
        """Draw frequency samples from the spectral distribution (used by RFF).

        Args:
            key: JAX PRNG key.
            sample_shape: Shape of the sample array.

        Returns:
            Sampled frequencies.
        """
        return self._distribution.sample(key=key, sample_shape=sample_shape)

    def __call__(
        self,
        omega: Float[Array, " M"],
        variance: ScalarFloat,
        lengthscale: ScalarFloat,
    ) -> Float[Array, " M"]:
        r"""Evaluate :math:`S(\omega)` at the given frequencies (used by HSGP).

        Args:
            omega: Frequencies at which to evaluate the spectral density.
            variance: Kernel variance :math:`\sigma^2`.
            lengthscale: Kernel lengthscale :math:`\ell`.

        Returns:
            Spectral density values :math:`S(\omega)`.
        """
        return self._evaluate_fn(omega, variance, lengthscale)


def squared_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Squared Euclidean distance :math:`\lVert x - y \rVert^2`.

    Args:
        x: First input vector.
        y: Second input vector.

    Returns:
        The squared distance between the inputs.
    """
    return jnp.sum((x - y) ** 2)


def euclidean_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Euclidean distance :math:`\lVert x - y \rVert`, clamped for stability.

    Args:
        x: First input vector.
        y: Second input vector.

    Returns:
        The Euclidean distance between the inputs.
    """
    return jnp.sqrt(jnp.maximum(squared_distance(x, y), 1e-36))
