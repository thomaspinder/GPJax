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
    r"""Build a Student's t distribution with a fixed smoothness parameter.

    For a fixed half-integer smoothness parameter, compute the spectral density of a
    Matérn kernel; a Student's t distribution.

    Args:
        nu (int): The smoothness parameter of the Matérn kernel.

    Returns
    -------
        tfp.Distribution: A Student's t distribution with the same smoothness parameter.
    """
    dist = npd.StudentT(df=nu, loc=0.0, scale=1.0)
    return dist


class SpectralDensity:
    """Spectral density of a stationary kernel.

    Wraps a NumPyro distribution (for sampling, used by RFF) and adds
    evaluation of the spectral density function S(omega) at arbitrary
    frequencies (used by HSGP).

    Args:
        distribution: A NumPyro distribution to delegate ``sample()`` to.
        evaluate_fn: A callable ``(omega, variance, lengthscale) -> S(omega)``
            that computes the un-normalized spectral density at the given
            frequencies incorporating kernel variance and lengthscale.
    """

    def __init__(
        self,
        distribution: npd.Distribution,
        evaluate_fn: tp.Callable,
    ):
        self._distribution = distribution
        self._evaluate_fn = evaluate_fn

    def sample(self, key, sample_shape):
        """Draw samples from the spectral density distribution.

        This delegates to the wrapped NumPyro distribution and is used by
        Random Fourier Features (RFF).
        """
        return self._distribution.sample(key=key, sample_shape=sample_shape)

    def __call__(self, omega, variance, lengthscale):
        """Evaluate S(omega) incorporating kernel variance and lengthscale.

        Parameters
        ----------
        omega : Array
            Frequencies at which to evaluate the spectral density.
        variance : ScalarFloat
            Kernel variance parameter (sigma^2).
        lengthscale : ScalarFloat
            Kernel lengthscale parameter (ell).

        Returns
        -------
        Array
            Spectral density values S(omega).
        """
        return self._evaluate_fn(omega, variance, lengthscale)


def squared_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Compute the squared distance between a pair of inputs.

    Args:
        x (Float[Array, " D"]): First input.
        y (Float[Array, " D"]): Second input.

    Returns
    -------
        ScalarFloat: The squared distance between the inputs.
    """
    return jnp.sum((x - y) ** 2)


def euclidean_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Compute the euclidean distance between a pair of inputs.

    Args:
        x (Float[Array, " D"]): First input.
        y (Float[Array, " D"]): Second input.

    Returns
    -------
        ScalarFloat: The euclidean distance between the inputs.
    """
    return jnp.sqrt(jnp.maximum(squared_distance(x, y), 1e-36))
