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
