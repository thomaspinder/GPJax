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


def build_student_t_distribution(
    nu: int, scale_tril: Float[Array, "D D"]
) -> npd.MultivariateStudentT:
    r"""Build the spectral measure of a Matérn kernel.

    The Matérn kernel with smoothness $\nu$ has spectral density proportional to
    $(2\nu/\ell^2 + \lVert\omega\rVert^2)^{-(\nu + D/2)}$, which normalises to a
    multivariate Student's t measure with $2\nu$ degrees of freedom and scale
    matrix $\mathrm{diag}(\ell)^{-1}$.

    Args:
        nu (int): Twice the smoothness parameter of the Matérn kernel, i.e. 1, 3
            or 5 for the Matérn-1/2, -3/2 and -5/2 kernels respectively.
        scale_tril (Float[Array, "D D"]): The scale matrix $\mathrm{diag}(\ell)^{-1}$
            of the measure, as returned by `StationaryKernel._spectral_scale_tril`.

    Returns
    -------
        npd.MultivariateStudentT: The spectral measure over $\mathbb{R}^D$.
    """
    return npd.MultivariateStudentT(
        df=nu, loc=jnp.zeros(scale_tril.shape[0]), scale_tril=scale_tril
    )


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
