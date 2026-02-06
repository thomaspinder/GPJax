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

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import (
        Array,
        Float,
    )

from gpjax.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
    squared_distance,
)
import numpyro.distributions as npd
import pytest


@pytest.mark.parametrize(
    ("a", "b", "distance_to_3dp"),
    [
        ([1.0], [-4.0], 5.0),
        ([1.0, -2.0], [-4.0, 3.0], 7.071),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 2.236),
    ],
)
def test_euclidean_distance(
    a: list[float], b: list[float], distance_to_3dp: float
) -> None:
    # Convert lists to JAX arrays:
    a: Float[Array, " D"] = jnp.array(a)
    b: Float[Array, " D"] = jnp.array(b)

    # Test distance is correct to 3dp:
    assert jnp.round(euclidean_distance(a, b), 3) == distance_to_3dp


# ---------------------------------------------------------------------------
# squared_distance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0], [-4.0], 25.0),
        ([1.0, -2.0], [-4.0, 3.0], 50.0),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 5.0),
        ([0.0], [0.0], 0.0),
    ],
)
def test_squared_distance(a: list[float], b: list[float], expected: float) -> None:
    a_arr = jnp.array(a)
    b_arr = jnp.array(b)
    assert jnp.allclose(squared_distance(a_arr, b_arr), expected)


def test_squared_distance_symmetry() -> None:
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    assert jnp.allclose(squared_distance(a, b), squared_distance(b, a))


def test_euclidean_distance_is_sqrt_of_squared() -> None:
    a = jnp.array([1.0, 2.0])
    b = jnp.array([4.0, 6.0])
    assert jnp.allclose(
        euclidean_distance(a, b) ** 2, squared_distance(a, b), atol=1e-5
    )


def test_euclidean_distance_same_point() -> None:
    """Euclidean distance of same point should be close to zero (clamped by 1e-36)."""
    a = jnp.array([1.0, 2.0])
    assert euclidean_distance(a, a) >= 0.0
    assert euclidean_distance(a, a) < 1e-10


# ---------------------------------------------------------------------------
# build_student_t_distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [1, 3, 5, 10])
def test_build_student_t_distribution(nu: int) -> None:
    dist = build_student_t_distribution(nu)
    assert isinstance(dist, npd.StudentT)
    assert dist.df == nu
    assert dist.loc == 0.0
    assert dist.scale == 1.0


def test_student_t_is_sampleable() -> None:
    import jax.random as jr

    dist = build_student_t_distribution(5)
    samples = dist.sample(jr.key(0), (100,))
    assert samples.shape == (100,)
    assert jnp.all(jnp.isfinite(samples))
