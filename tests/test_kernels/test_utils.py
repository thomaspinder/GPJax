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

from typing import (
    TYPE_CHECKING,
    List,
)

import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import (
        Array,
        Float,
    )

import pytest

from gpjax.kernels.stationary.utils import euclidean_distance


@pytest.mark.parametrize(
    ("a", "b", "distance_to_3dp"),
    [
        ([1.0], [-4.0], 5.0),
        ([1.0, -2.0], [-4.0, 3.0], 7.071),
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 2.236),
    ],
)
def test_euclidean_distance(
    a: List[float], b: List[float], distance_to_3dp: float
) -> None:
    # Convert lists to JAX arrays:
    a: Float[Array, " D"] = jnp.array(a)
    b: Float[Array, " D"] = jnp.array(b)

    # Test distance is correct to 3dp:
    assert jnp.round(euclidean_distance(a, b), 3) == distance_to_3dp
