# Copyright 2023 The GPJax Contributors. All Rights Reserved.
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

from collections.abc import Callable

from jaxtyping import (
    Array as JAXArray,
    Bool,
    Float,
    Int,
    Key,
    UInt32,
)
from numpy import ndarray as NumpyArray

OldKeyArray = UInt32[JAXArray, "2"]
JAXKeyArray = Key[JAXArray, ""]
KeyArray = (
    OldKeyArray | JAXKeyArray
)  # for compatibility regardless of enable_custom_prng setting

Array = JAXArray | NumpyArray

ScalarArray = Float[Array, ""]
ScalarBool = bool | Bool[Array, ""]
ScalarInt = int | Int[Array, ""]
ScalarFloat = float | Float[Array, ""]

VecNOrMatNM = Float[Array, " N"] | Float[Array, "N M"]

FunctionalSample = Callable[[Float[Array, "N D"]], Float[Array, "N B"]]
r""" Type alias for functions representing $B$ samples from a model, to be evaluated on
any set of $N$ inputs (of dimension $D$) and returning the evaluations of each
(potentially approximate) sample draw across these inputs.
"""

__all__ = ["FunctionalSample", "KeyArray", "ScalarBool", "ScalarFloat", "ScalarInt"]
