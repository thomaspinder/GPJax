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
from dataclasses import dataclass
import warnings

from beartype.typing import Optional
import jax
import jax.numpy as jnp
from jaxtyping import Num

from gpjax.typing import Array


@jax.tree_util.register_pytree_node_class
@dataclass(slots=True)
class Dataset:
    r"""Base class for datasets.

    Args:
        X: input data.
        y: output data.
    """

    X: Optional[Num[Array, "N D"]] = None
    y: Optional[Num[Array, "N Q"]] = None

    def __post_init__(self) -> None:
        r"""Checks that the shapes of $X$ and $y$ are compatible,
        and provides warnings regarding the precision of $X$ and $y$."""
        _check_shape(self.X, self.y)
        _check_precision(self.X, self.y)

    def __repr__(self) -> str:
        r"""Returns a string representation of the dataset."""
        repr = f"Dataset(Number of observations: {self.n:=} - Input dimension: {self.in_dim})"
        return repr

    def is_supervised(self) -> bool:
        r"""Returns `True` if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        r"""Returns `True` if the dataset is unsupervised."""
        return self.X is None and self.y is not None

    def __add__(self, other: "Dataset") -> "Dataset":
        r"""Combine two datasets. Right hand dataset is stacked beneath the left."""
        X = None
        y = None

        if self.X is not None and other.X is not None:
            X = jnp.concatenate((self.X, other.X))

        if self.y is not None and other.y is not None:
            y = jnp.concatenate((self.y, other.y))

        return Dataset(X=X, y=y)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        r"""Dimension of the inputs, $X$."""
        return self.X.shape[1]

    @property
    def multi_output(self) -> bool:
        """True if the dataset has more than one output dimension."""
        if self.y is None or self.y.ndim < 2:
            return False
        return self.y.shape[1] > 1

    @property
    def num_outputs(self) -> int:
        """Number of output dimensions."""
        if self.y is None or self.y.ndim < 2:
            return 1
        return self.y.shape[1]

    def tree_flatten(self):
        return (self.X, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _check_shape(
    X: Optional[Num[Array, "..."]], y: Optional[Num[Array, "..."]]
) -> None:
    r"""Checks that the shapes of $X$ and $y$ are compatible."""
    if X is not None and y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(
            "Inputs, X, and outputs, y, must have the same number of rows."
            f" Got X.shape={X.shape} and y.shape={y.shape}."
        )

    if X is not None and X.ndim != 2:
        raise ValueError(
            f"Inputs, X, must be a 2-dimensional array. Got X.ndim={X.ndim}."
        )

    if y is not None and y.ndim != 2:
        raise ValueError(
            f"Outputs, y, must be a 2-dimensional array. Got y.ndim={y.ndim}."
        )


def _check_precision(
    X: Optional[Num[Array, "..."]], y: Optional[Num[Array, "..."]]
) -> None:
    r"""Checks the precision of $X$ and $y`."""
    if X is not None and X.dtype != jnp.float64:
        warnings.warn(
            "X is not of type float64. "
            f"Got X.dtype={X.dtype}. This may lead to numerical instability. ",
            stacklevel=2,
        )

    if y is not None and y.dtype != jnp.float64:
        warnings.warn(
            "y is not of type float64."
            f"Got y.dtype={y.dtype}. This may lead to numerical instability.",
            stacklevel=2,
        )


__all__ = [
    "Dataset",
]
