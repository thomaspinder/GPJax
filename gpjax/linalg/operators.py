"""Linear operator abstractions for GPJax."""

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
)

from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Float


class LinearOperator(ABC):
    """Abstract base class for linear operators."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Return the shape of the operator."""

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        """Return the data type of the operator."""

    @abstractmethod
    def to_dense(self) -> Float[Array, "M N"]:
        """Convert the operator to a dense JAX array."""

    @property
    def T(self) -> "LinearOperator":
        """Return the transpose of the operator."""
        # Default implementation: convert to dense and transpose
        return Dense(self.to_dense().T)

    def __matmul__(self, other):
        """Matrix multiplication with another array or operator."""
        if hasattr(other, "to_dense"):
            # Other is a LinearOperator
            return Dense(self.to_dense() @ other.to_dense())
        else:
            # Other is a JAX array
            return self.to_dense() @ other

    def __rmatmul__(self, other):
        """Right matrix multiplication (other @ self)."""
        if hasattr(other, "to_dense"):
            # Other is a LinearOperator
            return Dense(other.to_dense() @ self.to_dense())
        else:
            # Other is a JAX array
            return other @ self.to_dense()

    def __add__(self, other):
        """Addition with another array or operator."""
        if hasattr(other, "to_dense"):
            # Other is a LinearOperator
            return Dense(self.to_dense() + other.to_dense())
        else:
            # Other is a JAX array
            return Dense(self.to_dense() + other)

    def __radd__(self, other):
        """Right addition (other + self)."""
        if hasattr(other, "to_dense"):
            # Other is a LinearOperator
            return Dense(other.to_dense() + self.to_dense())
        else:
            # Other is a JAX array
            return Dense(other + self.to_dense())

    def __sub__(self, other):
        """Subtraction with another array or operator."""
        if hasattr(other, "to_dense"):
            # Other is a LinearOperator
            return Dense(self.to_dense() - other.to_dense())
        else:
            # Other is a JAX array
            return Dense(self.to_dense() - other)

    def __rsub__(self, other):
        """Right subtraction (other - self)."""
        if hasattr(other, "to_dense"):
            # Other is a LinearOperator
            return Dense(other.to_dense() - self.to_dense())
        else:
            # Other is a JAX array
            return Dense(other - self.to_dense())

    def __mul__(self, other):
        """Scalar multiplication (self * scalar)."""
        if jnp.isscalar(other):
            return Dense(self.to_dense() * other)
        else:
            # Element-wise multiplication with array
            return Dense(self.to_dense() * other)

    def __rmul__(self, other):
        """Right scalar multiplication (scalar * self)."""
        if jnp.isscalar(other):
            return Dense(other * self.to_dense())
        else:
            # Element-wise multiplication with array
            return Dense(other * self.to_dense())


class Dense(LinearOperator):
    """Dense linear operator wrapping a JAX array."""

    def __init__(self, array: Float[Array, "M N"]):
        super().__init__()
        self.array = array

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    def to_dense(self) -> Float[Array, "M N"]:
        return self.array

    @property
    def T(self) -> "Dense":
        return Dense(self.array.T)


class Diagonal(LinearOperator):
    """Diagonal linear operator."""

    def __init__(self, diagonal: Float[Array, " N"]):
        super().__init__()
        self.diagonal = diagonal

    @property
    def shape(self) -> tuple[int, int]:
        n = self.diagonal.shape[0]
        return (n, n)

    @property
    def dtype(self) -> jnp.dtype:
        return self.diagonal.dtype

    def to_dense(self) -> Float[Array, "N N"]:
        return jnp.diag(self.diagonal)

    @property
    def T(self) -> "Diagonal":
        return Diagonal(self.diagonal)


class Identity(LinearOperator):
    """Identity linear operator."""

    def __init__(self, shape: int | tuple[int, int], dtype=jnp.float64):
        super().__init__()
        if isinstance(shape, int):
            self._shape = (shape, shape)
        else:
            if shape[0] != shape[1]:
                raise ValueError(f"Identity matrix must be square, got shape {shape}")
            self._shape = shape
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> Any:
        return self._dtype

    def to_dense(self) -> Float[Array, "N N"]:
        n = self._shape[0]
        return jnp.eye(n, dtype=self._dtype)

    @property
    def T(self) -> "Identity":
        return Identity(self._shape, dtype=self._dtype)


class Triangular(LinearOperator):
    """Triangular linear operator."""

    def __init__(self, array: Float[Array, "N N"], lower: bool = True):
        super().__init__()
        self.array = array
        self.lower = lower

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    @property
    def dtype(self) -> Any:
        return self.array.dtype

    def to_dense(self) -> Float[Array, "N N"]:
        if self.lower:
            return jnp.tril(self.array)
        else:
            return jnp.triu(self.array)

    @property
    def T(self) -> "Triangular":
        return Triangular(self.array.T, lower=not self.lower)


class BlockDiag(LinearOperator):
    """Block diagonal linear operator."""

    def __init__(
        self, operators: list[LinearOperator], multiplicities: list[int] | None = None
    ):
        super().__init__()
        self.operators = operators

        # Handle multiplicities - how many times each block is repeated
        if multiplicities is None:
            self.multiplicities = [1] * len(operators)
        else:
            if len(multiplicities) != len(operators):
                raise ValueError(
                    f"Length of multiplicities ({len(multiplicities)}) must match operators ({len(operators)})"
                )
            self.multiplicities = multiplicities

        # Calculate total shape with multiplicities
        rows = sum(
            op.shape[0] * mult
            for op, mult in zip(operators, self.multiplicities, strict=False)
        )
        cols = sum(
            op.shape[1] * mult
            for op, mult in zip(operators, self.multiplicities, strict=False)
        )
        self._shape = (rows, cols)

        # Use dtype of first operator (assuming all same dtype)
        if operators:
            self._dtype = operators[0].dtype
        else:
            self._dtype = jnp.float64

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> Any:
        return self._dtype

    def to_dense(self) -> Float[Array, "M N"]:
        if not self.operators:
            return jnp.zeros(self._shape, dtype=self._dtype)

        # Convert each operator to dense and create block diagonal with multiplicities
        expanded_blocks = []
        for op, mult in zip(self.operators, self.multiplicities, strict=False):
            op_dense = op.to_dense()
            for _ in range(mult):
                expanded_blocks.append(op_dense)

        # Create the full block diagonal matrix
        n_blocks = len(expanded_blocks)
        if n_blocks == 0:
            return jnp.zeros(self._shape, dtype=self._dtype)

        # Build the block diagonal matrix
        rows = []
        for i in range(n_blocks):
            row = []
            for j in range(n_blocks):
                if i == j:
                    row.append(expanded_blocks[i])
                else:
                    row.append(
                        jnp.zeros(
                            (expanded_blocks[i].shape[0], expanded_blocks[j].shape[1]),
                            dtype=self._dtype,
                        )
                    )
            rows.append(row)
        return jnp.block(rows)

    @property
    def T(self) -> "BlockDiag":
        transposed_ops = [op.T for op in self.operators]
        return BlockDiag(transposed_ops, multiplicities=self.multiplicities)


class Kronecker(LinearOperator):
    """Kronecker product linear operator."""

    def __init__(self, operators: list[LinearOperator]):
        super().__init__()
        if len(operators) < 2:
            raise ValueError("Kronecker product requires at least 2 operators")
        self.operators = operators

        # Calculate shape as product of individual shapes
        rows = 1
        cols = 1
        for op in operators:
            rows *= op.shape[0]
            cols *= op.shape[1]
        self._shape = (rows, cols)

        # Use dtype of first operator
        self._dtype = operators[0].dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> Any:
        return self._dtype

    def to_dense(self) -> Float[Array, "M N"]:
        # Convert to dense and compute Kronecker product
        result = self.operators[0].to_dense()
        for op in self.operators[1:]:
            result = jnp.kron(result, op.to_dense())
        return result

    @property
    def T(self) -> "Kronecker":
        transposed_ops = [op.T for op in self.operators]
        return Kronecker(transposed_ops)


def _dense_tree_flatten(dense):
    return (dense.array,), None


def _dense_tree_unflatten(aux_data, children):
    return Dense(children[0])


jtu.register_pytree_node(Dense, _dense_tree_flatten, _dense_tree_unflatten)


def _diagonal_tree_flatten(diagonal):
    return (diagonal.diagonal,), None


def _diagonal_tree_unflatten(aux_data, children):
    return Diagonal(children[0])


jtu.register_pytree_node(Diagonal, _diagonal_tree_flatten, _diagonal_tree_unflatten)


def _identity_tree_flatten(identity):
    return (), (identity._shape, identity._dtype)


def _identity_tree_unflatten(aux_data, children):
    shape, dtype = aux_data
    return Identity(shape, dtype)


jtu.register_pytree_node(Identity, _identity_tree_flatten, _identity_tree_unflatten)


def _triangular_tree_flatten(triangular):
    return (triangular.array,), triangular.lower


def _triangular_tree_unflatten(aux_data, children):
    return Triangular(children[0], aux_data)


jtu.register_pytree_node(
    Triangular, _triangular_tree_flatten, _triangular_tree_unflatten
)


def _blockdiag_tree_flatten(blockdiag):
    return tuple(blockdiag.operators), blockdiag.multiplicities


def _blockdiag_tree_unflatten(aux_data, children):
    return BlockDiag(list(children), aux_data)


jtu.register_pytree_node(BlockDiag, _blockdiag_tree_flatten, _blockdiag_tree_unflatten)


def _kronecker_tree_flatten(kronecker):
    return tuple(kronecker.operators), None


def _kronecker_tree_unflatten(aux_data, children):
    return Kronecker(list(children))


jtu.register_pytree_node(Kronecker, _kronecker_tree_flatten, _kronecker_tree_unflatten)

__all__ = [
    "BlockDiag",
    "Dense",
    "Diagonal",
    "Identity",
    "Kronecker",
    "LinearOperator",
    "Triangular",
]
