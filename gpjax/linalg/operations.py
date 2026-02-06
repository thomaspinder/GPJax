"""Linear algebra operations for GPJax LinearOperators."""

from jax import Array
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from gpjax.linalg.operators import (
    BlockDiag,
    Dense,
    Diagonal,
    Identity,
    Kronecker,
    LinearOperator,
    Triangular,
)
from gpjax.typing import ScalarFloat


def lower_cholesky(A: LinearOperator) -> LinearOperator:
    """Compute the lower Cholesky decomposition of a positive semi-definite operator.

    This function dispatches on the type of the input LinearOperator to provide
    efficient implementations for different operator structures.

    Args:
        A: A positive semi-definite LinearOperator.

    Returns:
        The lower triangular Cholesky factor L such that A = L @ L.T.
    """

    def _handle_identity(A):
        return A

    def _handle_diagonal(A):
        return Diagonal(jnp.sqrt(A.diagonal))

    def _handle_triangular(A):
        if A.lower:
            return A
        return Triangular(jnp.linalg.cholesky(A.to_dense()), lower=True)

    def _handle_kronecker(A):
        cholesky_ops = [lower_cholesky(op) for op in A.operators]
        return Kronecker(cholesky_ops)

    def _handle_blockdiag(A):
        cholesky_ops = [lower_cholesky(op) for op in A.operators]
        return BlockDiag(cholesky_ops, multiplicities=A.multiplicities)

    def _handle_dense(A):
        return Triangular(jnp.linalg.cholesky(A.array), lower=True)

    def _handle_default(A):
        return Triangular(jnp.linalg.cholesky(A.to_dense()), lower=True)

    dispatch_table = {
        Identity: _handle_identity,
        Diagonal: _handle_diagonal,
        Triangular: _handle_triangular,
        Kronecker: _handle_kronecker,
        BlockDiag: _handle_blockdiag,
        Dense: _handle_dense,
    }

    handler = dispatch_table.get(type(A), _handle_default)
    return handler(A)


def solve(
    A: LinearOperator,
    b: Float[Array, " N"] | Float[Array, " N M"],
) -> Float[Array, " N"] | Float[Array, " N M"]:
    """Solve the linear system A @ x = b for x.

    This function dispatches on the type of the input LinearOperator to provide
    efficient implementations for different operator structures.

    Args:
        A: A LinearOperator representing the matrix A.
        b: The right-hand side vector or matrix.

    Returns:
        The solution x to the linear system.
    """
    # Handle different shapes of b
    if b.ndim == 1:
        b = b[:, None]
        squeeze_output = True
    else:
        squeeze_output = False

    # Dispatch based on operator type
    if isinstance(A, Identity):
        # Identity matrix: x = b
        result = b

    elif isinstance(A, Diagonal):
        # Diagonal matrix: element-wise division
        result = b / A.diagonal[:, None]

    elif isinstance(A, Triangular):
        # Triangular matrix: use triangular solver
        result = jsp.linalg.solve_triangular(A.array, b, lower=A.lower)

    elif isinstance(A, Dense):
        # Dense matrix: use standard solver
        result = jnp.linalg.solve(A.array, b)

    else:
        # Default: convert to dense and solve
        result = jnp.linalg.solve(A.to_dense(), b)

    if squeeze_output:
        result = result.squeeze(-1)

    return result


def logdet(A: LinearOperator) -> ScalarFloat:
    """Compute the log-determinant of a linear operator.

    This function dispatches on the type of the input LinearOperator to provide
    efficient implementations for different operator structures.

    Args:
        A: A LinearOperator.

    Returns:
        The log-determinant of A.
    """

    def _handle_identity(A):
        return jnp.array(0.0)

    def _handle_diagonal(A):
        return jnp.sum(jnp.log(A.diagonal))

    def _handle_triangular(A):
        diag_elements = jnp.diag(A.array)
        return jnp.sum(jnp.log(diag_elements))

    def _handle_kronecker(A):
        logdet_val = 0.0
        for i, op in enumerate(A.operators):
            op_logdet = logdet(op)
            power = 1
            for j, other_op in enumerate(A.operators):
                if i != j:
                    power *= other_op.shape[0]
            logdet_val += power * op_logdet
        return logdet_val

    def _handle_blockdiag(A):
        logdet_val = 0.0
        for op, mult in zip(A.operators, A.multiplicities, strict=False):
            logdet_val += mult * logdet(op)
        return logdet_val

    def _handle_dense(A):
        _, logdet_val = jnp.linalg.slogdet(A.array)
        return logdet_val

    def _handle_default(A):
        _, logdet_val = jnp.linalg.slogdet(A.to_dense())
        return logdet_val

    dispatch_table = {
        Identity: _handle_identity,
        Diagonal: _handle_diagonal,
        Triangular: _handle_triangular,
        Kronecker: _handle_kronecker,
        BlockDiag: _handle_blockdiag,
        Dense: _handle_dense,
    }

    handler = dispatch_table.get(type(A), _handle_default)
    return handler(A)


def diag(A: LinearOperator) -> Float[Array, " N"]:
    """Extract the diagonal of a linear operator.

    This function dispatches on the type of the input LinearOperator to provide
    efficient implementations for different operator structures.

    Args:
        A: A LinearOperator.

    Returns:
        The diagonal elements of A as a 1D array.
    """

    def _handle_identity(A):
        n = A.shape[0]
        return jnp.ones(n, dtype=A.dtype)

    def _handle_diagonal(A):
        return A.diagonal

    def _handle_triangular(A):
        return jnp.diag(A.array)

    def _handle_kronecker(A):
        result = diag(A.operators[0])
        for op in A.operators[1:]:
            result = jnp.kron(result, diag(op))
        return result

    def _handle_blockdiag(A):
        diags = []
        for op, mult in zip(A.operators, A.multiplicities, strict=False):
            op_diag = diag(op)
            for _ in range(mult):
                diags.append(op_diag)
        return jnp.concatenate(diags)

    def _handle_dense(A):
        return jnp.diag(A.array)

    def _handle_default(A):
        return jnp.diag(A.to_dense())

    dispatch_table = {
        Identity: _handle_identity,
        Diagonal: _handle_diagonal,
        Triangular: _handle_triangular,
        Kronecker: _handle_kronecker,
        BlockDiag: _handle_blockdiag,
        Dense: _handle_dense,
    }

    handler = dispatch_table.get(type(A), _handle_default)
    return handler(A)
