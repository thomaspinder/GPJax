"""Utility functions for the linear algebra module."""

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array
import lineax as lx

from gpjax.linalg.custom_operators import BlockDiag, Kronecker


def add_jitter(matrix: Array, jitter: float | Array = 1e-6) -> Array:
    """Add jitter to the diagonal of a matrix for numerical stability."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
    return matrix + jnp.eye(matrix.shape[0]) * jitter


@functools.singledispatch
def cholesky_factor(op: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Cholesky factor of a PSD operator. Returns lower-triangular L s.t. A = L L^T."""
    L = jnp.linalg.cholesky(op.as_matrix())
    return lx.MatrixLinearOperator(L, tags=lx.lower_triangular_tag)


@cholesky_factor.register(lx.DiagonalLinearOperator)
def _cholesky_diagonal(op):
    return lx.DiagonalLinearOperator(jnp.sqrt(lx.diagonal(op)))


@cholesky_factor.register(lx.IdentityLinearOperator)
def _cholesky_identity(op):
    return op


@cholesky_factor.register(BlockDiag)
def _cholesky_blockdiag(op):
    return BlockDiag([cholesky_factor(block) for block in op.blocks])


@cholesky_factor.register(Kronecker)
def _cholesky_kronecker(op):
    return Kronecker(cholesky_factor(op.A), cholesky_factor(op.B))


@functools.singledispatch
def logdet(op: lx.AbstractLinearOperator) -> jax.Array:
    """Log-determinant of a PSD operator via its Cholesky factor."""
    L = cholesky_factor(op)
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L.as_matrix())))


@logdet.register(lx.DiagonalLinearOperator)
def _logdet_diagonal(op):
    return jnp.sum(jnp.log(lx.diagonal(op)))


@logdet.register(lx.IdentityLinearOperator)
def _logdet_identity(op):
    return jnp.array(0.0)


@logdet.register(BlockDiag)
def _logdet_blockdiag(op):
    return sum(logdet(block) for block in op.blocks)


@logdet.register(Kronecker)
def _logdet_kronecker(op):
    n = op.A.out_structure().shape[0]
    m = op.B.out_structure().shape[0]
    return m * logdet(op.A) + n * logdet(op.B)
