"""Tests for the Lineax-based linear algebra module."""

from gpjax.linalg import add_jitter, cholesky_factor, logdet
from gpjax.linalg.custom_operators import BlockDiag, Kronecker
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

# --- cholesky_factor tests ---


def test_cholesky_factor_dense():
    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    op = lx.MatrixLinearOperator(A)
    L_op = cholesky_factor(op)
    L = L_op.as_matrix()
    assert jnp.allclose(L @ L.T, A, atol=1e-5)
    assert jnp.allclose(L, jnp.tril(L))


def test_cholesky_factor_diagonal():
    d = jnp.array([4.0, 9.0, 16.0])
    op = lx.DiagonalLinearOperator(d)
    L_op = cholesky_factor(op)
    assert isinstance(L_op, lx.DiagonalLinearOperator)
    assert jnp.allclose(lx.diagonal(L_op), jnp.sqrt(d))


def test_cholesky_factor_identity():
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float64))
    L_op = cholesky_factor(op)
    assert isinstance(L_op, lx.IdentityLinearOperator)


# --- logdet tests ---


def test_logdet_dense():
    A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    op = lx.MatrixLinearOperator(A)
    expected = jnp.log(jnp.linalg.det(A))
    assert jnp.allclose(logdet(op), expected, atol=1e-5)


def test_logdet_diagonal():
    d = jnp.array([2.0, 3.0, 5.0])
    op = lx.DiagonalLinearOperator(d)
    assert jnp.allclose(logdet(op), jnp.sum(jnp.log(d)))


def test_logdet_identity():
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((4,), jnp.float64))
    assert jnp.allclose(logdet(op), 0.0)


# --- add_jitter tests ---


def test_add_jitter():
    m = jnp.eye(3)
    result = add_jitter(m, 0.1)
    assert jnp.allclose(jnp.diag(result), 1.1)


def test_add_jitter_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        add_jitter(jnp.ones((2, 3)))


def test_add_jitter_non_2d_raises():
    with pytest.raises(ValueError, match="2D"):
        add_jitter(jnp.ones((2,)))


# --- BlockDiag tests ---


def test_block_diag_mv():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    B = lx.MatrixLinearOperator(jnp.array([[5.0]]))
    bd = BlockDiag(blocks=(A, B))
    x = jnp.array([1.0, 0.0, 2.0])
    result = bd.mv(x)
    expected = jnp.array([1.0, 3.0, 10.0])
    assert jnp.allclose(result, expected)


def test_block_diag_as_matrix():
    A = lx.MatrixLinearOperator(jnp.eye(2))
    B = lx.MatrixLinearOperator(2.0 * jnp.eye(3))
    bd = BlockDiag(blocks=(A, B))
    mat = bd.as_matrix()
    assert mat.shape == (5, 5)
    expected = jax.scipy.linalg.block_diag(jnp.eye(2), 2.0 * jnp.eye(3))
    assert jnp.allclose(mat, expected)


def test_block_diag_structures():
    A = lx.MatrixLinearOperator(jnp.eye(2))
    B = lx.MatrixLinearOperator(jnp.eye(3))
    bd = BlockDiag(blocks=(A, B))
    assert bd.in_structure().shape == (5,)
    assert bd.out_structure().shape == (5,)


# --- Kronecker tests ---


def test_kronecker_mv():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    B = lx.MatrixLinearOperator(jnp.array([[5.0, 6.0], [7.0, 8.0]]))
    kron = Kronecker(A=A, B=B)
    x = jnp.ones(4)
    expected = jnp.kron(A.as_matrix(), B.as_matrix()) @ x
    result = kron.mv(x)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_kronecker_as_matrix():
    A = lx.MatrixLinearOperator(jnp.array([[1.0, 0.0], [0.0, 2.0]]))
    B = lx.MatrixLinearOperator(jnp.eye(3))
    kron = Kronecker(A=A, B=B)
    mat = kron.as_matrix()
    expected = jnp.kron(A.as_matrix(), B.as_matrix())
    assert jnp.allclose(mat, expected)


def test_kronecker_structures():
    A = lx.MatrixLinearOperator(jnp.eye(2))
    B = lx.MatrixLinearOperator(jnp.eye(3))
    kron = Kronecker(A=A, B=B)
    assert kron.in_structure().shape == (6,)
    assert kron.out_structure().shape == (6,)


# --- Deprecated wrappers ---


def test_deprecated_dense_wrapper():
    from gpjax.linalg._compat import Dense

    with pytest.warns(DeprecationWarning, match="deprecated"):
        op = Dense(jnp.eye(2))
    assert isinstance(op, lx.MatrixLinearOperator)


def test_deprecated_diagonal_wrapper():
    from gpjax.linalg._compat import Diagonal

    with pytest.warns(DeprecationWarning, match="deprecated"):
        op = Diagonal(jnp.array([1.0, 2.0]))
    assert isinstance(op, lx.DiagonalLinearOperator)
