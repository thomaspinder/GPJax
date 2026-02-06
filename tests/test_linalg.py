"""Tests for the gpjax.linalg module."""

from gpjax.linalg import (
    BlockDiag,
    Dense,
    Diagonal,
    Identity,
    Kronecker,
    Triangular,
    diag,
    logdet,
    lower_cholesky,
    psd,
    solve,
)
from gpjax.linalg.utils import add_jitter
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

# Enable 64-bit precision for tests
config.update("jax_enable_x64", True)


class TestDenseOperator:
    """Tests for Dense linear operator."""

    def test_shape_and_dtype(self):
        """Test shape and dtype properties."""
        key = jr.key(123)
        array = jr.normal(key, shape=(5, 5))
        op = Dense(array)

        assert op.shape == (5, 5)
        assert op.dtype == array.dtype

    def test_to_dense(self):
        """Test conversion to dense array."""
        key = jr.key(123)
        array = jr.normal(key, shape=(3, 4))
        op = Dense(array)

        dense = op.to_dense()
        assert jnp.allclose(dense, array)
        assert dense.shape == array.shape


class TestDiagonalOperator:
    """Tests for Diagonal linear operator."""

    def test_shape_and_dtype(self):
        """Test shape and dtype properties."""
        diag = jnp.array([1.0, 2.0, 3.0])
        op = Diagonal(diag)

        assert op.shape == (3, 3)
        assert op.dtype == diag.dtype

    def test_to_dense(self):
        """Test conversion to dense array."""
        diag = jnp.array([1.0, 2.0, 3.0, 4.0])
        op = Diagonal(diag)

        dense = op.to_dense()
        expected = jnp.diag(diag)
        assert jnp.allclose(dense, expected)
        assert dense.shape == (4, 4)


class TestIdentityOperator:
    """Tests for Identity linear operator."""

    def test_shape_and_dtype_from_int(self):
        """Test shape and dtype with integer input."""
        op = Identity(5)

        assert op.shape == (5, 5)
        assert op.dtype == jnp.float64

    def test_shape_and_dtype_from_tuple(self):
        """Test shape and dtype with tuple input."""
        op = Identity((4, 4), dtype=jnp.float32)

        assert op.shape == (4, 4)
        assert op.dtype == jnp.float32

    def test_non_square_raises_error(self):
        """Test that non-square shape raises error."""
        with pytest.raises(ValueError, match="Identity matrix must be square"):
            Identity((3, 4))

    def test_to_dense(self):
        """Test conversion to dense array."""
        op = Identity(3)

        dense = op.to_dense()
        expected = jnp.eye(3)
        assert jnp.allclose(dense, expected)
        assert dense.shape == (3, 3)


class TestTriangularOperator:
    """Tests for Triangular linear operator."""

    def test_lower_triangular(self):
        """Test lower triangular operator."""
        key = jr.key(123)
        full_array = jr.normal(key, shape=(4, 4))
        op = Triangular(full_array, lower=True)

        assert op.shape == (4, 4)
        assert op.dtype == full_array.dtype

        dense = op.to_dense()
        expected = jnp.tril(full_array)
        assert jnp.allclose(dense, expected)

    def test_upper_triangular(self):
        """Test upper triangular operator."""
        key = jr.key(123)
        full_array = jr.normal(key, shape=(3, 3))
        op = Triangular(full_array, lower=False)

        assert op.shape == (3, 3)

        dense = op.to_dense()
        expected = jnp.triu(full_array)
        assert jnp.allclose(dense, expected)


class TestBlockDiagOperator:
    """Tests for BlockDiag linear operator."""

    def test_shape_and_dtype(self):
        """Test shape and dtype for block diagonal."""
        ops = [
            Dense(jnp.ones((2, 2))),
            Dense(jnp.ones((3, 3))),
            Dense(jnp.ones((1, 1))),
        ]
        op = BlockDiag(ops)

        assert op.shape == (6, 6)  # 2+3+1 = 6
        assert op.dtype == jnp.float64

    def test_to_dense(self):
        """Test conversion to dense array."""
        block1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        block2 = jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])

        ops = [Dense(block1), Dense(block2)]
        op = BlockDiag(ops)

        dense = op.to_dense()

        # Check that blocks are on diagonal
        assert jnp.allclose(dense[:2, :2], block1)
        assert jnp.allclose(dense[2:5, 2:5], block2)

        # Check off-diagonal blocks are zero
        assert jnp.allclose(dense[:2, 2:], 0.0)
        assert jnp.allclose(dense[2:, :2], 0.0)

    def test_empty_operators_list(self):
        """Test block diagonal with empty list."""
        op = BlockDiag([])
        assert op.shape == (0, 0)

        dense = op.to_dense()
        assert dense.shape == (0, 0)


class TestKroneckerOperator:
    """Tests for Kronecker product linear operator."""

    def test_shape_and_dtype(self):
        """Test shape and dtype for Kronecker product."""
        ops = [
            Dense(jnp.ones((2, 3))),
            Dense(jnp.ones((4, 5))),
        ]
        op = Kronecker(ops)

        assert op.shape == (2 * 4, 3 * 5)  # (8, 15)
        assert op.dtype == jnp.float64

    def test_to_dense(self):
        """Test conversion to dense array."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        ops = [Dense(A), Dense(B)]
        op = Kronecker(ops)

        dense = op.to_dense()
        expected = jnp.kron(A, B)
        assert jnp.allclose(dense, expected)

    def test_multiple_operators(self):
        """Test Kronecker product with more than 2 operators."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0]])
        C = jnp.array([[6.0, 7.0]])

        ops = [Dense(A), Dense(B), Dense(C)]
        op = Kronecker(ops)

        dense = op.to_dense()
        expected = jnp.kron(jnp.kron(A, B), C)
        assert jnp.allclose(dense, expected)
        assert op.shape == (2 * 1 * 1, 2 * 1 * 2)  # (2, 4)

    def test_insufficient_operators_raises_error(self):
        """Test that less than 2 operators raises error."""
        with pytest.raises(ValueError, match="at least 2 operators"):
            Kronecker([Dense(jnp.ones((2, 2)))])


class TestPSDWrapper:
    """Tests for the psd wrapper function."""

    def test_psd_returns_operator_unchanged(self):
        """Test that psd returns the operator unchanged."""
        key = jr.key(123)
        array = jr.normal(key, shape=(3, 3))
        op = Dense(array)

        psd_op = psd(op)
        assert psd_op is op  # Should be the same object


class TestLowerCholesky:
    """Tests for the lower_cholesky function."""

    def test_dense_cholesky(self):
        """Test Cholesky decomposition of dense matrices."""
        # Create a positive definite matrix
        key = jr.key(123)
        A_raw = jr.normal(key, shape=(3, 3))
        A_dense = A_raw @ A_raw.T + jnp.eye(3)  # Make it positive definite

        op = Dense(A_dense)
        L = lower_cholesky(op)

        assert isinstance(L, Triangular)
        assert L.lower

        # Check that L @ L.T ≈ A
        reconstructed = L.to_dense() @ L.to_dense().T
        assert jnp.allclose(reconstructed, A_dense, atol=1e-6)

    def test_diagonal_cholesky(self):
        """Test Cholesky decomposition of diagonal matrices."""
        diag_values = jnp.array([1.0, 4.0, 9.0])
        op = Diagonal(diag_values)
        L = lower_cholesky(op)

        assert isinstance(L, Diagonal)
        assert jnp.allclose(L.diagonal, jnp.sqrt(diag_values))

    def test_identity_cholesky(self):
        """Test Cholesky decomposition of identity matrices."""
        op = Identity(4)
        L = lower_cholesky(op)

        assert isinstance(L, Identity)
        assert L.shape == (4, 4)

    def test_kronecker_cholesky(self):
        """Test Cholesky decomposition of Kronecker products."""
        # Create two small positive definite matrices
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        B = jnp.array([[2.0, 0.0], [0.0, 2.0]])

        op = Kronecker([Dense(A), Dense(B)])
        L = lower_cholesky(op)

        assert isinstance(L, Kronecker)
        assert len(L.operators) == 2
        assert isinstance(L.operators[0], Triangular)
        assert isinstance(L.operators[1], Triangular)

        # Check the Cholesky property
        L_dense = L.to_dense()
        K_dense = op.to_dense()
        assert jnp.allclose(L_dense @ L_dense.T, K_dense, atol=1e-6)

    def test_block_diag_cholesky(self):
        """Test Cholesky decomposition of block diagonal matrices."""
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        B = jnp.array([[2.0]])

        op = BlockDiag([Dense(A), Dense(B)], multiplicities=[2, 3])
        L = lower_cholesky(op)

        assert isinstance(L, BlockDiag)
        assert L.multiplicities == [2, 3]
        assert isinstance(L.operators[0], Triangular)
        assert isinstance(L.operators[1], Triangular)

        # Check the Cholesky property
        L_dense = L.to_dense()
        B_dense = op.to_dense()
        assert jnp.allclose(L_dense @ L_dense.T, B_dense, atol=1e-6)


class TestSolve:
    """Tests for the solve function."""

    def test_identity_solve(self):
        """Test solving with identity matrix."""
        op = Identity(3)
        b = jnp.array([1.0, 2.0, 3.0])
        x = solve(op, b)

        assert jnp.allclose(x, b)

    def test_diagonal_solve(self):
        """Test solving with diagonal matrix."""
        diag_values = jnp.array([2.0, 4.0, 5.0])
        op = Diagonal(diag_values)
        b = jnp.array([2.0, 8.0, 10.0])
        x = solve(op, b)

        expected = b / diag_values
        assert jnp.allclose(x, expected)

    def test_dense_solve(self):
        """Test solving with dense matrix."""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        op = Dense(A)
        b = jnp.array([1.0, 2.0])
        x = solve(op, b)

        # Check A @ x = b
        assert jnp.allclose(A @ x, b)

    def test_triangular_solve(self):
        """Test solving with triangular matrix."""
        L = jnp.array([[2.0, 0.0], [1.0, 3.0]])
        op = Triangular(L, lower=True)
        b = jnp.array([2.0, 7.0])
        x = solve(op, b)

        # Check L @ x = b
        assert jnp.allclose(L @ x, b)

    def test_solve_matrix_rhs(self):
        """Test solving with matrix right-hand side."""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        op = Dense(A)
        B = jnp.array([[1.0, 2.0], [2.0, 4.0]])
        X = solve(op, B)

        # Check A @ X = B
        assert jnp.allclose(A @ X, B)


class TestLogdet:
    """Tests for the logdet function."""

    def test_identity_logdet(self):
        """Test log-determinant of identity matrix."""
        op = Identity(3)
        ld = logdet(op)

        assert jnp.allclose(ld, 0.0)  # log(det(I)) = log(1) = 0

    def test_diagonal_logdet(self):
        """Test log-determinant of diagonal matrix."""
        diag_values = jnp.array([2.0, 3.0, 4.0])
        op = Diagonal(diag_values)
        ld = logdet(op)

        expected = jnp.sum(jnp.log(diag_values))
        assert jnp.allclose(ld, expected)

    def test_dense_logdet(self):
        """Test log-determinant of dense matrix."""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        op = Dense(A)
        ld = logdet(op)

        _, expected = jnp.linalg.slogdet(A)
        assert jnp.allclose(ld, expected)

    def test_triangular_logdet(self):
        """Test log-determinant of triangular matrix."""
        L = jnp.array([[2.0, 0.0], [1.0, 3.0]])
        op = Triangular(L, lower=True)
        ld = logdet(op)

        expected = jnp.sum(jnp.log(jnp.diag(L)))
        assert jnp.allclose(ld, expected)

    def test_kronecker_logdet(self):
        """Test log-determinant of Kronecker product."""
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        B = jnp.array([[4.0, 0.0], [0.0, 5.0]])

        op = Kronecker([Dense(A), Dense(B)])
        ld = logdet(op)

        # For Kronecker: log(det(A ⊗ B)) = n*log(det(A)) + m*log(det(B))
        # where B is n×n and A is m×m
        _, ld_A = jnp.linalg.slogdet(A)
        _, ld_B = jnp.linalg.slogdet(B)
        expected = 2 * ld_A + 2 * ld_B  # Both are 2×2
        assert jnp.allclose(ld, expected)

    def test_block_diag_logdet(self):
        """Test log-determinant of block diagonal matrix."""
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        B = jnp.array([[4.0]])

        op = BlockDiag([Dense(A), Dense(B)], multiplicities=[2, 1])
        ld = logdet(op)

        _, ld_A = jnp.linalg.slogdet(A)
        _, ld_B = jnp.linalg.slogdet(B)
        expected = 2 * ld_A + 1 * ld_B  # Multiplicities
        assert jnp.allclose(ld, expected)


class TestDiag:
    """Tests for the diag function."""

    def test_identity_diag(self):
        """Test diagonal extraction from identity matrix."""
        op = Identity(4)
        d = diag(op)

        expected = jnp.ones(4)
        assert jnp.allclose(d, expected)

    def test_diagonal_diag(self):
        """Test diagonal extraction from diagonal matrix."""
        diag_values = jnp.array([1.0, 2.0, 3.0])
        op = Diagonal(diag_values)
        d = diag(op)

        assert jnp.allclose(d, diag_values)

    def test_dense_diag(self):
        """Test diagonal extraction from dense matrix."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        op = Dense(A)
        d = diag(op)

        expected = jnp.array([1.0, 4.0])
        assert jnp.allclose(d, expected)

    def test_triangular_diag(self):
        """Test diagonal extraction from triangular matrix."""
        L = jnp.array([[2.0, 0.0], [1.0, 3.0]])
        op = Triangular(L, lower=True)
        d = diag(op)

        expected = jnp.array([2.0, 3.0])
        assert jnp.allclose(d, expected)

    def test_kronecker_diag(self):
        """Test diagonal extraction from Kronecker product."""
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[3.0, 0.0], [0.0, 4.0]])

        op = Kronecker([Dense(A), Dense(B)])
        d = diag(op)

        # diag(A ⊗ B) = kron(diag(A), diag(B))
        expected = jnp.kron(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        assert jnp.allclose(d, expected)

    def test_block_diag_diag(self):
        """Test diagonal extraction from block diagonal matrix."""
        A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        B = jnp.array([[3.0]])

        op = BlockDiag([Dense(A), Dense(B)], multiplicities=[1, 2])
        d = diag(op)

        expected = jnp.array([1.0, 2.0, 3.0, 3.0])  # B is repeated twice
        assert jnp.allclose(d, expected)


class TestAddJitter:
    """Tests for the add_jitter utility function."""

    def test_square_matrix_default_jitter(self):
        """Test adding default jitter to a square matrix."""
        key = jr.key(123)
        matrix = jr.normal(key, shape=(3, 3))

        jittered = add_jitter(matrix)
        expected = matrix + jnp.eye(3) * 1e-6

        assert jnp.allclose(jittered, expected)
        assert jittered.shape == matrix.shape

    def test_square_matrix_custom_jitter(self):
        """Test adding custom jitter value to a square matrix."""
        matrix = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        jitter_val = 0.01

        jittered = add_jitter(matrix, jitter=jitter_val)
        expected = matrix + jnp.eye(2) * jitter_val

        assert jnp.allclose(jittered, expected)
        # Check diagonal elements specifically
        assert jnp.allclose(jnp.diag(jittered), jnp.array([1.01, 1.01]))

    def test_1x1_matrix(self):
        """Test edge case with 1x1 matrix."""
        matrix = jnp.array([[5.0]])
        jitter_val = 0.1

        jittered = add_jitter(matrix, jitter=jitter_val)
        expected = jnp.array([[5.1]])

        assert jnp.allclose(jittered, expected)

    def test_large_matrix_performance(self):
        """Test with a large matrix for performance."""
        key = jr.key(456)
        matrix = jr.normal(key, shape=(100, 100))
        jitter_val = 1e-4

        jittered = add_jitter(matrix, jitter=jitter_val)

        # Check that only diagonal changed
        off_diagonal_unchanged = jnp.allclose(
            jittered - jnp.diag(jnp.diag(jittered)), matrix - jnp.diag(jnp.diag(matrix))
        )
        assert off_diagonal_unchanged

        # Check diagonal changed correctly
        diagonal_diff = jnp.diag(jittered) - jnp.diag(matrix)
        assert jnp.allclose(diagonal_diff, jitter_val)

    def test_non_square_matrix_raises_error(self):
        """Test that non-square matrix raises ValueError."""
        matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError, match="Expected square matrix"):
            add_jitter(matrix)

    def test_non_2d_array_raises_error(self):
        """Test that non-2D array raises ValueError."""
        array_1d = jnp.array([1.0, 2.0, 3.0])
        array_3d = jnp.ones((2, 2, 2))

        with pytest.raises(ValueError, match="Expected 2D matrix"):
            add_jitter(array_1d)

        with pytest.raises(ValueError, match="Expected 2D matrix"):
            add_jitter(array_3d)
