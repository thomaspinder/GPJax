from gpjax.kernels.computations import (
    ConstantDiagonalKernelComputation,
    DiagonalKernelComputation,
)
from gpjax.kernels.stationary import RBF
import jax.numpy as jnp
import lineax as lx


def test_dense_computation():
    """Default DenseKernelComputation produces a PSD matrix linear operator."""
    kernel = RBF()
    x = jnp.linspace(-3.0, 3.0, 5).reshape(-1, 1)

    dense_linop = kernel.gram(x)
    dense_matrix = dense_linop.as_matrix()
    dense_diagonals = jnp.diag(dense_matrix)

    assert isinstance(dense_linop, lx.AbstractLinearOperator)
    assert lx.is_positive_semidefinite(dense_linop)
    assert dense_matrix.shape == (5, 5)
    assert jnp.all(dense_diagonals > 0.0)


def test_diagonal_computation():
    """DiagonalKernelComputation produces a PSD diagonal linear operator."""
    kernel = RBF(compute_engine=DiagonalKernelComputation())
    x = jnp.linspace(-3.0, 3.0, 5).reshape(-1, 1)

    # Also compute the dense version for comparison
    dense_kernel = RBF()
    dense_matrix = dense_kernel.gram(x).as_matrix()
    dense_diagonals = jnp.diag(dense_matrix)

    diagonal_linop = kernel.gram(x)
    diagonal_matrix = diagonal_linop.as_matrix()
    diag_entries = jnp.diag(diagonal_matrix)

    assert isinstance(diagonal_linop, lx.AbstractLinearOperator)
    assert lx.is_positive_semidefinite(diagonal_linop)

    # The diagonal entries should be the same as the dense matrix
    assert jnp.allclose(diag_entries, dense_diagonals)

    # All the off diagonal entries should be zero
    assert jnp.allclose(diagonal_matrix - jnp.diag(diag_entries), 0.0)


def test_constant_diagonal_computation():
    """ConstantDiagonalKernelComputation produces a PSD diagonal operator with equal entries."""
    kernel = RBF(compute_engine=ConstantDiagonalKernelComputation())
    x = jnp.linspace(-3.0, 3.0, 5).reshape(-1, 1)

    constant_diagonal_linop = kernel.gram(x)
    constant_diagonal_matrix = constant_diagonal_linop.as_matrix()
    constant_entries = jnp.diag(constant_diagonal_matrix)

    assert lx.is_positive_semidefinite(constant_diagonal_linop)

    # Assert all the diagonal entries are the same
    assert jnp.allclose(constant_entries, constant_entries[0])

    # All the off diagonal entries should be zero
    assert jnp.allclose(constant_diagonal_matrix - jnp.diag(constant_entries), 0.0)
