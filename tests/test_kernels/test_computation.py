from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    ConstantDiagonalKernelComputation,
    DiagonalKernelComputation,
)
from gpjax.kernels.nonstationary import (
    Linear,
    Polynomial,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
)
from gpjax.linalg import PSD
from gpjax.linalg.operators import (
    Dense,
    Diagonal,
)
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern12(),
        Matern32(),
        Matern52(),
        RationalQuadratic(),
        PoweredExponential(),
        Periodic(),
        Linear(),
        Polynomial(),
    ],
)
def test_change_computation(kernel: AbstractKernel):
    x = jnp.linspace(-3.0, 3.0, 5).reshape(-1, 1)

    # The default computation is DenseKernelComputation
    dense_linop = kernel.gram(x)
    dense_matrix = dense_linop.to_dense()
    dense_diagonals = jnp.diag(dense_matrix)

    assert isinstance(dense_linop, Dense)
    assert PSD in dense_linop.annotations

    # Let's now change the computation to DiagonalKernelComputation
    kernel.compute_engine = DiagonalKernelComputation()
    diagonal_linop = kernel.gram(x)
    diagonal_matrix = diagonal_linop.to_dense()
    diag_entries = jnp.diag(diagonal_matrix)

    assert isinstance(diagonal_linop, Diagonal)
    assert PSD in diagonal_linop.annotations

    # The diagonal entries should be the same as the dense matrix
    assert jnp.allclose(diag_entries, dense_diagonals)

    # All the off diagonal entries should be zero
    assert jnp.allclose(diagonal_matrix - jnp.diag(diag_entries), 0.0)

    # Let's now change the computation to ConstantDiagonalKernelComputation
    kernel.compute_engine = ConstantDiagonalKernelComputation()
    constant_diagonal_linop = kernel.gram(x)
    constant_diagonal_matrix = constant_diagonal_linop.to_dense()
    constant_entries = jnp.diag(constant_diagonal_matrix)

    assert PSD in constant_diagonal_linop.annotations

    # Assert all the diagonal entries are the same
    assert jnp.allclose(constant_entries, constant_entries[0])

    # All the off diagonal entries should be zero
    assert jnp.allclose(constant_diagonal_matrix - jnp.diag(constant_entries), 0.0)
