r"""Woodbury identity helpers for efficient low-rank + diagonal solves.

When a kernel is approximated by a low-rank factorisation K \approx W W^T
(e.g. via HSGP or RFF), the marginal covariance becomes

    \Sigma = W W^T + D,   where  D = diag(noise),  W is (N, m),  m << N.

Naively inverting \Sigma costs O(N^3).  The *Woodbury matrix identity* reduces
this to O(N m^2 + m^3) by working with the small (m x m) *capacitance matrix*

    A = I_m + W^T D^{-1} W.

The three operations exposed here --- solve, log-determinant, and quadratic
form --- are the building blocks for the GP marginal likelihood and posterior
prediction when the kernel has low-rank structure.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float

from gpjax.typing import Array, ScalarFloat


def _capacitance_cholesky(
    W: Float[Array, "N m"],
    noise_inv: Float[Array, " N"],
) -> Float[Array, "m m"]:
    r"""Cholesky factor of the capacitance matrix A = I_m + W^T D^{-1} W.

    This is the shared computation underlying all Woodbury operations.

    Args:
        W: Factor matrix, shape (N, m).
        noise_inv: Element-wise reciprocal of the diagonal noise, shape (N,).

    Returns:
        Lower-triangular Cholesky factor L_A such that A = L_A L_A^T.
    """
    m = W.shape[1]
    Dinv_W = noise_inv[:, None] * W
    A = jnp.eye(m) + W.T @ Dinv_W
    return jnp.linalg.cholesky(A)


def woodbury_solve(
    W: Float[Array, "N m"],
    noise: Float[Array, " N"],
    b: Float[Array, "N ..."],
) -> Float[Array, "N ..."]:
    r"""Solve (W W^T + D) x = b via the Woodbury identity.

    .. math::

        x = D^{-1} b  -  D^{-1} W \, A^{-1} \, W^T D^{-1} b

    where D = diag(noise) and A = I_m + W^T D^{-1} W.

    Args:
        W: Factor matrix, shape (N, m).
        noise: Diagonal noise vector, shape (N,).
        b: Right-hand side, shape (N,) or (N, K).

    Returns:
        Solution x with same shape as b.
    """
    noise_inv = 1.0 / noise
    Dinv_W = noise_inv[:, None] * W
    Dinv_b = noise_inv * b if b.ndim == 1 else noise_inv[:, None] * b

    L_A = _capacitance_cholesky(W, noise_inv)

    WtDinv_b = Dinv_W.T @ b
    forward = jsp.linalg.solve_triangular(L_A, WtDinv_b, lower=True)
    Ainv_WtDinv_b = jsp.linalg.solve_triangular(L_A.T, forward, lower=False)

    return Dinv_b - Dinv_W @ Ainv_WtDinv_b


def woodbury_logdet(
    W: Float[Array, "N m"],
    noise: Float[Array, " N"],
) -> ScalarFloat:
    r"""Log-determinant of W W^T + D via the matrix determinant lemma.

    .. math::

        \log|\Sigma| = \log|A| + \sum_i \log(\text{noise}_i)

    where A = I_m + W^T D^{-1} W.

    Args:
        W: Factor matrix, shape (N, m).
        noise: Diagonal noise vector, shape (N,).

    Returns:
        Log-determinant as a scalar.
    """
    noise_inv = 1.0 / noise
    L_A = _capacitance_cholesky(W, noise_inv)
    logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L_A)))
    return logdet_A + jnp.sum(jnp.log(noise))


def woodbury_quad(
    W: Float[Array, "N m"],
    noise: Float[Array, " N"],
    diff: Float[Array, " N"],
) -> ScalarFloat:
    r"""Quadratic form diff^T \Sigma^{-1} diff where \Sigma = W W^T + D.

    Args:
        W: Factor matrix, shape (N, m).
        noise: Diagonal noise vector, shape (N,).
        diff: Vector, shape (N,).

    Returns:
        Quadratic form as a scalar.
    """
    solved = woodbury_solve(W, noise, diff)
    return diff @ solved
