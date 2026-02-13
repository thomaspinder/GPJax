"""Sobol indices for the Orthogonal Additive Kernel.

Computes analytic Sobol indices per interaction order for the constrained
SE kernel under standard normal input density.

Reference:
    Lu, X., Boukouvalas, A., & Hensman, J. (2022).
    Additive Gaussian Processes Revisited. ICML. (Eq. 14, Appendix G.1)
"""

from __future__ import annotations

import typing as tp

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.typing import Array

if tp.TYPE_CHECKING:
    from gpjax.kernels.additive.oak import OrthogonalAdditiveKernel


def _sobol_integral_matrix(
    x_train: Float[Array, " N"],
    lengthscale: Float[Array, ""],
    variance: Float[Array, ""],
) -> Float[Array, "N N"]:
    r"""Compute the integral matrix for a single dimension's Sobol index.

    Computes \int k_tilde(x, X) k_tilde(x, X)^T dp(x) analytically
    for the constrained SE kernel with standard normal input density.

    This decomposes into four terms (Appendix G.1 of OAK paper):
    \int p(x) k(x,a) k(x,b) dx  (Eq. 44)
    - \int p(x) k(x,a) k_hat(x,b) dx  (Eq. 45)
    - \int p(x) k_hat(x,a) k(x,b) dx  (Eq. 46)
    + \int p(x) k_hat(x,a) k_hat(x,b) dx  (Eq. 47)

    All four terms are computed in closed form via broadcasting (no loops).
    """
    l = lengthscale
    l_sq = jnp.square(l)
    sigma_sq = variance

    a = x_train[:, None]  # (N, 1)
    b = x_train[None, :]  # (1, N)

    # --- Term 1 (Eq. 44): \int p(x) k(x,a) k(x,b) dx ---
    term1_coeff = sigma_sq**2 * l / jnp.sqrt(2.0 + l_sq)
    term1 = term1_coeff * jnp.exp(
        -jnp.square(a - b) / (4.0 * l_sq)
        - jnp.square(a + b) / (4.0 * (2.0 + l_sq))
    )

    # --- Projection coefficient for k_hat ---
    proj_coeff = sigma_sq * l * jnp.sqrt(l_sq + 2.0) / (l_sq + 1.0)

    # --- Term 2 (Eq. 45): \int p(x) k(x,a) k_hat(x,b) dx ---
    M2 = 1.0 + 1.0 / l_sq + 1.0 / (l_sq + 1.0)
    c2 = (a / l_sq) / M2
    C2 = jnp.square(a) / l_sq - jnp.square(c2) * M2
    term2 = (
        sigma_sq
        * proj_coeff
        * jnp.exp(-jnp.square(b) / (2.0 * (l_sq + 1.0)))
        * jnp.exp(-C2 / 2.0)
        / jnp.sqrt(M2)
    )

    # --- Term 3 (Eq. 46): symmetric to term 2 with a <-> b ---
    c3 = (b / l_sq) / M2
    C3 = jnp.square(b) / l_sq - jnp.square(c3) * M2
    term3 = (
        sigma_sq
        * proj_coeff
        * jnp.exp(-jnp.square(a) / (2.0 * (l_sq + 1.0)))
        * jnp.exp(-C3 / 2.0)
        / jnp.sqrt(M2)
    )

    # --- Term 4 (Eq. 47): \int p(x) k_hat(x,a) k_hat(x,b) dx ---
    term4 = (
        proj_coeff**2
        * jnp.exp(-(jnp.square(a) + jnp.square(b)) / (2.0 * (l_sq + 1.0)))
        * jnp.sqrt((l_sq + 1.0) / (l_sq + 3.0))
    )

    return term1 - term2 - term3 + term4


def _newton_girard_matrices(
    matrices: Float[Array, "D N N"],
    max_order: int,
) -> Float[Array, "D_tilde_plus_1 N N"]:
    """Element-wise Newton-Girard on a stack of (D, N, N) matrices.

    Computes the elementary symmetric polynomials E_0, ..., E_{max_order}
    where E_d[i,j] = sum over all size-d subsets u of prod_{k in u} M_k[i,j].

    This is the matrix-level analogue of _newton_girard in oak.py, using
    lax.fori_loop for JAX compatibility.
    """
    D, N, _ = matrices.shape

    # Power sums: S[k, i, j] = sum_{d=0}^{D-1} matrices[d, i, j]^{k+1}
    powers = jnp.arange(1, max_order + 1)[
        :, None, None, None
    ]  # (max_order, 1, 1, 1)
    S = jnp.sum(
        matrices[None, :, :, :] ** powers, axis=1
    )  # (max_order, N, N)

    signs = (-1.0) ** jnp.arange(max_order)
    signed_S = signs[:, None, None] * S  # (max_order, N, N)

    E = jnp.zeros((max_order + 1, N, N))
    E = E.at[0].set(jnp.ones((N, N)))

    def body_fn(ell, E):
        k = jnp.arange(max_order)
        e_idx = (ell - 1 - k).clip(0)
        mask = (k < ell)[:, None, None]  # (max_order, 1, 1)
        e_vals = jnp.where(mask, E[e_idx], 0.0)  # (max_order, N, N)
        val = jnp.sum(e_vals * signed_S, axis=0) / ell  # (N, N)
        return E.at[ell].set(val)

    E = lax.fori_loop(1, max_order + 1, body_fn, E)
    return E


def sobol_indices(
    kernel: "OrthogonalAdditiveKernel",
    x_train: Float[Array, "N D"],
    y_train: Float[Array, "N 1"],
    noise_variance: float,
) -> Float[Array, " D_tilde"]:
    r"""Compute normalized Sobol indices per interaction order.

    The Sobol index for interaction order d measures what fraction of the
    posterior variance is explained by d-th order interactions. Indices
    are normalized to sum to 1.

    Uses vmap for per-dimension integral matrices and matrix-level
    Newton-Girard to sum over all subsets of each order (no Python loops
    in the computation path).

    Args:
        kernel: A fitted OrthogonalAdditiveKernel.
        x_train: Training inputs of shape (N, D).
        y_train: Training targets of shape (N, 1).
        noise_variance: Observation noise variance.

    Returns:
        Array of shape (max_order,) with normalized Sobol indices
        for orders 1 through max_order.
    """
    N, D = x_train.shape
    max_order = kernel.max_order
    ov = kernel.order_variances[...]

    # Compute K(X, X)^{-1} y
    K = kernel.gram(x_train).to_dense()
    K_noisy = K + noise_variance * jnp.eye(N)
    alpha = jnp.linalg.solve(K_noisy, y_train.squeeze())  # (N,)

    # Compute per-dimension integral matrices via vmap
    lengthscales = kernel._lengthscales  # (D,)
    variances = kernel._variances  # (D,)
    M_stack = jax.vmap(_sobol_integral_matrix)(
        x_train.T, lengthscales, variances
    )  # (D, N, N)

    # Matrix-level Newton-Girard: E_d[i,j] = sum over size-d subsets
    # of element-wise products of integral matrices
    E = _newton_girard_matrices(M_stack, max_order)  # (max_order+1, N, N)

    # Sobol index for order d: V_d = ov[d]^2 * alpha^T E_d alpha
    # Skip E[0] (the offset term), use E[1:] for orders 1..max_order
    E_orders = E[1:]  # (max_order, N, N)
    ov_orders = ov[1:]  # (max_order,)

    # Vectorized quadratic forms: alpha^T E_d alpha for each order d
    quad_forms = jax.vmap(lambda E_d: alpha @ E_d @ alpha)(
        E_orders
    )  # (max_order,)
    raw_sobol = jnp.square(ov_orders) * quad_forms

    # Normalize to sum to 1
    total = jnp.sum(raw_sobol)
    return jnp.where(total > 0, raw_sobol / total, raw_sobol)
