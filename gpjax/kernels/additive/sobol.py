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
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.typing import Array

if tp.TYPE_CHECKING:
    from gpjax.kernels.additive.oak import OrthogonalAdditiveKernel


def _projection_coefficient(
    sigma_sq: Float[Array, ""],
    ell_sq: Float[Array, ""],
    lengthscale: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Shared projection coefficient for k_hat terms.

    .. math::
        c = \sigma^2 \, \ell \, \sqrt{\ell^2 + 2} \,/\, (\ell^2 + 1)

    This coefficient appears in all four integral terms.
    """
    return sigma_sq * lengthscale * jnp.sqrt(ell_sq + 2.0) / (ell_sq + 1.0)


def _sobol_integral_matrix(
    x_train: Float[Array, " N"],
    lengthscale: Float[Array, ""],
    variance: Float[Array, ""],
) -> Float[Array, "N N"]:
    r"""Compute the integral matrix for a single dimension's Sobol index.

    Computes \int k_tilde(x, X) k_tilde(x, X)^T dp(x) analytically
    for the constrained SE kernel with standard normal input density.

    This decomposes into four terms (Appendix G.1 of OAK paper):
      Term 1:  \int p(x) k(x,a) k(x,b) dx        (Eq. 44)
    - Term 2:  \int p(x) k(x,a) k_hat(x,b) dx     (Eq. 45)
    - Term 3:  \int p(x) k_hat(x,a) k(x,b) dx     (Eq. 46)
    + Term 4:  \int p(x) k_hat(x,a) k_hat(x,b) dx  (Eq. 47)

    All four terms are computed in closed form via broadcasting (no loops).
    """
    num_points = x_train.shape[0]
    sigma_sq = variance
    ell_sq = jnp.square(lengthscale)

    # Guard against numerically inactive dimensions
    _EPS = 1e-6
    inactive = (lengthscale < _EPS) | (sigma_sq < _EPS)

    a = x_train[:, None]  # (N, 1)  -- left training point
    b = x_train[None, :]  # (1, N)  -- right training point

    # --- Term 1 (Eq. 44): integral of k(x,a) * k(x,b) under p(x) ---
    # = sigma^4 * l / sqrt(2 + l^2)
    #   * exp(-(a-b)^2 / (4l^2) - (a+b)^2 / (4(2 + l^2)))
    term1_coeff = sigma_sq**2 * lengthscale / jnp.sqrt(2.0 + ell_sq)
    term1 = term1_coeff * jnp.exp(
        -jnp.square(a - b) / (4.0 * ell_sq) - jnp.square(a + b) / (4.0 * (2.0 + ell_sq))
    )

    # Shared projection coefficient for terms 2-4
    proj_coeff = _projection_coefficient(sigma_sq, ell_sq, lengthscale)

    # Shared precision for the completing-the-square step in terms 2 & 3:
    # M = 1 + 1/l^2 + 1/(l^2 + 1)
    precision = 1.0 + 1.0 / ell_sq + 1.0 / (ell_sq + 1.0)

    # --- Term 2 (Eq. 45): integral of k(x,a) * k_hat(x,b) under p(x) ---
    completed_mean_a = (a / ell_sq) / precision
    quadratic_residual_a = (
        jnp.square(a) / ell_sq - jnp.square(completed_mean_a) * precision
    )
    term2 = (
        sigma_sq
        * proj_coeff
        * jnp.exp(-jnp.square(b) / (2.0 * (ell_sq + 1.0)))
        * jnp.exp(-quadratic_residual_a / 2.0)
        / jnp.sqrt(precision)
    )

    # --- Term 3 (Eq. 46): symmetric to Term 2 with a <-> b ---
    completed_mean_b = (b / ell_sq) / precision
    quadratic_residual_b = (
        jnp.square(b) / ell_sq - jnp.square(completed_mean_b) * precision
    )
    term3 = (
        sigma_sq
        * proj_coeff
        * jnp.exp(-jnp.square(a) / (2.0 * (ell_sq + 1.0)))
        * jnp.exp(-quadratic_residual_b / 2.0)
        / jnp.sqrt(precision)
    )

    # --- Term 4 (Eq. 47): integral of k_hat(x,a) * k_hat(x,b) under p(x) ---
    term4 = (
        proj_coeff**2
        * jnp.exp(-(jnp.square(a) + jnp.square(b)) / (2.0 * (ell_sq + 1.0)))
        * jnp.sqrt((ell_sq + 1.0) / (ell_sq + 3.0))
    )

    result = term1 - term2 - term3 + term4
    return jnp.where(inactive, jnp.zeros((num_points, num_points)), result)


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
    _, num_points, _ = matrices.shape

    # Power sums: S[k, i, j] = sum_{d=0}^{D-1} matrices[d, i, j]^{k+1}
    exponents = jnp.arange(1, max_order + 1)[:, None, None, None]
    power_sums = jnp.sum(matrices[None, :, :, :] ** exponents, axis=1)

    signs = (-1.0) ** jnp.arange(max_order)
    signed_power_sums = signs[:, None, None] * power_sums

    elem_sym = jnp.zeros((max_order + 1, num_points, num_points))
    elem_sym = elem_sym.at[0].set(jnp.ones((num_points, num_points)))

    def _recursion_step(order, elem_sym):
        k_indices = jnp.arange(max_order)
        lookback_indices = (order - 1 - k_indices).clip(0)
        mask = (k_indices < order)[:, None, None]
        previous_values = jnp.where(mask, elem_sym[lookback_indices], 0.0)
        value = jnp.sum(previous_values * signed_power_sums, axis=0) / order
        return elem_sym.at[order].set(value)

    elem_sym = lax.fori_loop(1, max_order + 1, _recursion_step, elem_sym)
    return elem_sym


def sobol_indices(
    kernel: OrthogonalAdditiveKernel,
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
    num_points = x_train.shape[0]
    max_order = kernel.max_order
    order_variances = kernel.order_variances[...]

    # Solve alpha = (K + sigma_n^2 I)^{-1} y
    gram_matrix = kernel.gram(x_train).to_dense()
    noisy_gram = gram_matrix + noise_variance * jnp.eye(num_points)
    alpha = jnp.linalg.solve(noisy_gram, y_train.squeeze())

    # Per-dimension integral matrices via vmap
    lengthscales = kernel._lengthscales
    variances = kernel._variances
    integral_matrices = jax.vmap(_sobol_integral_matrix)(
        x_train.T, lengthscales, variances
    )  # (D, N, N)

    # Matrix-level Newton-Girard: E_d[i,j] = sum over size-d subsets
    # of element-wise products of integral matrices
    elem_sym = _newton_girard_matrices(integral_matrices, max_order)

    # Sobol index for order d: V_d = sigma^2_d * alpha^T E_d alpha
    # Skip E[0] (the offset term); use E[1:] for orders 1..max_order
    elem_sym_orders = elem_sym[1:]
    variance_orders = order_variances[1:]

    # Vectorised quadratic forms: alpha^T E_d alpha for each order d
    quadratic_forms = jax.vmap(lambda E_d: alpha @ E_d @ alpha)(elem_sym_orders)
    raw_sobol = jnp.square(variance_orders) * quadratic_forms

    # Normalise to sum to 1
    total = jnp.sum(raw_sobol)
    return jnp.where(total > 0, raw_sobol / total, raw_sobol)
