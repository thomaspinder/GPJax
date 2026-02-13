"""First-order decomposition utilities for the Orthogonal Additive Kernel.

Provides functions to rank features by their first-order importance and
to compute posterior predictions for individual first-order components.

Reference:
    Lu, X., Boukouvalas, A., & Hensman, J. (2022).
    Additive Gaussian Processes Revisited. ICML.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.typing import Array

if tp.TYPE_CHECKING:
    from gpjax.kernels.additive.oak import OrthogonalAdditiveKernel


def _solve_alpha(
    kernel: OrthogonalAdditiveKernel,
    x_train: Float[Array, "N D"],
    y_train: Float[Array, "N 1"],
    noise_variance: float,
) -> tuple[Float[Array, " N"], Float[Array, "N N"]]:
    r"""Build noisy kernel matrix and solve for alpha.

    Computes :math:`\boldsymbol{\alpha} = (K + \sigma_n^2 I)^{-1} \mathbf{y}`.

    Args:
        kernel: A fitted OrthogonalAdditiveKernel.
        x_train: Training inputs of shape (N, D).
        y_train: Training targets of shape (N, 1).
        noise_variance: Observation noise variance.

    Returns:
        Tuple of (alpha, K_noisy) where alpha has shape (N,) and
        K_noisy has shape (N, N).
    """
    N = x_train.shape[0]
    K = kernel.gram(x_train).to_dense()
    K_noisy = K + noise_variance * jnp.eye(N)
    alpha = jnp.linalg.solve(K_noisy, y_train.squeeze())
    return alpha, K_noisy


def rank_first_order(
    kernel: OrthogonalAdditiveKernel,
    x_train: Float[Array, "N D"],
    y_train: Float[Array, "N 1"],
    noise_variance: float,
) -> Float[Array, " D"]:
    r"""Unnormalised per-feature first-order importance scores.

    For each dimension d, computes
    :math:`\sigma_1^2 \, \boldsymbol{\alpha}^\top M_d \boldsymbol{\alpha}`
    where :math:`M_d` is the per-dimension integral matrix and
    :math:`\sigma_1^2` is the first-order variance.

    Args:
        kernel: A fitted OrthogonalAdditiveKernel.
        x_train: Training inputs of shape (N, D).
        y_train: Training targets of shape (N, 1).
        noise_variance: Observation noise variance.

    Returns:
        Array of shape (D,) with unnormalised importance scores.
    """
    from gpjax.kernels.additive.sobol import _sobol_integral_matrix

    alpha, _ = _solve_alpha(kernel, x_train, y_train, noise_variance)

    lengthscales = kernel._lengthscales
    variances = kernel._variances
    ov = kernel.order_variances[...]

    M_stack = jax.vmap(_sobol_integral_matrix)(
        x_train.T, lengthscales, variances
    )  # (D, N, N)
    scores = jax.vmap(lambda M_d: alpha @ M_d @ alpha)(M_stack)
    return jnp.square(ov[1]) * scores


def predict_first_order(
    kernel: OrthogonalAdditiveKernel,
    x_train: Float[Array, "N D"],
    y_train: Float[Array, "N 1"],
    noise_variance: float,
    dim: int,
    x_grid: Float[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    r"""Posterior mean and variance for a single first-order component.

    Evaluates the posterior of the first-order GP component for dimension
    ``dim`` on the 1-D grid ``x_grid``.

    Args:
        kernel: A fitted OrthogonalAdditiveKernel.
        x_train: Training inputs of shape (N, D).
        y_train: Training targets of shape (N, 1).
        noise_variance: Observation noise variance.
        dim: Feature dimension index.
        x_grid: 1-D evaluation grid of shape (M,).

    Returns:
        Tuple of (mean, variance) each of shape (M,).  Variance is
        clipped to non-negative values.
    """
    from gpjax.kernels.additive.oak import _constrained_se_kernel

    alpha, K_noisy = _solve_alpha(kernel, x_train, y_train, noise_variance)

    ls_d = kernel._lengthscales[dim]
    var_d = kernel._variances[dim]
    ov = kernel.order_variances[...]

    # K_star: (M, N) constrained kernel between grid and training points
    K_star = jax.vmap(
        jax.vmap(
            lambda xg, xt: _constrained_se_kernel(xg, xt, ls_d, var_d),
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )(x_grid, x_train[:, dim])
    K_star = ov[1] * K_star

    # Posterior mean
    mean = K_star @ alpha

    # Posterior variance: diag(K_star K_noisy^{-1} K_star^T)
    K_star_Kinv = jnp.linalg.solve(K_noisy, K_star.T).T  # (M, N)
    k_diag = jax.vmap(lambda x: _constrained_se_kernel(x, x, ls_d, var_d))(x_grid)
    k_diag = ov[1] * k_diag
    variance = k_diag - jnp.sum(K_star_Kinv * K_star, axis=1)
    variance = jnp.maximum(variance, 0.0)

    return mean, variance
