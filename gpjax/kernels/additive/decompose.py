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
        Tuple of (alpha, noisy_gram) where alpha has shape (N,) and
        noisy_gram has shape (N, N).
    """
    num_points = x_train.shape[0]
    gram_matrix = kernel.gram(x_train).to_dense()
    noisy_gram = gram_matrix + noise_variance * jnp.eye(num_points)
    alpha = jnp.linalg.solve(noisy_gram, y_train.squeeze())
    return alpha, noisy_gram


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
    order_variances = kernel.order_variances[...]

    integral_matrices = jax.vmap(_sobol_integral_matrix)(
        x_train.T, lengthscales, variances
    )  # (D, N, N)

    scores = jax.vmap(lambda M_d: alpha @ M_d @ alpha)(integral_matrices)
    return jnp.square(order_variances[1]) * scores


def _build_first_order_cross_covariance(
    x_grid: Float[Array, " M"],
    x_train_dim: Float[Array, " N"],
    lengthscale_dim: Float[Array, ""],
    variance_dim: Float[Array, ""],
    first_order_variance: Float[Array, ""],
) -> Float[Array, "M N"]:
    """Build the cross-covariance matrix between grid and training points.

    Computes K_star[i, j] = sigma^2_1 * k_tilde(x_grid[i], x_train[j])
    for a single dimension.
    """
    from gpjax.kernels.additive.oak import _constrained_se_kernel

    K_star = jax.vmap(
        jax.vmap(
            lambda grid_point, train_point: _constrained_se_kernel(
                grid_point, train_point, lengthscale_dim, variance_dim
            ),
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )(x_grid, x_train_dim)
    return first_order_variance * K_star


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

    alpha, noisy_gram = _solve_alpha(kernel, x_train, y_train, noise_variance)

    lengthscale_dim = kernel._lengthscales[dim]
    variance_dim = kernel._variances[dim]
    first_order_variance = kernel.order_variances[...][1]

    # K_star: (M, N) cross-covariance between grid and training points
    K_star = _build_first_order_cross_covariance(
        x_grid, x_train[:, dim], lengthscale_dim, variance_dim, first_order_variance
    )

    # Posterior mean: K_star @ alpha
    mean = K_star @ alpha

    # Posterior variance: diag(K_star @ K_noisy^{-1} @ K_star^T)
    K_star_solved = jnp.linalg.solve(noisy_gram, K_star.T).T  # (M, N)
    prior_diag = jax.vmap(
        lambda grid_point: _constrained_se_kernel(
            grid_point, grid_point, lengthscale_dim, variance_dim
        )
    )(x_grid)
    prior_diag = first_order_variance * prior_diag
    variance = prior_diag - jnp.sum(K_star_solved * K_star, axis=1)
    variance = jnp.maximum(variance, 0.0)

    return mean, variance
