"""Orthogonal Additive Kernel (OAK).

Reference:
    Lu, X., Boukouvalas, A., & Hensman, J. (2022).
    Additive Gaussian Processes Revisited. ICML.
"""

import jax.numpy as jnp
from jaxtyping import Float

from gpjax.typing import Array, ScalarFloat


def _constrained_se_kernel(
    x: Float[Array, ""],
    y: Float[Array, ""],
    lengthscale: Float[Array, ""],
    variance: Float[Array, ""],
) -> ScalarFloat:
    """Compute the constrained SE kernel under standard normal input density.

    Given base SE kernel k(x,y) = sigma^2 exp(-(x-y)^2 / (2l^2)),
    the constrained kernel is k_tilde(x,y) = k(x,y) - k_hat(x,y)
    where k_hat is the projection term ensuring orthogonality
    w.r.t. N(0,1) input density.

    Args:
        x: First scalar input.
        y: Second scalar input.
        lengthscale: Kernel lengthscale l.
        variance: Kernel variance sigma^2.

    Returns:
        Scalar constrained kernel value.
    """
    l = lengthscale
    l_sq = jnp.square(l)

    # Base SE kernel: k(x, y) = sigma^2 * exp(-(x - y)^2 / (2 * l^2))
    k_base = variance * jnp.exp(-0.5 * jnp.square(x - y) / l_sq)

    # Projection term (Eq. 10 of Lu et al. 2022 with mu=0, delta^2=1):
    # k_hat(x, y) = sigma^2 * l * sqrt(l^2 + 2) / (l^2 + 1)
    #               * exp(-(x^2 + y^2) / (2(l^2 + 1)))
    coeff = variance * l * jnp.sqrt(l_sq + 2.0) / (l_sq + 1.0)
    k_hat = coeff * jnp.exp(
        -(jnp.square(x) + jnp.square(y)) / (2.0 * (l_sq + 1.0))
    )

    return k_base - k_hat


class OrthogonalAdditiveKernel:
    """Placeholder - will be implemented in Task 4."""
