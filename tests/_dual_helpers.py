"""Shared fixtures for the dual (t-SVGP) variational family.

``tests/test_variational_families.py``, ``tests/test_objectives.py`` and
``tests/test_natural_gradients.py`` all need the same two things: a reproducible draw
of positive semi-definite sites, and the ``VariationalGaussian`` carrying a dual
family's implied moments. Written once here so that a change to the site convention
cannot be applied to one copy and missed by the other two.
"""

from gpjax.parameters import _val
from gpjax.variational_families import (
    DualVariationalGaussian,
    VariationalGaussian,
)
import jax.numpy as jnp
import jax.random as jr

# Well below the 1e-6 default: the dual/moment equivalence assertions compare two
# routes to the same q, and a large jitter is a large common-mode offset on both.
DUAL_JITTER = 1e-8


def random_dual_sites(seed: int, num_inducing: int):
    r"""Draw $(\lambda_1,\Lambda_2)$ with $\Lambda_2$ positive semi-definite.

    Args:
        seed (int): Seed for the draw.
        num_inducing (int): The number of inducing points, $M$.

    Returns:
        tuple: The $(M, 1)$ site vector and the $(M, M)$ site precision.
    """
    key_vector, key_matrix = jr.split(jr.key(seed))
    raw = jr.normal(key_matrix, (num_inducing, num_inducing))
    return jr.normal(key_vector, (num_inducing, 1)), raw @ raw.T / num_inducing


def build_dual(posterior, inducing_inputs, jitter=DUAL_JITTER, seed=None):
    """Build a dual family, optionally started at random positive semi-definite sites.

    Args:
        posterior: The posterior the family approximates.
        inducing_inputs: The inducing inputs, shape ``(M, D)``.
        jitter (float): The family's jitter.
        seed (int | None): When given, seeds a random site draw; otherwise the sites
            are left at their zero defaults, where ``q(u) = p(u)``.

    Returns:
        DualVariationalGaussian: The family.
    """
    dual_vector = dual_matrix = None
    if seed is not None:
        dual_vector, dual_matrix = random_dual_sites(seed, inducing_inputs.shape[0])
    return DualVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
        dual_vector=dual_vector,
        dual_matrix=dual_matrix,
        jitter=jitter,
    )


def matched_variational_gaussian(q_dual: DualVariationalGaussian):
    """Return a ``VariationalGaussian`` carrying ``q_dual``'s implied moments.

    Args:
        q_dual (DualVariationalGaussian): The dual family.

    Returns:
        VariationalGaussian: The same $q(u)$, stored in moment coordinates.
    """
    mean, covariance = q_dual.moments()
    return VariationalGaussian(
        posterior=q_dual.posterior,
        inducing_inputs=_val(q_dual.inducing_inputs),
        variational_mean=mean,
        variational_root_covariance=jnp.linalg.cholesky(covariance),
        jitter=q_dual.jitter,
    )
