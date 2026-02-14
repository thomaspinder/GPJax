"""Normalising-flow transforms for OAK input preprocessing.

Provides a :class:`SinhArcsinhTransform` (a NumPyro ``Transform``
subclass) and convenience functions for fitting per-feature normalising
flows that map raw inputs to approximately standard normal.
"""

import jax
import jax.numpy as jnp
import jax.scipy.optimize
from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    ExpTransform,
    Transform,
)

from gpjax.typing import Array


class SinhArcsinhTransform(Transform):
    r"""Jones & Pewsey (2009) sinh-arcsinh bijector.

    .. math::
        y = \sinh\bigl(\tau\,\operatorname{arcsinh}(x) - \varepsilon\bigr)

    where :math:`\varepsilon` (skewness) is unconstrained and
    :math:`\tau` (tailweight) is strictly positive.

    Args:
        skewness: Skewness parameter :math:`\varepsilon`.
        tailweight: Tailweight parameter :math:`\tau > 0`.
    """

    domain = constraints.real
    codomain = constraints.real
    sign = 1

    def __init__(self, skewness, tailweight):
        self.skewness = skewness
        self.tailweight = tailweight

    def __call__(self, x):
        return jnp.sinh(self.tailweight * jnp.arcsinh(x) - self.skewness)

    def _inverse(self, y):
        return jnp.sinh((jnp.arcsinh(y) + self.skewness) / self.tailweight)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return (
            jnp.log(self.tailweight)
            + jnp.log(jnp.cosh(self.tailweight * jnp.arcsinh(x) - self.skewness))
            - 0.5 * jnp.log1p(jnp.square(x))
        )

    def tree_flatten(self):
        return (self.skewness, self.tailweight), (
            ("skewness", "tailweight"),
            dict(),
        )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if not isinstance(other, SinhArcsinhTransform):
            return False
        return bool(
            jnp.array_equal(self.skewness, other.skewness)
            and jnp.array_equal(self.tailweight, other.tailweight)
        )


def _negative_log_likelihood(params, standardised_values, fixed_log_det_jacobian):
    """Negative log-likelihood for the sinh-arcsinh transform.

    Minimising this finds skewness/tailweight that best Gaussianise the data.
    """
    skewness = params[0]
    tailweight = jax.nn.softplus(params[1])
    transform = SinhArcsinhTransform(skewness, tailweight)

    transformed = transform(standardised_values)
    log_det_jacobian = transform.log_abs_det_jacobian(standardised_values, transformed)

    # -E[log p(z)] where p = N(0,1), so log p = -0.5 z^2 + const
    return 0.5 * jnp.mean(jnp.square(transformed)) - jnp.mean(
        fixed_log_det_jacobian + log_det_jacobian
    )


def fit_normalising_flow(x_col: Array) -> ComposeTransform:
    r"""Fit a per-feature normalising flow mapping raw values to ~N(0,1).

    The bijector chain is **Shift -> Log -> Standardise -> SinhArcsinh**.
    Only the SinhArcsinh skewness and tailweight are optimised (via BFGS);
    the first three steps are determined by summary statistics of *x_col*.

    Args:
        x_col: 1-D array of feature values (training data only).

    Returns:
        A :class:`~numpyro.distributions.transforms.ComposeTransform`
        mapping raw feature values to approximately standard normal.
    """
    x = jnp.asarray(x_col)

    # Step 1-2: Shift to positive then log-transform
    offset = -x.min() + 1e-3
    log_values = jnp.log(x + offset)

    # Step 3: Standardise the log-transformed values
    mean_log = jnp.mean(log_values)
    std_log = jnp.std(log_values)
    standardised = (log_values - mean_log) / std_log

    # Log-det-Jacobian contribution from the fixed (non-optimised) steps
    fixed_log_det_jacobian = -jnp.log(x + offset) - jnp.log(std_log)

    # Step 4: Optimise sinh-arcsinh parameters via BFGS
    initial_params = jnp.array([0.0, 1.0])  # [skewness, softplus_inv(tailweight)]
    result = jax.scipy.optimize.minimize(
        lambda params: _negative_log_likelihood(
            params, standardised, fixed_log_det_jacobian
        ),
        initial_params,
        method="BFGS",
    )

    skewness = result.x[0]
    tailweight = jax.nn.softplus(result.x[1])

    return ComposeTransform(
        [
            AffineTransform(offset, 1.0),
            ExpTransform().inv,
            AffineTransform(-mean_log / std_log, 1.0 / std_log),
            SinhArcsinhTransform(skewness, tailweight),
        ]
    )


def fit_all_normalising_flows(X: Array) -> list[ComposeTransform]:
    r"""Fit independent normalising flows for each column of *X*.

    Args:
        X: Array of shape ``(N, D)``.

    Returns:
        List of *D* :class:`~numpyro.distributions.transforms.ComposeTransform`
        instances, one per feature.
    """
    return [fit_normalising_flow(X[:, d]) for d in range(X.shape[1])]
