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


def fit_normalising_flow(x_col: Array) -> ComposeTransform:
    r"""Fit a per-feature normalising flow mapping raw values to ~N(0,1).

    The bijector chain is **Shift → Log → Standardise → SinhArcsinh**.
    Only the SinhArcsinh skewness and tailweight are optimised (via BFGS);
    the first three steps are determined by summary statistics of *x_col*.

    Args:
        x_col: 1-D array of feature values (training data only).

    Returns:
        A :class:`~numpyro.distributions.transforms.ComposeTransform`
        mapping raw feature values to approximately standard normal.
    """
    x = jnp.asarray(x_col)

    # Fixed statistics from data
    offset = -x.min() + 1e-3
    log_vals = jnp.log(x + offset)
    mean_log = jnp.mean(log_vals)
    std_log = jnp.std(log_vals)

    # Pre-compute standardised values (first 3 steps are fixed)
    z_pre = (log_vals - mean_log) / std_log
    ldj_fixed = -jnp.log(x + offset) - jnp.log(std_log)

    def loss(params):
        skewness = params[0]
        tailweight = jax.nn.softplus(params[1])
        sa = SinhArcsinhTransform(skewness, tailweight)
        z = sa(z_pre)
        ldj_sa = sa.log_abs_det_jacobian(z_pre, z)
        return 0.5 * jnp.mean(jnp.square(z)) - jnp.mean(ldj_fixed + ldj_sa)

    result = jax.scipy.optimize.minimize(loss, jnp.array([0.0, 1.0]), method="BFGS")

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
