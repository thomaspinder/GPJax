# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pin each stationary kernel to the closed form its docstring advertises.

These formulas are rendered into the public API documentation, so a drift
between the docstring and `__call__` is a user-facing bug even when the
covariance itself is correct.
"""

from itertools import pairwise

from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    PoweredExponential,
    RationalQuadratic,
)
from jax import config
import jax.numpy as jnp
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

LENGTHSCALE = 0.7
VARIANCE = 2.0
X = jnp.array([0.3])
Y = jnp.array([1.1])
RADIUS = jnp.abs(X - Y)[0]


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.5])
def test_rational_quadratic_matches_docstring(alpha: float):
    r"""$k(x,y) = \sigma^2 (1 + \lVert x-y \rVert_2^2 / (2\alpha\ell^2))^{-\alpha}$."""
    kernel = RationalQuadratic(
        n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE, alpha=alpha
    )
    expected = VARIANCE * (1 + RADIUS**2 / (2 * alpha * LENGTHSCALE**2)) ** (-alpha)

    assert jnp.allclose(kernel(X, Y), expected)


def test_rational_quadratic_tends_to_rbf_for_large_alpha():
    """The RQ docstring claims the RBF kernel as the α → ∞ limit."""
    rbf = RBF(n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE)
    errors = [
        abs(
            RationalQuadratic(
                n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE, alpha=alpha
            )(X, Y)
            - rbf(X, Y)
        )
        for alpha in (1e2, 1e4, 1e6)
    ]

    assert errors[-1] < 1e-6
    # Error falls off like 1/α, so each 100× in α buys ~100× accuracy.
    assert all(a > b for a, b in pairwise(errors))


@pytest.mark.parametrize("power", [0.5, 1.0, 1.5, 2.0])
def test_powered_exponential_matches_docstring(power: float):
    r"""$k(x,y) = \sigma^2 \exp(-(\lVert x-y \rVert_2 / \ell)^\kappa)$."""
    kernel = PoweredExponential(
        n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE, power=power
    )
    expected = VARIANCE * jnp.exp(-((RADIUS / LENGTHSCALE) ** power))

    assert jnp.allclose(kernel(X, Y), expected)


def test_rbf_matches_docstring():
    r"""$k(x,y) = \sigma^2 \exp(-\lVert x-y \rVert_2^2 / (2\ell^2))$."""
    kernel = RBF(n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE)
    expected = VARIANCE * jnp.exp(-(RADIUS**2) / (2 * LENGTHSCALE**2))

    assert jnp.allclose(kernel(X, Y), expected)


def test_matern12_matches_docstring():
    r"""$k(x,y) = \sigma^2 \exp(-\lvert x-y \rvert / \ell)$."""
    kernel = Matern12(n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE)
    expected = VARIANCE * jnp.exp(-RADIUS / LENGTHSCALE)

    assert jnp.allclose(kernel(X, Y), expected)


def test_matern32_matches_docstring():
    r"""$k(x,y) = \sigma^2 (1 + \sqrt{3}\lvert x-y \rvert/\ell)\exp(-\sqrt{3}\lvert x-y \rvert/\ell)$."""
    kernel = Matern32(n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE)
    scaled = jnp.sqrt(3.0) * RADIUS / LENGTHSCALE
    expected = VARIANCE * (1 + scaled) * jnp.exp(-scaled)

    assert jnp.allclose(kernel(X, Y), expected)


def test_matern52_matches_docstring():
    r"""$k(x,y) = \sigma^2 (1 + \sqrt{5}r/\ell + 5r^2/(3\ell^2))\exp(-\sqrt{5}r/\ell)$."""
    kernel = Matern52(n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE)
    scaled = jnp.sqrt(5.0) * RADIUS / LENGTHSCALE
    expected = (
        VARIANCE
        * (1 + scaled + 5 * RADIUS**2 / (3 * LENGTHSCALE**2))
        * jnp.exp(-scaled)
    )

    assert jnp.allclose(kernel(X, Y), expected)


def test_periodic_matches_docstring():
    r"""$k(x,y) = \sigma^2 \exp(-\tfrac12 \sum_i (\sin(\pi(x_i-y_i)/p)/\ell)^2)$."""
    period = 2.0
    kernel = Periodic(
        n_dims=1, lengthscale=LENGTHSCALE, variance=VARIANCE, period=period
    )
    sine_squared = (jnp.sin(jnp.pi * (X - Y) / period) / LENGTHSCALE) ** 2
    expected = VARIANCE * jnp.exp(-0.5 * jnp.sum(sine_squared))

    assert jnp.allclose(kernel(X, Y), expected)
