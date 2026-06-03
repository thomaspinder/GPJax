"""Tests for _stable_scaled_ive — the three-regime modified-Bessel helper.

See plans/2026-04-21-state-space-gps-design.md §Bessel stability.
"""

from gpjax.state_space._bessel import _stable_scaled_ive
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special


@pytest.mark.parametrize("c", [10.0, 100.0, 1000.0])
@pytest.mark.parametrize("truncation_order", [0, 1, 4, 10])
def test_stable_scaled_ive_large_c_matches_scipy(c, truncation_order):
    out = np.asarray(_stable_scaled_ive(c, truncation_order))
    expected = np.asarray(
        [scipy.special.ive(k, c) for k in range(truncation_order + 1)]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("c", [1e-3, 0.1, 1.0])
@pytest.mark.parametrize("truncation_order", [4, 10, 20])
def test_stable_scaled_ive_moderate_c_matches_scipy(c, truncation_order):
    out = np.asarray(_stable_scaled_ive(c, truncation_order))
    expected = np.asarray(
        [scipy.special.ive(k, c) for k in range(truncation_order + 1)]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("c", [1e-12, 1e-8])
def test_stable_scaled_ive_tiny_c_matches_scipy(c):
    truncation_order = 20
    out = np.asarray(_stable_scaled_ive(c, truncation_order))
    expected = np.asarray(
        [scipy.special.ive(k, c) for k in range(truncation_order + 1)]
    )
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-10)


def test_stable_scaled_ive_non_negative():
    for c in (1e-12, 1e-8, 1e-3, 0.1, 1.0, 10.0, 1e3):
        out = np.asarray(_stable_scaled_ive(c, 20))
        assert (out >= 0).all(), f"Negative Bessel at c={c}: {out}"


@pytest.mark.parametrize("c", [1e-6, 1e-3, 0.5, 2.0, 50.0])
def test_stable_scaled_ive_gradient_is_finite(c):
    c = jnp.asarray(c, dtype=jnp.float64)
    grad_fn = jax.jacobian(lambda c_: _stable_scaled_ive(c_, 8))
    grads = np.asarray(grad_fn(c))
    assert np.all(np.isfinite(grads)), f"Non-finite gradient at c={c}: {grads}"
