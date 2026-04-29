"""Tests for _sign_normalise — row-sign flipper that preserves R.T @ R.

See plans/2026-04-21-state-space-gps-design.md §Stage 2 square-root Kalman filter.
"""

from gpjax.state_space.inference import _sign_normalise
import jax.numpy as jnp
import numpy as np


def test_sign_normalise_makes_diagonal_non_negative():
    R = jnp.array([[-2.0, 1.0, 0.5], [0.0, -3.0, 0.2], [0.0, 0.0, 1.5]])
    R_norm = np.asarray(_sign_normalise(R))
    assert np.all(np.diag(R_norm) >= 0)


def test_sign_normalise_preserves_R_T_R():
    R = jnp.array([[-2.0, 1.0, 0.5], [0.0, -3.0, 0.2], [0.0, 0.0, 1.5]])
    R_norm = _sign_normalise(R)
    np.testing.assert_allclose(
        np.asarray(R.T @ R), np.asarray(R_norm.T @ R_norm), atol=1e-14
    )
