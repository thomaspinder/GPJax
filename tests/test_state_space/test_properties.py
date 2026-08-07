"""Property-based tests for state_space_mll using hypothesis.

See plans/2026-04-22-state-space-gps-implementation.md §Phase 13.
"""

from __future__ import annotations

import warnings

import gpjax as gpx
from gpjax.state_space._validation import (
    sort_state_space_data,
    validate_state_space_data,
)
from gpjax.state_space.gps import StateSpacePrior
from gpjax.state_space.objectives import state_space_mll
from hypothesis import given, settings
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import pytest


# Reusable strategy: a small sorted-time-series dataset.
def _make_state_space_dataset_strategy(n_min=8, n_max=20):
    """Hypothesis strategy yielding (X_sorted, y) suitable for state-space MLL."""

    @st.composite
    def _strategy(draw):
        n = draw(st.integers(min_value=n_min, max_value=n_max))
        # Distinct sorted timestamps in [0, 5].
        times_unique = sorted(
            draw(
                st.lists(
                    st.floats(
                        min_value=0.0,
                        max_value=5.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=n,
                    max_size=n,
                    unique=True,
                )
            )
        )
        X = jnp.asarray(times_unique).reshape(-1, 1)
        # y entries in a tame range to avoid extreme-MLL flakiness.
        y_values = draw(
            hnp.arrays(
                dtype=np.float64,
                shape=(n,),
                elements=st.floats(
                    min_value=-3.0,
                    max_value=3.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        y = jnp.asarray(y_values).reshape(-1, 1)
        return X, y

    return _strategy()


def _build_matern_posterior(kernel_class, lengthscale, variance, obs_stddev, n):
    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel_class(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    return prior * likelihood


# --- Task 13.1 — Sort-invariance --------------------------------------------


@settings(deadline=None, max_examples=15)
@given(data=_make_state_space_dataset_strategy())
def test_state_space_mll_invariant_under_permutation(data):
    X_sorted, y_sorted = data
    n = X_sorted.shape[0]
    permutation = np.random.RandomState(0).permutation(n)
    X_perm = X_sorted[permutation]
    y_perm = y_sorted[permutation]

    posterior = _build_matern_posterior(
        gpx.kernels.Matern12,
        lengthscale=1.0,
        variance=1.0,
        obs_stddev=0.3,
        n=n,
    )

    # Validate + sort the permuted data, then compare MLL.
    validate_state_space_data(X_perm, y_perm)
    with warnings.catch_warnings():
        # sort_state_space_data warns when unsorted.
        warnings.simplefilter("ignore", UserWarning)
        X_resorted, y_resorted, _ = sort_state_space_data(X_perm, y_perm)

    train_perm = gpx.Dataset(X=X_resorted, y=y_resorted)
    train_sorted = gpx.Dataset(X=X_sorted, y=y_sorted)

    mll_resorted = float(state_space_mll(posterior, train_perm))
    mll_sorted = float(state_space_mll(posterior, train_sorted))
    np.testing.assert_allclose(mll_resorted, mll_sorted, atol=1e-9, rtol=1e-9)


# --- Task 13.2 — Mask-augmentation invariance --------------------------------


@settings(deadline=None, max_examples=15)
@given(
    data=_make_state_space_dataset_strategy(),
    insert_position_offset=st.floats(0.01, 4.99),
)
def test_state_space_mll_invariant_under_masked_augmentation(
    data, insert_position_offset
):
    """Appending a fake all-False masked observation must not change the MLL."""
    X_orig, y_orig = data
    n = X_orig.shape[0]

    # Insert a fake (timestamp, observation) at position insert_position_offset
    # (inside [0, 5]) and mark it unobserved.
    fake_time = jnp.asarray(insert_position_offset).reshape(1, 1)
    fake_y = jnp.asarray([[42.0]])
    X_aug = jnp.concatenate([X_orig, fake_time], axis=0)
    y_aug = jnp.concatenate([y_orig, fake_y], axis=0)
    mask_aug = jnp.concatenate([jnp.ones(n, dtype=bool), jnp.array([False])])

    # Validate + sort the augmented data.
    validate_state_space_data(X_aug, y_aug, mask_aug)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X_aug_sorted, y_aug_sorted, mask_aug_sorted = sort_state_space_data(
            X_aug, y_aug, mask_aug
        )

    posterior_orig = _build_matern_posterior(
        gpx.kernels.Matern12,
        lengthscale=1.0,
        variance=1.0,
        obs_stddev=0.3,
        n=n,
    )
    posterior_aug = _build_matern_posterior(
        gpx.kernels.Matern12,
        lengthscale=1.0,
        variance=1.0,
        obs_stddev=0.3,
        n=n + 1,
    )

    train_orig = gpx.Dataset(X=X_orig, y=y_orig)
    train_aug = gpx.Dataset(X=X_aug_sorted, y=y_aug_sorted)

    mll_orig = float(state_space_mll(posterior_orig, train_orig))
    mll_aug = float(
        state_space_mll(posterior_aug, train_aug, observation_mask=mask_aug_sorted)
    )
    np.testing.assert_allclose(mll_aug, mll_orig, atol=1e-9, rtol=1e-9)


# --- Task 13.3 — Time-scaling equivariance (Matérn) -------------------------


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
@settings(deadline=None, max_examples=15)
@given(
    data=_make_state_space_dataset_strategy(),
    scale_factor=st.floats(min_value=0.5, max_value=4.0, allow_nan=False),
)
def test_state_space_mll_equivariant_under_time_and_lengthscale_rescale(
    data, scale_factor, kernel_class
):
    """Scaling X and lengthscale by the same factor leaves the MLL unchanged."""
    X_sorted, y_sorted = data
    n = X_sorted.shape[0]

    base_lengthscale = 1.0
    variance = 1.0
    obs_stddev = 0.3

    posterior_base = _build_matern_posterior(
        kernel_class, base_lengthscale, variance, obs_stddev, n
    )
    posterior_scaled = _build_matern_posterior(
        kernel_class,
        base_lengthscale * scale_factor,
        variance,
        obs_stddev,
        n,
    )

    train_base = gpx.Dataset(X=X_sorted, y=y_sorted)
    train_scaled = gpx.Dataset(X=X_sorted * scale_factor, y=y_sorted)

    mll_base = float(state_space_mll(posterior_base, train_base))
    mll_scaled = float(state_space_mll(posterior_scaled, train_scaled))
    # Scaling t and ℓ by the same factor leaves the kernel value k(τ/ℓ)
    # unchanged for stationary Matérns; obs_variance is unchanged; therefore
    # the MLL is invariant.
    np.testing.assert_allclose(mll_scaled, mll_base, atol=1e-7, rtol=1e-7)
