"""Tests for the square-root Kalman filter primitives.

See plans/2026-04-21-state-space-gps-design.md §Stage 2.
"""

import gpjax as gpx
from gpjax.state_space.inference import (
    _sqrt_predict,
    _sqrt_update,
    kalman_filter,
)
from gpjax.state_space.sde import Matern12SDE
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tests._reference.dense_kalman import dense_kalman_predict, dense_kalman_update


@pytest.fixture
def predict_inputs_2d():
    """A non-trivial 2-D predict step input set."""
    mean_prev = jnp.array([0.5, -0.3])
    covariance_prev = jnp.array([[1.5, 0.2], [0.2, 0.8]])
    L_prev = jnp.linalg.cholesky(covariance_prev)
    transition_matrix = jnp.array([[0.9, 0.05], [0.0, 0.85]])
    process_noise_cov = jnp.array([[0.1, 0.02], [0.02, 0.15]])
    L_Q = jnp.linalg.cholesky(process_noise_cov)
    return mean_prev, covariance_prev, L_prev, transition_matrix, process_noise_cov, L_Q


def test_sqrt_predict_matches_dense_reference(predict_inputs_2d):
    mean_prev, covariance_prev, L_prev, transition_matrix, process_noise_cov, L_Q = (
        predict_inputs_2d
    )
    mean_pred_sqrt, L_pred = _sqrt_predict(mean_prev, L_prev, transition_matrix, L_Q)
    mean_pred_ref, cov_pred_ref = dense_kalman_predict(
        mean_prev,
        covariance_prev,
        transition_matrix,
        process_noise_cov,
    )
    np.testing.assert_allclose(np.asarray(mean_pred_sqrt), mean_pred_ref, atol=1e-10)
    cov_pred_sqrt = np.asarray(L_pred @ L_pred.T)
    np.testing.assert_allclose(cov_pred_sqrt, cov_pred_ref, atol=1e-10)


def test_sqrt_predict_returns_non_negative_diagonal(predict_inputs_2d):
    mean_prev, _covariance_prev, L_prev, transition_matrix, _process_noise_cov, L_Q = (
        predict_inputs_2d
    )
    _, L_pred = _sqrt_predict(mean_prev, L_prev, transition_matrix, L_Q)
    assert (np.diag(np.asarray(L_pred)) >= 0).all()


@pytest.fixture
def update_inputs_2d():
    mean_pred = jnp.array([0.5, -0.3])
    covariance_pred = jnp.array([[1.0, 0.1], [0.1, 0.5]])
    L_pred = jnp.linalg.cholesky(covariance_pred)
    observation_matrix = jnp.array([[1.0, 0.0]])
    observation = jnp.asarray(0.7)
    observation_stddev = jnp.asarray(0.4)
    return (
        mean_pred,
        covariance_pred,
        L_pred,
        observation_matrix,
        observation,
        observation_stddev,
    )


def test_sqrt_update_matches_dense_reference(update_inputs_2d):
    mean_pred, covariance_pred, L_pred, H, y, sigma = update_inputs_2d
    mean_updated_sqrt, L_updated_sqrt, log_likelihood_sqrt = _sqrt_update(
        mean_pred,
        L_pred,
        y,
        H,
        sigma,
    )
    mean_updated_ref, cov_updated_ref, log_likelihood_ref = dense_kalman_update(
        mean_pred,
        covariance_pred,
        y,
        H,
        sigma**2,
    )
    np.testing.assert_allclose(
        np.asarray(mean_updated_sqrt), mean_updated_ref, atol=1e-10
    )
    cov_updated_sqrt = np.asarray(L_updated_sqrt @ L_updated_sqrt.T)
    np.testing.assert_allclose(cov_updated_sqrt, cov_updated_ref, atol=1e-10)
    np.testing.assert_allclose(
        float(log_likelihood_sqrt), log_likelihood_ref, atol=1e-10
    )


def test_sqrt_update_returns_non_negative_diagonal(update_inputs_2d):
    mean_pred, _, L_pred, H, y, sigma = update_inputs_2d
    _, L_updated, _ = _sqrt_update(mean_pred, L_pred, y, H, sigma)
    assert (np.diag(np.asarray(L_updated)) >= 0).all()


def _build_matern12_dataset(
    n=20, lengthscale=1.0, variance=1.0, obs_stddev=0.3, seed=0
):
    """Sample a small 1-D Matern12 dataset for filter integration tests."""
    key = jr.key(seed)
    key_x, key_eps = jr.split(key)
    X = jnp.sort(jr.uniform(key_x, shape=(n,), minval=0.0, maxval=10.0))
    kernel_dense = gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance)
    K = kernel_dense.gram(X.reshape(-1, 1)).as_matrix()
    L = jnp.linalg.cholesky(K + 1e-9 * jnp.eye(n))
    y = L @ jr.normal(key_eps, shape=(n,)) + obs_stddev * jr.normal(
        jr.fold_in(key_eps, 1), shape=(n,)
    )
    return X, y


@pytest.mark.parametrize("jitter", [0.0, 1e-6])
def test_kalman_filter_matches_dense_mll_on_matern12(jitter):
    """Square-root Kalman filter MLL should match conjugate_mll for Matern12."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n = 20

    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )

    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
        jitter=jitter,
    )
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=obs_stddev)
    posterior = prior * likelihood
    dense_mll = gpx.objectives.conjugate_mll(posterior, train_data)

    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)
    sigma_eff = jnp.sqrt(obs_stddev**2 + jitter)
    sqrt_mll = kalman_filter(sde, y, time_steps, is_observed, sigma_eff)

    np.testing.assert_allclose(float(sqrt_mll), float(dense_mll), atol=1e-6, rtol=1e-8)


def test_kalman_filter_masking_equivalent_to_dropping():
    """Masking a single observation should give the same MLL as dropping it."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n = 20
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )

    mask_index = 7
    is_observed = jnp.ones(n, dtype=bool).at[mask_index].set(False)

    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    sigma_eff = jnp.asarray(obs_stddev)
    sqrt_mll_masked = kalman_filter(sde, y, time_steps, is_observed, sigma_eff)

    keep = jnp.ones(n, dtype=bool).at[mask_index].set(False)
    X_kept = X[keep]
    y_kept = y[keep]
    time_steps_kept = jnp.concatenate([jnp.array([0.0]), jnp.diff(X_kept)])
    is_observed_kept = jnp.ones(X_kept.shape[0], dtype=bool)
    sqrt_mll_dropped = kalman_filter(
        sde, y_kept, time_steps_kept, is_observed_kept, sigma_eff
    )

    np.testing.assert_allclose(
        float(sqrt_mll_masked), float(sqrt_mll_dropped), atol=1e-10
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 5, 20])
def test_kalman_filter_chunked_matches_unchunked(chunk_size):
    """MLL must agree across chunk_size in {None, 1, sqrt(N), N}."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n = 20
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)
    sigma_eff = jnp.asarray(obs_stddev)

    # Reference: chunk_size=N (one big chunk, equivalent to un-chunked path).
    mll_one_chunk = kalman_filter(
        sde, y, time_steps, is_observed, sigma_eff, chunk_size=n
    )
    mll_chunked = kalman_filter(
        sde, y, time_steps, is_observed, sigma_eff, chunk_size=chunk_size
    )
    np.testing.assert_allclose(
        float(mll_chunked), float(mll_one_chunk), atol=1e-10, rtol=1e-10
    )


def test_kalman_filter_chunked_with_padding_does_not_change_mll():
    """Filter with N=23 (not a multiple of small chunk) and chunk_size=5 → pad_len=2.
    Padded no-op steps must contribute exactly zero."""
    lengthscale = 1.0
    variance = 1.0
    obs_stddev = 0.3
    n = 23
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)
    sigma_eff = jnp.asarray(obs_stddev)

    mll_chunk_5 = kalman_filter(
        sde, y, time_steps, is_observed, sigma_eff, chunk_size=5
    )
    mll_chunk_n = kalman_filter(
        sde, y, time_steps, is_observed, sigma_eff, chunk_size=n
    )
    np.testing.assert_allclose(
        float(mll_chunk_5), float(mll_chunk_n), atol=1e-10, rtol=1e-10
    )


def test_kalman_filter_handles_repeated_timestamps():
    """Two observations at the same t must be sequential Bayes updates with no
    numerical fudge: time_step=0 returns (I, 0) exactly so the second update sees
    the post-first-update state at the same time."""
    lengthscale = 1.0
    variance = 1.5
    obs_stddev = 0.2
    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)

    # Three timestamps, the middle one duplicated.
    X = jnp.array([0.0, 1.0, 1.0, 2.5])
    y = jnp.array([0.3, -0.5, 0.4, 0.1])
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    # time_steps == [0, 1, 0, 1.5] — the dt=0 between duplicates triggers the
    # algebraic identity branch.
    is_observed = jnp.ones(4, dtype=bool)
    sigma_eff = jnp.asarray(obs_stddev)

    # Filter MLL.
    sqrt_mll = kalman_filter(sde, y, time_steps, is_observed, sigma_eff, chunk_size=4)

    # Dense reference: apply 4 sequential predict+update steps using
    # dense_kalman_* with A and Q drawn from sde.discretise.
    L_inf = np.asarray(sde.stationary_state_cov_sqrt)
    P_inf = L_inf @ L_inf.T

    H = np.asarray(sde.observation_matrix)
    mean_curr = np.zeros(sde.state_dim)
    cov_curr = P_inf.copy()
    log_likelihood_total = 0.0
    for i in range(4):
        A_i, L_Q_i = sde.discretise(jnp.asarray(time_steps[i]))
        Q_i = np.asarray(L_Q_i @ L_Q_i.T)
        A_i_np = np.asarray(A_i)
        mean_curr, cov_curr = dense_kalman_predict(mean_curr, cov_curr, A_i_np, Q_i)
        mean_curr, cov_curr, log_likelihood_increment = dense_kalman_update(
            mean_curr,
            cov_curr,
            float(y[i]),
            H,
            float(obs_stddev) ** 2,
        )
        log_likelihood_total += log_likelihood_increment

    np.testing.assert_allclose(float(sqrt_mll), log_likelihood_total, atol=1e-8)


def test_kalman_filter_repeated_timestamp_dt_is_zero():
    """Smoke check: time_steps[i] == 0 at duplicate; SDE.discretise(0) returns (I, 0)."""
    sde = Matern12SDE(lengthscale=1.0, variance=1.0)
    A_zero, L_Q_zero = sde.discretise(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(A_zero), np.eye(1))
    np.testing.assert_array_equal(np.asarray(L_Q_zero), np.zeros((1, 1)))
