"""Tests for the square-root RTS smoother.

See plans/2026-04-21-state-space-gps-design.md §Stage 3.
"""

import gpjax as gpx
from gpjax.state_space import StateSpacePrior
from gpjax.state_space.inference import _sqrt_filter_forward, rts_smoother
from gpjax.state_space.sde import Matern12SDE, _psd_sqrt
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_state_space.test_filter import _build_matern12_dataset


@pytest.mark.parametrize("jitter", [0.0, 1e-6])
def test_rts_smoother_marginals_match_dense_gp_posterior(jitter):
    """Smoothed posterior marginals at training points must match the dense
    GP posterior conditioned on all training data."""
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    n = 15
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    sigma_eff = jnp.sqrt(obs_stddev**2 + jitter)

    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)

    # Forward filter trajectory.
    forward_outputs, _ = _sqrt_filter_forward(
        sde,
        y,
        time_steps,
        is_observed,
        sigma_eff,
    )
    # Backward smoother.
    smoothed_means, smoothed_Ls = rts_smoother(sde, forward_outputs, time_steps)
    smoothed_observation_means = jnp.einsum(
        "ij,nj->ni", sde.observation_matrix, smoothed_means
    ).squeeze(-1)
    smoothed_observation_variances = jax.vmap(
        lambda L: (
            sde.observation_matrix @ (L @ L.T) @ sde.observation_matrix.T
        ).squeeze()
    )(smoothed_Ls)

    # Dense GP posterior at training points.
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
        jitter=jitter,
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = prior * likelihood
    latent_dist = posterior.predict(X.reshape(-1, 1), train_data=train_data)
    dense_means = np.asarray(latent_dist.mean)
    dense_variances = np.asarray(latent_dist.variance)

    # The dense GP path (cola/lineax solves) accumulates ~1e-6 roundoff at
    # jitter=0; an independent NumPy RTS smoother agrees with this implementation
    # to machine precision (~2e-16), so the residual is dense-GP noise.
    np.testing.assert_allclose(
        np.asarray(smoothed_observation_means), dense_means, atol=5e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(smoothed_observation_variances),
        dense_variances,
        atol=5e-6,
        rtol=1e-6,
    )


def _numpy_rts_reference(sde, y, time_steps, obs_stddev_squared):
    """Plain NumPy RTS smoother used as a high-precision oracle."""
    state_dim = sde.state_dim
    L_inf = np.asarray(sde.stationary_state_cov_sqrt, dtype=np.float64)
    H = np.asarray(sde.observation_matrix, dtype=np.float64)
    n = int(time_steps.shape[0])

    mean_curr = np.zeros(state_dim)
    cov_curr = L_inf @ L_inf.T

    means_filtered, covs_filtered = [], []
    means_predicted, covs_predicted = [], []
    for i in range(n):
        transition_matrix, L_Q = sde.discretise(jnp.asarray(time_steps[i]))
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        L_Q = np.asarray(L_Q, dtype=np.float64)
        process_noise_cov = L_Q @ L_Q.T

        mean_predicted = transition_matrix @ mean_curr
        cov_predicted = (
            transition_matrix @ cov_curr @ transition_matrix.T + process_noise_cov
        )
        means_predicted.append(mean_predicted)
        covs_predicted.append(cov_predicted)

        innovation_var = float((H @ cov_predicted @ H.T).item() + obs_stddev_squared)
        kalman_gain = (cov_predicted @ H.T).reshape(-1) / innovation_var
        innovation = float(y[i]) - float((H @ mean_predicted).item())
        mean_curr = mean_predicted + kalman_gain * innovation
        cov_curr = cov_predicted - np.outer(
            kalman_gain, (H @ cov_predicted).reshape(-1)
        )
        means_filtered.append(mean_curr)
        covs_filtered.append(cov_curr)

    means_smoothed = [None] * n
    covs_smoothed = [None] * n
    gains = [None] * (n - 1)
    means_smoothed[-1] = means_filtered[-1]
    covs_smoothed[-1] = covs_filtered[-1]
    for i in range(n - 2, -1, -1):
        transition_matrix_next, _ = sde.discretise(jnp.asarray(time_steps[i + 1]))
        transition_matrix_next = np.asarray(transition_matrix_next, dtype=np.float64)
        cov_filtered = covs_filtered[i]
        cov_predicted_next = covs_predicted[i + 1]
        smoother_gain = (
            cov_filtered @ transition_matrix_next.T @ np.linalg.inv(cov_predicted_next)
        )
        gains[i] = smoother_gain
        means_smoothed[i] = means_filtered[i] + smoother_gain @ (
            means_smoothed[i + 1] - means_predicted[i + 1]
        )
        covs_smoothed[i] = (
            cov_filtered
            + smoother_gain
            @ (covs_smoothed[i + 1] - cov_predicted_next)
            @ smoother_gain.T
        )
    return np.array(means_smoothed), np.array(covs_smoothed), np.array(gains)


def test_rts_smoother_matches_numpy_reference_to_machine_precision():
    """Square-root RTS smoother must match a plain NumPy RTS reference at fp64
    machine precision; this isolates the correctness of the recursion from any
    dense-GP roundoff."""
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    n = 15
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    sigma_eff = jnp.asarray(obs_stddev)
    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)

    forward_outputs, _ = _sqrt_filter_forward(
        sde, y, time_steps, is_observed, sigma_eff
    )
    smoothed_means, smoothed_Ls = rts_smoother(sde, forward_outputs, time_steps)

    means_reference, covs_reference, _gains_reference = _numpy_rts_reference(
        sde, y, time_steps, float(obs_stddev**2)
    )
    smoothed_covs = jax.vmap(lambda L: L @ L.T)(smoothed_Ls)

    np.testing.assert_allclose(
        np.asarray(smoothed_means), means_reference, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(smoothed_covs), covs_reference, atol=1e-12, rtol=1e-12
    )


def test_rts_smoother_return_gains_matches_numpy_reference():
    """``return_gains=True`` exposes exactly the gains used internally; check
    them against the same NumPy oracle used for the smoothed means/covariances,
    and confirm the default call is unaffected (2-tuple, unchanged values)."""
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    n = 15
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    sigma_eff = jnp.asarray(obs_stddev)
    sde = Matern12SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)

    forward_outputs, _ = _sqrt_filter_forward(
        sde, y, time_steps, is_observed, sigma_eff
    )
    smoothed_means, smoothed_Ls, smoother_gains = rts_smoother(
        sde, forward_outputs, time_steps, return_gains=True
    )
    smoothed_means_default, smoothed_Ls_default = rts_smoother(
        sde, forward_outputs, time_steps
    )

    means_reference, _covs_reference, gains_reference = _numpy_rts_reference(
        sde, y, time_steps, float(obs_stddev**2)
    )

    assert smoother_gains.shape == (n - 1, sde.state_dim, sde.state_dim)
    np.testing.assert_allclose(
        np.asarray(smoother_gains), gains_reference, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(smoothed_means), np.asarray(smoothed_means_default), atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(smoothed_Ls), np.asarray(smoothed_Ls_default), atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(smoothed_means), means_reference, atol=1e-12, rtol=1e-12
    )


def test_smoother_is_finite_under_near_noiseless_dense_sampling():
    """Robustness guard: stiff regime (tiny obs noise, dense Matern-5/2 grid)
    must stay finite.

    Green both before and after the _psd_sqrt swap — documents the contract, not
    a red→green reproduction.
    """
    dense_times = jnp.linspace(0.0, 1.0, 200).reshape(-1, 1)
    targets = jnp.sin(20.0 * dense_times)
    data = gpx.Dataset(X=dense_times, y=targets)

    prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern52(lengthscale=0.05, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=1e-4)
    posterior = prior * likelihood

    test_times = jnp.linspace(0.0, 1.0, 50).reshape(-1, 1)
    predictive = posterior.predict(test_times, data)

    assert bool(jnp.all(jnp.isfinite(predictive.mean)))
    assert bool(jnp.all(jnp.isfinite(predictive.variance)))
    assert bool(jnp.all(predictive.variance >= 0.0))


def test_psd_sqrt_handles_marginally_indefinite_where_cholesky_nans():
    """The smoother's PSD-difference P can be marginally indefinite from round-off.

    jnp.linalg.cholesky NaNs on such a matrix; _psd_sqrt (used by rts_smoother
    after this fix) clips the tiny negative eigenvalue and stays finite,
    reconstructing the PSD part via L @ L.T. This is the failure mode Issue #3
    fixes — green on _psd_sqrt, red on cholesky.
    """
    # Fixed orthogonal basis (QR of a deterministic matrix; no RNG).
    basis, _ = jnp.linalg.qr(
        jnp.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    )
    eigenvalues = jnp.array([1.0, 1e-3, -2e-16])  # tiny negative from "round-off"
    marginally_indefinite = basis @ jnp.diag(eigenvalues) @ basis.T
    symmetrised = 0.5 * (marginally_indefinite + marginally_indefinite.T)

    # RED path: cholesky NaNs on the (marginally) non-PSD matrix.
    cholesky_factor = jnp.linalg.cholesky(symmetrised)
    assert bool(jnp.any(jnp.isnan(cholesky_factor)))

    # GREEN path: _psd_sqrt stays finite and reconstructs the PSD part.
    psd_root = _psd_sqrt(symmetrised)
    assert bool(jnp.all(jnp.isfinite(psd_root)))
    psd_part = basis @ jnp.diag(jnp.clip(eigenvalues, 0.0)) @ basis.T
    # float64 round-trip; conftest enables x64
    assert jnp.allclose(psd_root @ psd_root.T, psd_part, atol=1e-10)
