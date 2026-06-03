"""Dense covariance-form Kalman filter reference for cross-checks.

Used only to regression-test the square-root implementation.
"""

from __future__ import annotations

import numpy as np


def dense_kalman_predict(
    mean_prev, covariance_prev, transition_matrix, process_noise_covariance
):
    """Standard Kalman predict.

    mean_pred = A @ mean_prev
    cov_pred  = A @ cov_prev @ A.T + Q
    """
    mean_pred = np.asarray(transition_matrix) @ np.asarray(mean_prev)
    cov_pred = np.asarray(transition_matrix) @ np.asarray(covariance_prev) @ np.asarray(
        transition_matrix
    ).T + np.asarray(process_noise_covariance)
    return mean_pred, cov_pred


def dense_kalman_update(
    mean_pred,
    covariance_pred,
    observation,
    observation_matrix,
    observation_variance,
):
    """Standard scalar Kalman update.

    Returns (mean_updated, covariance_updated, log_likelihood_increment).
    """
    H = np.asarray(observation_matrix).reshape(1, -1)
    cov_pred = np.asarray(covariance_pred)
    mean_pred = np.asarray(mean_pred)
    innovation = float(observation) - float((H @ mean_pred).squeeze())
    innovation_variance = float((H @ cov_pred @ H.T).squeeze()) + float(
        observation_variance
    )
    kalman_gain = (cov_pred @ H.T).squeeze() / innovation_variance
    mean_updated = mean_pred + kalman_gain * innovation
    covariance_updated = cov_pred - np.outer(kalman_gain, H @ cov_pred)
    log_likelihood_increment = -0.5 * (
        np.log(2.0 * np.pi * innovation_variance) + innovation**2 / innovation_variance
    )
    return mean_updated, covariance_updated, float(log_likelihood_increment)
