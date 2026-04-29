"""Reference TruncatedPeriodic SDE with a 2-D DC block (BayesNewton convention).

Used only to regression-test v5.1's 1-D DC implementation.
"""

import numpy as np
import scipy.special


def two_d_dc_periodic_stationary_variance_diag(lengthscale, variance, period, K):
    bessel_argument = 1.0 / (4.0 * lengthscale**2)
    q0 = variance * scipy.special.ive(0, bessel_argument)
    qks = (
        2.0
        * variance
        * np.array([scipy.special.ive(k, bessel_argument) for k in range(1, K + 1)])
    )
    diag = np.concatenate([np.array([q0, q0]), np.repeat(qks, 2)])
    return diag  # shape (2 * (K + 1),)


def two_d_dc_periodic_A(dt, period, K):
    state_dim = 2 * (K + 1)
    omegas = np.concatenate(
        [[0.0], 2 * np.pi * np.arange(1, K + 1) / period]
    )  # includes ω_0 = 0
    A = np.zeros((state_dim, state_dim))
    for k, omega_k in enumerate(omegas):
        block_start = 2 * k
        A[block_start, block_start] = np.cos(omega_k * dt)
        A[block_start, block_start + 1] = np.sin(omega_k * dt)
        A[block_start + 1, block_start] = -np.sin(omega_k * dt)
        A[block_start + 1, block_start + 1] = np.cos(omega_k * dt)
    return A
