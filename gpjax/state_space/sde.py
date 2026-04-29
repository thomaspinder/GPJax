"""Linear SDE types and closed-form discretisations for state-space GPs.

See plans/2026-04-21-state-space-gps-design.md §Core types.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Float

from gpjax.state_space._bessel import _stable_scaled_ive


class LinearSDE(eqx.Module):
    """Linear time-invariant SDE for a state-space GP.

    dx(t) = F x(t) dt + L dw(t),  w Brownian with spectral density Q_c
    y(t)  = H x(t) + ε,           ε ~ N(0, σ_y²)

    Kernel-specific closed-form discretisations override ``discretise``; the
    base implementation returns the algebraic Δt = 0 result (A = I, L_Q = 0).
    """

    drift_matrix: Float[Array, "state_dim state_dim"]
    diffusion_matrix: Float[Array, "state_dim noise_dim"]
    process_noise_spectral_density: Float[Array, "noise_dim noise_dim"]
    observation_matrix: Float[Array, "1 state_dim"]
    stationary_state_cov_sqrt: Float[Array, "state_dim state_dim"]

    state_dim: int = eqx.field(static=True)

    def discretise(
        self, time_step: Float[Array, ""]
    ) -> tuple[
        Float[Array, "state_dim state_dim"],
        Float[Array, "state_dim state_dim"],
    ]:
        """Return (A(Δt), L_Q(Δt)). Base implementation: Δt = 0 algebraic."""
        eye = jnp.eye(self.state_dim)
        zeros = jnp.zeros((self.state_dim, self.state_dim))
        return eye, zeros


class Matern12SDE(LinearSDE):
    """Matern-1/2 state-space representation.

    F = [[-1/ℓ]], L = [[1]], Q_c = [[2σ²/ℓ]], H = [[1]], P_∞ = [[σ²]].
    A(Δt) = exp(-Δt/ℓ);  L_Q(Δt) = σ · sqrt(-expm1(-2Δt/ℓ)).
    """

    lengthscale: Float[Array, ""]
    variance: Float[Array, ""]

    def __init__(self, lengthscale, variance):
        lengthscale = jnp.asarray(lengthscale)
        variance = jnp.asarray(variance)
        self.lengthscale = lengthscale
        self.variance = variance
        F = jnp.array([[-1.0 / lengthscale]])
        L = jnp.array([[1.0]])
        Qc = jnp.array([[2.0 * variance / lengthscale]])
        H = jnp.array([[1.0]])
        L_inf = jnp.array([[jnp.sqrt(variance)]])
        super().__init__(
            drift_matrix=F,
            diffusion_matrix=L,
            process_noise_spectral_density=Qc,
            observation_matrix=H,
            stationary_state_cov_sqrt=L_inf,
            state_dim=1,
        )

    def discretise(self, time_step):
        A_scalar = jnp.exp(-time_step / self.lengthscale)
        L_Q_scalar = jnp.sqrt(
            self.variance * -jnp.expm1(-2.0 * time_step / self.lengthscale)
        )
        return jnp.array([[A_scalar]]), jnp.array([[L_Q_scalar]])


def _psd_sqrt(matrix):
    """PSD matrix square root via eigendecomposition with non-negative clipping.

    Returns ``M`` such that ``M @ M.T`` reconstructs the input up to round-off,
    even when round-off has produced eigenvalues marginally below zero. This is
    not a Cholesky factor (not lower-triangular); use only where the consumer
    needs any square root, not specifically a Cholesky one.
    """
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    eigvals_nonneg = jnp.maximum(eigvals, 0.0)
    return eigvecs * jnp.sqrt(eigvals_nonneg)


def _matern32_discrete_q(dt, lengthscale, variance):
    """Discrete process-noise covariance Q(Δt) via stationary identity, symmetrised."""
    lam = jnp.sqrt(3.0) / lengthscale
    dtl = dt * lam
    exp_neg_dtl = jnp.exp(-dtl)
    A = exp_neg_dtl * jnp.array(
        [
            [1.0 + dtl, dt],
            [-(lam**2) * dt, 1.0 - dtl],
        ]
    )
    P_inf = jnp.diag(jnp.stack([variance, 3.0 * variance / lengthscale**2]))
    Q = P_inf - A @ P_inf @ A.T
    return 0.5 * (Q + Q.T)


class Matern32SDE(LinearSDE):
    """Matern-3/2 state-space representation (Särkkä & Solin 2019 §12.3)."""

    lengthscale: Float[Array, ""]
    variance: Float[Array, ""]

    def __init__(self, lengthscale, variance):
        lengthscale = jnp.asarray(lengthscale)
        variance = jnp.asarray(variance)
        self.lengthscale = lengthscale
        self.variance = variance
        lam = jnp.sqrt(3.0) / lengthscale
        F = jnp.array([[0.0, 1.0], [-(lam**2), -2.0 * lam]])
        L = jnp.array([[0.0], [1.0]])
        Qc = jnp.array([[12.0 * jnp.sqrt(3.0) * variance / lengthscale**3]])
        H = jnp.array([[1.0, 0.0]])
        P_inf = jnp.diag(jnp.stack([variance, 3.0 * variance / lengthscale**2]))
        L_inf = jnp.linalg.cholesky(P_inf)
        super().__init__(
            drift_matrix=F,
            diffusion_matrix=L,
            process_noise_spectral_density=Qc,
            observation_matrix=H,
            stationary_state_cov_sqrt=L_inf,
            state_dim=2,
        )

    def discretise(self, time_step):
        eye = jnp.eye(2)
        zeros = jnp.zeros((2, 2))

        def zero_dt(_):
            return eye, zeros

        def positive_dt(dt):
            lam = jnp.sqrt(3.0) / self.lengthscale
            dtl = dt * lam
            exp_neg_dtl = jnp.exp(-dtl)
            A = exp_neg_dtl * jnp.array(
                [
                    [1.0 + dtl, dt],
                    [-(lam**2) * dt, 1.0 - dtl],
                ]
            )
            Q = _matern32_discrete_q(dt, self.lengthscale, self.variance)
            L_Q = _psd_sqrt(Q)
            return A, L_Q

        return jax.lax.cond(time_step == 0.0, zero_dt, positive_dt, time_step)


def _matern52_discrete_q(dt, lengthscale, variance):
    lam = jnp.sqrt(5.0) / lengthscale
    dtlam = dt * lam
    A = jnp.exp(-dtlam) * (
        dt
        * jnp.array(
            [
                [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                [
                    lam**3 * (0.5 * dtlam - 1.0),
                    lam**2 * (dtlam - 3.0),
                    lam * (0.5 * dtlam - 2.0),
                ],
            ]
        )
        + jnp.eye(3)
    )
    kappa = 5.0 / 3.0 * variance / lengthscale**2
    P_inf = jnp.array(
        [
            [variance, 0.0, -kappa],
            [0.0, kappa, 0.0],
            [-kappa, 0.0, 25.0 * variance / lengthscale**4],
        ]
    )
    Q = P_inf - A @ P_inf @ A.T
    return 0.5 * (Q + Q.T)


class Matern52SDE(LinearSDE):
    """Matern-5/2 state-space representation (Särkkä & Solin 2019 §12.3)."""

    lengthscale: Float[Array, ""]
    variance: Float[Array, ""]

    def __init__(self, lengthscale, variance):
        lengthscale = jnp.asarray(lengthscale)
        variance = jnp.asarray(variance)
        self.lengthscale = lengthscale
        self.variance = variance
        lam = jnp.sqrt(5.0) / lengthscale
        F = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3), -3.0 * lam**2, -3.0 * lam],
            ]
        )
        L = jnp.array([[0.0], [0.0], [1.0]])
        Qc = jnp.array([[400.0 * jnp.sqrt(5.0) / 3.0 * variance / lengthscale**5]])
        H = jnp.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * variance / lengthscale**2
        P_inf = jnp.array(
            [
                [variance, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * variance / lengthscale**4],
            ]
        )
        L_inf = jnp.linalg.cholesky(P_inf)
        super().__init__(
            drift_matrix=F,
            diffusion_matrix=L,
            process_noise_spectral_density=Qc,
            observation_matrix=H,
            stationary_state_cov_sqrt=L_inf,
            state_dim=3,
        )

    def discretise(self, time_step):
        eye = jnp.eye(3)
        zeros = jnp.zeros((3, 3))

        def zero_dt(_):
            return eye, zeros

        def positive_dt(dt):
            lam = jnp.sqrt(5.0) / self.lengthscale
            dtlam = dt * lam
            A = jnp.exp(-dtlam) * (
                dt
                * jnp.array(
                    [
                        [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                        [
                            -0.5 * dtlam * lam**2,
                            lam * (1.0 - dtlam),
                            1.0 - 0.5 * dtlam,
                        ],
                        [
                            lam**3 * (0.5 * dtlam - 1.0),
                            lam**2 * (dtlam - 3.0),
                            lam * (0.5 * dtlam - 2.0),
                        ],
                    ]
                )
                + jnp.eye(3)
            )
            Q = _matern52_discrete_q(dt, self.lengthscale, self.variance)
            return A, _psd_sqrt(Q)

        return jax.lax.cond(time_step == 0.0, zero_dt, positive_dt, time_step)


class TruncatedPeriodicSDE(LinearSDE):
    """Truncated-Fourier approximation of the periodic kernel (Solin & Särkkä 2014).

    State dim = 1 + 2K with a 1-D DC block and K 2-D harmonic blocks.
    See plans/2026-04-21-state-space-gps-design.md §TruncatedPeriodic.
    """

    lengthscale: Float[Array, ""]
    variance: Float[Array, ""]
    period: Float[Array, ""]
    truncation_order: int = eqx.field(static=True)

    def __init__(self, lengthscale, variance, period, truncation_order):
        lengthscale = jnp.asarray(lengthscale)
        variance = jnp.asarray(variance)
        period = jnp.asarray(period)
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period
        self.truncation_order = truncation_order

        state_dim = 1 + 2 * truncation_order
        bessel_argument = 1.0 / (4.0 * lengthscale**2)
        scaled_bessels = _stable_scaled_ive(bessel_argument, truncation_order)  # (K+1,)

        # DC (k=0) block: F=0, L=0, variance = σ² · Ĩ_0(c).
        dc_variance = variance * scaled_bessels[0]
        harmonic_variances = 2.0 * variance * scaled_bessels[1:]  # (K,)

        omega_per_harmonic = 2.0 * jnp.pi * jnp.arange(1, truncation_order + 1) / period

        # Drift F: block-diagonal. DC block is scalar 0. k-th harmonic is
        # [[0, ωₖ], [-ωₖ, 0]].
        F = jnp.zeros((state_dim, state_dim))
        for k in range(truncation_order):
            omega_k = omega_per_harmonic[k]
            block_start = 1 + 2 * k
            F = F.at[block_start, block_start + 1].set(omega_k)
            F = F.at[block_start + 1, block_start].set(-omega_k)

        L = jnp.zeros((state_dim, 1))
        Qc = jnp.zeros((1, 1))

        # H = [1, 1, 0, 1, 0, ..., 1, 0]: DC scalar + cos component of each harmonic.
        H = jnp.zeros((1, state_dim))
        H = H.at[0, 0].set(1.0)
        for k in range(truncation_order):
            H = H.at[0, 1 + 2 * k].set(1.0)

        # Stationary covariance: diag([σ²Ĩ_0, 2σ²Ĩ_1, 2σ²Ĩ_1, 2σ²Ĩ_2, 2σ²Ĩ_2, ...]).
        P_inf_diag = jnp.concatenate(
            [
                jnp.array([dc_variance]),
                jnp.repeat(harmonic_variances, 2),
            ]
        )
        L_inf = jnp.diag(jnp.sqrt(P_inf_diag))

        super().__init__(
            drift_matrix=F,
            diffusion_matrix=L,
            process_noise_spectral_density=Qc,
            observation_matrix=H,
            stationary_state_cov_sqrt=L_inf,
            state_dim=state_dim,
        )

    def discretise(self, time_step):
        K = self.truncation_order
        state_dim = 1 + 2 * K
        omega_per_harmonic = 2.0 * jnp.pi * jnp.arange(1, K + 1) / self.period
        phases = omega_per_harmonic * time_step
        cos_phases = jnp.cos(phases)
        sin_phases = jnp.sin(phases)

        A = jnp.eye(state_dim)
        for k in range(K):
            block_start = 1 + 2 * k
            A = A.at[block_start, block_start].set(cos_phases[k])
            A = A.at[block_start, block_start + 1].set(sin_phases[k])
            A = A.at[block_start + 1, block_start].set(-sin_phases[k])
            A = A.at[block_start + 1, block_start + 1].set(cos_phases[k])

        L_Q = jnp.zeros((state_dim, state_dim))
        return A, L_Q


class SumSDE(LinearSDE):
    """Block-diagonal sum of LinearSDE components.

    Each component contributes its own state-space block; the observable
    process is the sum of component contributions, expressed via row-concatenation
    of per-component observation matrices.
    """

    components: tuple[LinearSDE, ...]

    def __init__(self, components: tuple[LinearSDE, ...]):
        self.components = tuple(components)
        sum_state_dim = sum(component.state_dim for component in self.components)

        F = jax.scipy.linalg.block_diag(
            *[component.drift_matrix for component in self.components]
        )
        L = jax.scipy.linalg.block_diag(
            *[component.diffusion_matrix for component in self.components]
        )
        Qc = jax.scipy.linalg.block_diag(
            *[component.process_noise_spectral_density for component in self.components]
        )
        L_inf = jax.scipy.linalg.block_diag(
            *[component.stationary_state_cov_sqrt for component in self.components]
        )
        H = jnp.concatenate(
            [component.observation_matrix for component in self.components], axis=1
        )

        super().__init__(
            drift_matrix=F,
            diffusion_matrix=L,
            process_noise_spectral_density=Qc,
            observation_matrix=H,
            stationary_state_cov_sqrt=L_inf,
            state_dim=sum_state_dim,
        )

    def discretise(self, time_step):
        per_component_A = []
        per_component_L_Q = []
        for component in self.components:
            A_component, L_Q_component = component.discretise(time_step)
            per_component_A.append(A_component)
            per_component_L_Q.append(L_Q_component)
        A_sum = jax.scipy.linalg.block_diag(*per_component_A)
        L_Q_sum = jax.scipy.linalg.block_diag(*per_component_L_Q)
        return A_sum, L_Q_sum
