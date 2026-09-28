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

    ``process_noise_spectral_density`` (``Q_c``) defines the continuous-time
    model and is validated by the Lyapunov identity ``F P∞ + P∞ Fᵀ + L Q_c Lᵀ
    = 0`` in the tests; it is **not** consumed by ``discretise``, which uses
    the closed-form stationary identity ``Q(Δt) = P∞ − A(Δt) P∞ A(Δt)ᵀ``.
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
        eye = jnp.eye(1)
        zeros = jnp.zeros((1, 1))

        def zero_dt(_):
            return eye, zeros

        def positive_dt(dt):
            A_scalar = jnp.exp(-dt / self.lengthscale)
            L_Q_scalar = jnp.sqrt(
                self.variance * -jnp.expm1(-2.0 * dt / self.lengthscale)
            )
            return jnp.array([[A_scalar]]), jnp.array([[L_Q_scalar]])

        return jax.lax.cond(time_step == 0.0, zero_dt, positive_dt, time_step)


def _psd_sqrt(matrix):
    """PSD matrix square root via eigendecomposition with gradient-safe clipping.

    Returns ``M`` such that ``M @ M.T`` reconstructs the input up to round-off,
    even when round-off has produced eigenvalues marginally below zero. This is
    not a Cholesky factor (not lower-triangular); use only where the consumer
    needs any square root, not specifically a Cholesky one.

    Gradient safety. The naive ``jnp.maximum(eigvals, 0.0)`` clip combined
    with ``sqrt`` is gradient-fragile: ``maximum`` is non-differentiable at
    zero and ``d sqrt(x)/dx = 0.5 / sqrt(x)`` is unbounded as ``x -> 0``.
    When the input ``Q = P_inf - A P_inf A.T`` of a Matern SDE picks up a
    round-off-induced eigenvalue near zero (which happens for Matern-5/2 at
    realistic time-densely-sampled N because the third eigenvalue of Q
    involves a ``dt^5/lengthscale^5`` cancellation), the reverse-mode
    gradient through that branch becomes NaN.

    We use the double-``where`` idiom to make the gradient finite everywhere:
    on non-positive eigenvalues we route the value through a constant-1.0
    placeholder (so ``sqrt`` is differentiated at 1, never at 0), and we mask
    the output back to 0 with a second ``where``. This preserves the forward
    value (the placeholder branch never reaches the output) and guarantees
    finite gradients without applying any forward jitter to ``Q``.
    """
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    is_positive = eigvals > 0.0
    # Use the double-where idiom: route negatives to a safe positive value
    # for both forward and gradient computation, then mask the output.
    # This avoids the inf derivative of sqrt at 0 entirely.
    safe_eigvals = jnp.where(is_positive, eigvals, 1.0)
    sqrt_eigvals = jnp.where(is_positive, jnp.sqrt(safe_eigvals), 0.0)
    return eigvecs * sqrt_eigvals


def _matern32_discrete_transition_and_q(time_step, lengthscale, variance):
    """Return (A(Δt), Q(Δt)) for Matern-3/2 via the stationary identity."""
    decay_rate = jnp.sqrt(3.0) / lengthscale
    scaled_time = time_step * decay_rate
    exp_neg_scaled_time = jnp.exp(-scaled_time)
    transition_matrix = exp_neg_scaled_time * jnp.array(
        [
            [1.0 + scaled_time, time_step],
            [-(decay_rate**2) * time_step, 1.0 - scaled_time],
        ]
    )
    stationary_cov = jnp.diag(jnp.stack([variance, 3.0 * variance / lengthscale**2]))
    process_noise_cov = (
        stationary_cov - transition_matrix @ stationary_cov @ transition_matrix.T
    )
    return transition_matrix, 0.5 * (process_noise_cov + process_noise_cov.T)


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
            transition_matrix, process_noise_cov = _matern32_discrete_transition_and_q(
                dt, self.lengthscale, self.variance
            )
            return transition_matrix, _psd_sqrt(process_noise_cov)

        return jax.lax.cond(time_step == 0.0, zero_dt, positive_dt, time_step)


def _matern52_discrete_transition_and_q(time_step, lengthscale, variance):
    """Return (A(Δt), Q(Δt)) for Matern-5/2 via the stationary identity."""
    decay_rate = jnp.sqrt(5.0) / lengthscale
    scaled_time = time_step * decay_rate
    transition_matrix = jnp.exp(-scaled_time) * (
        time_step
        * jnp.array(
            [
                [
                    decay_rate * (0.5 * scaled_time + 1.0),
                    scaled_time + 1.0,
                    0.5 * time_step,
                ],
                [
                    -0.5 * scaled_time * decay_rate**2,
                    decay_rate * (1.0 - scaled_time),
                    1.0 - 0.5 * scaled_time,
                ],
                [
                    decay_rate**3 * (0.5 * scaled_time - 1.0),
                    decay_rate**2 * (scaled_time - 3.0),
                    decay_rate * (0.5 * scaled_time - 2.0),
                ],
            ]
        )
        + jnp.eye(3)
    )
    kappa = 5.0 / 3.0 * variance / lengthscale**2
    stationary_cov = jnp.array(
        [
            [variance, 0.0, -kappa],
            [0.0, kappa, 0.0],
            [-kappa, 0.0, 25.0 * variance / lengthscale**4],
        ]
    )
    process_noise_cov = (
        stationary_cov - transition_matrix @ stationary_cov @ transition_matrix.T
    )
    return transition_matrix, 0.5 * (process_noise_cov + process_noise_cov.T)


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
            transition_matrix, process_noise_cov = _matern52_discrete_transition_and_q(
                dt, self.lengthscale, self.variance
            )
            return transition_matrix, _psd_sqrt(process_noise_cov)

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


class ProductSDE(LinearSDE):
    """Kronecker state-space representation of TruncatedPeriodic × stationary-Matérn.

    For independent processes ``f_p ~ GP(0, k_p)`` (periodic) and
    ``f_m ~ GP(0, k_m)`` (Matérn), the product process
    ``f(t) = f_p(t) f_m(t)`` has covariance ``k_p(τ) k_m(τ)`` and an exact
    finite-dimensional state-space realisation on the Kronecker-product state
    ``x(t) = x_p(t) ⊗ x_m(t)`` (Solin & Särkkä 2014 §3):

        F = F_p ⊗ I_m + I_p ⊗ F_m
        L = S_p ⊗ L_m     (the periodic factor's own diffusion L_p is zero)
        Qc = I_p ⊗ Qc_m
        H = H_p ⊗ H_m
        P_∞ = P_∞,p ⊗ P_∞,m   (sqrt: S = S_p ⊗ S_m)

    where ``S_p``, ``S_m`` are the factors' ``stationary_state_cov_sqrt``.
    This relies on ``TruncatedPeriodicSDE`` having zero own diffusion (its
    transition is an exact rotation per harmonic, so its own discrete
    process noise is exactly zero — see ``test_truncated_periodic_L_Q_is_zero``);
    ``components`` must therefore be ``(periodic_factor, matern_factor)`` in
    that order. This ordering is enforced by ``to_sde``'s ``ProductKernel``
    dispatch, the only production caller.

    The state dimension is the *product* of the two factor state
    dimensions, so this is only used where both factors have small state
    dimension — e.g. ``TruncatedPeriodic`` × Matérn, whose state dimension
    is ``(2K + 1) · d`` (Solin & Särkkä 2014 §3.2) — not for arbitrary
    products, which is why ``to_sde`` gates which kernel pairs reach here.

    ``discretise`` avoids ever forming or eigendecomposing the full
    ``state_dim × state_dim`` stationary covariance. Two exact identities
    let it reuse each factor's own closed-form ``discretise`` instead:
    ``A(Δt) = expm(F Δt) = A_p(Δt) ⊗ A_m(Δt)``, since ``F_p ⊗ I_m`` and
    ``I_p ⊗ F_m`` commute; and, because the periodic factor is lossless
    (``A_p(t) P_∞,p A_p(t)ᵀ = P_∞,p`` for all ``t``), the discrete process
    noise factorises as ``Q(Δt) = P_∞,p ⊗ Q_m(Δt)`` with square root
    ``L_Q(Δt) = S_p ⊗ L_Q,m(Δt)``. A generic eigendecomposition-based square
    root of ``P_∞ − A P_∞ Aᵀ`` (as used by the Matérn SDEs) is deliberately
    avoided here: the periodic factor's per-harmonic eigenvalues repeat in
    pairs (cos/sin components share variance), which makes the combined
    stationary covariance's spectrum degenerate and ``jnp.linalg.eigh``'s
    reverse-mode gradient singular there.
    """

    components: tuple[LinearSDE, LinearSDE]

    def __init__(self, components: tuple[LinearSDE, LinearSDE]):
        periodic_factor, matern_factor = components
        self.components = (periodic_factor, matern_factor)

        eye_periodic = jnp.eye(periodic_factor.state_dim)
        eye_matern = jnp.eye(matern_factor.state_dim)
        stationary_cov_sqrt_periodic = periodic_factor.stationary_state_cov_sqrt
        stationary_cov_sqrt_matern = matern_factor.stationary_state_cov_sqrt

        F = jnp.kron(periodic_factor.drift_matrix, eye_matern) + jnp.kron(
            eye_periodic, matern_factor.drift_matrix
        )
        L = jnp.kron(stationary_cov_sqrt_periodic, matern_factor.diffusion_matrix)
        Qc = jnp.kron(eye_periodic, matern_factor.process_noise_spectral_density)
        H = jnp.kron(
            periodic_factor.observation_matrix, matern_factor.observation_matrix
        )
        L_inf = jnp.kron(stationary_cov_sqrt_periodic, stationary_cov_sqrt_matern)

        super().__init__(
            drift_matrix=F,
            diffusion_matrix=L,
            process_noise_spectral_density=Qc,
            observation_matrix=H,
            stationary_state_cov_sqrt=L_inf,
            state_dim=periodic_factor.state_dim * matern_factor.state_dim,
        )

    def discretise(self, time_step):
        periodic_factor, matern_factor = self.components
        A_periodic, _ = periodic_factor.discretise(time_step)
        A_matern, L_Q_matern = matern_factor.discretise(time_step)
        A = jnp.kron(A_periodic, A_matern)
        L_Q = jnp.kron(periodic_factor.stationary_state_cov_sqrt, L_Q_matern)
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
