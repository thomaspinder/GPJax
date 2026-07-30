"""Square-root Kalman filter and RTS smoother for state-space GPs.

See plans/2026-04-21-state-space-gps-design.md §Stage 2 and §Stage 3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpjax.state_space.sde import _psd_sqrt


def _sign_normalise(R):
    """Flip row signs so the diagonal of R is non-negative without changing R.T @ R.

    `jnp.linalg.qr` does not guarantee sign on the R-factor diagonal (LAPACK's
    `geqrf` returns arbitrary signs). This breaks square-root Kalman updates,
    where ``R[0, 0]`` is the innovation standard deviation.

    See plans/2026-04-21-state-space-gps-design.md §Stage 2 (`_sign_normalise`).
    """
    row_signs = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)
    return row_signs[:, None] * R


def _sqrt_predict(mean_prev, L_prev, transition_matrix, L_Q):
    """Square-root Kalman predict.

    Predicts mean and covariance-square-root one step forward:

        mean_pred = A @ mean_prev
        L_pred    such that L_pred @ L_pred.T = A @ L_prev @ L_prev.T @ A.T + L_Q @ L_Q.T

    Implementation: stack [(A @ L_prev).T; L_Q.T] and QR-decompose; the resulting
    R factor is an upper-triangular square root of the predicted covariance.
    Sign-normalise so the L_pred diagonal is non-negative.

    See plans/2026-04-21-state-space-gps-design.md §Stage 2.
    """
    mean_pred = transition_matrix @ mean_prev
    pre_array = jnp.concatenate([(transition_matrix @ L_prev).T, L_Q.T], axis=0)
    _, R = jnp.linalg.qr(pre_array, mode="reduced")
    L_pred = _sign_normalise(R).T
    return mean_pred, L_pred


def _sqrt_update(
    mean_pred, L_pred, observation, observation_matrix, observation_stddev
):
    """Square-root Kalman update for a scalar observation.

    Joseph-form square-root update via QR factorisation. Builds the pre-array

        [σ_y          0_{1×n}    ]
        [(H L_pred).T   L_pred.T ]

    of shape ``(1 + n, 1 + n)`` and reads off the innovation standard deviation,
    Kalman gain, and updated covariance square root from the upper-triangular R
    factor. Returns ``(mean_updated, L_updated, log_likelihood_increment)``.

    See plans/2026-04-21-state-space-gps-design.md §Stage 2.
    """
    state_dim = L_pred.shape[0]
    H_L_pred = (observation_matrix @ L_pred).reshape(-1)
    top_row = jnp.concatenate(
        [jnp.atleast_1d(observation_stddev), jnp.zeros(state_dim)]
    )
    bottom_rows = jnp.concatenate([H_L_pred[:, None], L_pred.T], axis=1)
    pre_array = jnp.concatenate([top_row[None, :], bottom_rows], axis=0)
    _, R = jnp.linalg.qr(pre_array, mode="reduced")
    R = _sign_normalise(R)

    innovation_stddev = R[0, 0]
    kalman_gain = R[0, 1:] / innovation_stddev
    L_updated = R[1:, 1:].T

    innovation = observation - (observation_matrix @ mean_pred).reshape(())
    mean_updated = mean_pred + kalman_gain * innovation

    log_likelihood_increment = -0.5 * (
        jnp.log(2.0 * jnp.pi * innovation_stddev**2)
        + (innovation / innovation_stddev) ** 2
    )
    return mean_updated, L_updated, log_likelihood_increment


def _resolve_chunk_size(num_steps: int, chunk_size: int | None) -> int:
    """Resolve a default chunk size of ``round(sqrt(N))`` when ``None``; clamp to ``[1, N]``.

    See plans/2026-04-21-state-space-gps-design.md §Stage 2 (Task 5.5).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if chunk_size is None:
        return max(1, round(num_steps**0.5))
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return min(chunk_size, num_steps)


def kalman_filter(
    sde,
    centred_targets,
    time_steps,
    is_observed,
    sigma_eff,
    *,
    chunk_size=None,
):
    """Square-root Kalman filter marginal log-likelihood.

    Pure-JAX, double-scan with checkpointed inner chunks. Caller is responsible
    for centring targets (subtracting the mean function), sorting by time, and
    computing ``sigma_eff = sqrt(σ_y² + jitter)``.

    The first entry of ``time_steps`` must be ``0.0`` so the first predict step
    is the identity and the filter starts from the prior at ``t = 0``. The
    initial filter state is ``(mean=0, L=sde.stationary_state_cov_sqrt)`` —
    centred targets imply a zero prior mean.

    Memory characteristics: an outer ``lax.scan`` iterates over chunks of
    ``chunk_size`` steps; an inner ``lax.scan`` (wrapped in
    :func:`jax.checkpoint`) runs the per-step filter inside a chunk. Under
    reverse-mode autodiff, only the chunk-boundary carries are saved on the
    tape — interior carries are recomputed on the backward pass — giving an
    AD memory footprint of ``O(sqrt(N) * d^2)`` for the default
    ``chunk_size = round(sqrt(N))``.

    To accommodate ``num_steps`` that is not a multiple of the resolved chunk
    size, the inputs are padded with no-op steps (``time_step = 0`` and
    ``is_observed = False``). Padding steps are exact algebraic no-ops because
    every SDE returns ``(A, L_Q) = (I, 0)`` at ``dt = 0`` and the masked update
    contributes zero log-likelihood.

    Args:
        sde (LinearSDE): State-space SDE; ``sde.discretise(dt)`` is called per
            scan step.
        centred_targets (Float[Array, "num_train"]): Targets with the mean
            function subtracted.
        time_steps (Float[Array, "num_train"]): ``time_steps[0] = 0`` and
            ``time_steps[i] = t_i - t_{i-1}`` for ``i > 0``.
        is_observed (Bool[Array, "num_train"]): At indices where this is
            ``False`` the update step is a no-op (predict only). Useful for
            masked observations.
        sigma_eff (Float[Array, ""]): Effective observation standard deviation.
        chunk_size (int | None, keyword-only): Static chunk length for the inner
            scan. ``None`` resolves to ``round(sqrt(num_steps))`` (clamped to
            ``>= 1``). Values larger than ``num_steps`` are clamped to
            ``num_steps`` (single chunk, no padding).

    Returns:
        Float[Array, ""]: Scalar marginal log-likelihood
            ``Σ_i log p(y_i | y_{<i})``.

    See plans/2026-04-21-state-space-gps-design.md §Stage 2.
    """
    num_steps = int(time_steps.shape[0])
    resolved_chunk_size = _resolve_chunk_size(num_steps, chunk_size)
    pad_len = (-num_steps) % resolved_chunk_size

    if pad_len > 0:
        centred_targets_padded = jnp.concatenate(
            [centred_targets, jnp.zeros(pad_len, dtype=centred_targets.dtype)]
        )
        time_steps_padded = jnp.concatenate(
            [time_steps, jnp.zeros(pad_len, dtype=time_steps.dtype)]
        )
        is_observed_padded = jnp.concatenate(
            [is_observed, jnp.zeros(pad_len, dtype=bool)]
        )
    else:
        centred_targets_padded = centred_targets
        time_steps_padded = time_steps
        is_observed_padded = is_observed

    num_padded_steps = num_steps + pad_len
    num_chunks = num_padded_steps // resolved_chunk_size

    time_steps_chunked = time_steps_padded.reshape(num_chunks, resolved_chunk_size)
    targets_chunked = centred_targets_padded.reshape(num_chunks, resolved_chunk_size)
    is_observed_chunked = is_observed_padded.reshape(num_chunks, resolved_chunk_size)

    mean_init = jnp.zeros(sde.state_dim)
    L_init = sde.stationary_state_cov_sqrt
    observation_matrix = sde.observation_matrix

    def step(carry, scan_input):
        mean_prev, L_prev = carry
        time_step, observation, observation_flag = scan_input

        transition_matrix, L_Q = sde.discretise(time_step)
        mean_pred, L_pred = _sqrt_predict(mean_prev, L_prev, transition_matrix, L_Q)

        def do_update(args):
            mean, L = args
            return _sqrt_update(mean, L, observation, observation_matrix, sigma_eff)

        def skip_update(args):
            mean, L = args
            return mean, L, jnp.asarray(0.0)

        mean_new, L_new, log_likelihood_increment = jax.lax.cond(
            observation_flag,
            do_update,
            skip_update,
            (mean_pred, L_pred),
        )

        return (mean_new, L_new), log_likelihood_increment

    @jax.checkpoint
    def inner_scan(initial_carry, chunk_inputs):
        final_carry, log_likelihood_increments_chunk = jax.lax.scan(
            step, initial_carry, chunk_inputs
        )
        return final_carry, log_likelihood_increments_chunk

    def outer_step(carry, chunk_inputs):
        new_carry, log_likelihood_increments_chunk = inner_scan(carry, chunk_inputs)
        return new_carry, log_likelihood_increments_chunk

    chunk_inputs_packed = (time_steps_chunked, targets_chunked, is_observed_chunked)
    _, log_likelihood_increments_per_chunk = jax.lax.scan(
        outer_step, (mean_init, L_init), chunk_inputs_packed
    )
    log_likelihood_increments = log_likelihood_increments_per_chunk.reshape(-1)[
        :num_steps
    ]
    return jnp.sum(log_likelihood_increments)


def _sqrt_filter_forward(
    sde,
    centred_targets,
    time_steps,
    is_observed,
    sigma_eff,
):
    """Forward filter that returns the per-step trajectory needed by the smoother.

    Unlike :func:`kalman_filter`, this returns the full per-step trajectory:

    - ``means_updated``        -- shape ``(num_steps, state_dim)``, filtered means
    - ``Ls_updated``           -- shape ``(num_steps, state_dim, state_dim)``,
      filtered covariance square-roots
    - ``means_predicted``      -- shape ``(num_steps, state_dim)``, the
      one-step-ahead predicted means at each scan step
    - ``Ls_predicted``         -- shape ``(num_steps, state_dim, state_dim)``,
      the one-step-ahead predicted covariance square-roots

    plus the scalar log-likelihood. The smoother's backward recursion uses the
    filtered state at index ``i`` together with the predicted state at index
    ``i + 1`` (i.e. the forward predict from ``i`` to ``i + 1``).

    Implementation: a single ``lax.scan`` (no chunking) with a richer per-step
    output. The smoother runs at predict time, not under reverse-mode autodiff
    of the marginal log-likelihood, so chunking for AD memory is unnecessary.

    See plans/2026-04-21-state-space-gps-design.md §Stage 3.
    """
    mean_init = jnp.zeros(sde.state_dim)
    L_init = sde.stationary_state_cov_sqrt
    observation_matrix = sde.observation_matrix

    def step(carry, scan_input):
        mean_prev, L_prev = carry
        time_step, observation, observation_flag = scan_input

        transition_matrix, L_Q = sde.discretise(time_step)
        mean_predicted, L_predicted = _sqrt_predict(
            mean_prev, L_prev, transition_matrix, L_Q
        )

        def do_update(args):
            mean, L = args
            return _sqrt_update(mean, L, observation, observation_matrix, sigma_eff)

        def skip_update(args):
            mean, L = args
            return mean, L, jnp.asarray(0.0)

        mean_updated, L_updated, log_likelihood_increment = jax.lax.cond(
            observation_flag,
            do_update,
            skip_update,
            (mean_predicted, L_predicted),
        )
        return (mean_updated, L_updated), (
            mean_updated,
            L_updated,
            mean_predicted,
            L_predicted,
            log_likelihood_increment,
        )

    scan_inputs = (time_steps, centred_targets, is_observed)
    _, outputs = jax.lax.scan(step, (mean_init, L_init), scan_inputs)
    (
        means_updated,
        Ls_updated,
        means_predicted,
        Ls_predicted,
        log_likelihood_increments,
    ) = outputs
    return (
        (means_updated, Ls_updated, means_predicted, Ls_predicted),
        jnp.sum(log_likelihood_increments),
    )


def rts_smoother(sde, forward_outputs, time_steps):
    r"""Square-root RTS smoother.

    Runs the standard Särkkä & Solin (2019) §10.7 backward recursion on the
    forward filter trajectory. Internally materialises ``P = L @ L.T`` for the
    smoother gain computation and re-roots the smoothed covariance after each
    backward step with :func:`gpjax.state_space.sde._psd_sqrt`. The returned
    ``Ls`` are therefore non-triangular ``V·Λ^½`` square roots (neither Cholesky
    nor symmetric); consumers must rely only on ``L @ Lᵀ``. The recursion is

    .. math::

        G_i &= P^{\text{filt}}_i A_{i+1}^\top (P^{\text{pred}}_{i+1})^{-1} \\
        m^{\text{smooth}}_i &= m^{\text{filt}}_i + G_i (m^{\text{smooth}}_{i+1} - m^{\text{pred}}_{i+1}) \\
        P^{\text{smooth}}_i &= P^{\text{filt}}_i + G_i (P^{\text{smooth}}_{i+1} - P^{\text{pred}}_{i+1}) G_i^\top

    The last step has no future, so its smoothed state equals its filtered
    state.

    Args:
        sde (LinearSDE): State-space SDE used in the forward pass;
            ``sde.discretise(dt)`` is called once per backward step.
        forward_outputs (tuple): Quadruple
            ``(means_updated, Ls_updated, means_predicted, Ls_predicted)``
            as returned by :func:`_sqrt_filter_forward`.
        time_steps (Float[Array, "num_train"]): Same ``time_steps`` that drove
            the forward pass; ``time_steps[i+1]`` is the inter-step ``dt``
            between filtered index ``i`` and predicted index ``i + 1``.

    Returns:
        tuple: ``smoothed_means`` of shape ``Float[Array, "num_train state_dim"]``
            and ``smoothed_Ls`` of shape
            ``Float[Array, "num_train state_dim state_dim"]``.

    See plans/2026-04-21-state-space-gps-design.md §Stage 3.
    """
    means_updated, Ls_updated, means_predicted, Ls_predicted = forward_outputs

    def backward_step(carry, scan_input):
        mean_smoothed_next, L_smoothed_next = carry
        (
            mean_filtered,
            L_filtered,
            mean_predicted_next,
            L_predicted_next,
            time_step_next,
        ) = scan_input

        transition_matrix_next, _ = sde.discretise(time_step_next)
        P_filtered = L_filtered @ L_filtered.T
        P_smoothed_next = L_smoothed_next @ L_smoothed_next.T

        # Smoother gain: G = P_filt @ A.T @ inv(P_pred_next), computed via the
        # square-root factor L_predicted_next for better conditioning. Solve
        # ``L L.T x = b`` as two triangular solves.
        cross_cov = transition_matrix_next @ P_filtered  # shape (n, n)
        temp = jax.scipy.linalg.solve_triangular(
            L_predicted_next, cross_cov, lower=True
        )
        smoother_gain = jax.scipy.linalg.solve_triangular(
            L_predicted_next.T, temp, lower=False
        ).T
        P_predicted_next = L_predicted_next @ L_predicted_next.T

        mean_smoothed = mean_filtered + smoother_gain @ (
            mean_smoothed_next - mean_predicted_next
        )
        P_smoothed = (
            P_filtered
            + smoother_gain @ (P_smoothed_next - P_predicted_next) @ smoother_gain.T
        )
        # Symmetrise to kill round-off antisymmetric error, then take a
        # gradient-safe PSD square root (cholesky NaNs on marginally-indefinite
        # round-off; _psd_sqrt clips negative eigenvalues to zero). _psd_sqrt
        # returns V·Λ^½ — a non-triangular root that is neither Cholesky nor
        # symmetric; this is fine because every consumer uses L @ L.T.
        P_smoothed = 0.5 * (P_smoothed + P_smoothed.T)
        L_smoothed = _psd_sqrt(P_smoothed)

        return (mean_smoothed, L_smoothed), (mean_smoothed, L_smoothed)

    # Initial smoother carry = filtered state at the last step (no future).
    init_carry = (means_updated[-1], Ls_updated[-1])

    # Backward scan over indices 0 .. N-2, in reverse:
    #   - filtered state at index i,
    #   - predicted state at index i+1 (forward predict from i to i+1),
    #   - time_step at index i+1 (dt from i to i+1).
    backward_inputs = (
        means_updated[:-1],
        Ls_updated[:-1],
        means_predicted[1:],
        Ls_predicted[1:],
        time_steps[1:],
    )
    backward_inputs_reversed = tuple(jnp.flip(arr, axis=0) for arr in backward_inputs)
    _, smoothed_outputs_reversed = jax.lax.scan(
        backward_step, init_carry, backward_inputs_reversed
    )
    smoothed_means_prefix, smoothed_Ls_prefix = smoothed_outputs_reversed

    # Un-reverse and append the last step (which equals the filtered state).
    smoothed_means = jnp.concatenate(
        [jnp.flip(smoothed_means_prefix, axis=0), means_updated[-1:]], axis=0
    )
    smoothed_Ls = jnp.concatenate(
        [jnp.flip(smoothed_Ls_prefix, axis=0), Ls_updated[-1:]], axis=0
    )
    return smoothed_means, smoothed_Ls
