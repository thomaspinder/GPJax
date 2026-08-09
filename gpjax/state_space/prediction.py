"""Merged-grid prediction for state-space GPs.

See plans/2026-04-21-state-space-gps-design.md §Stage 4.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
import paramax

from gpjax.distributions import GaussianDistribution
from gpjax.linalg.utils import add_jitter
from gpjax.state_space.inference import _sqrt_filter_forward, rts_smoother
from gpjax.state_space.kernels import to_sde


def _merge_grids(train_times, test_times, centred_targets, observation_mask):
    """Build the merged train+test grid for state-space prediction.

    Returns:
        sorted_times ((N+M,)): Combined timestamps in non-decreasing order.
            Train entries sort before test entries when timestamps tie.
        sorted_targets ((N+M,)): Centred targets at training positions; zero at
            test positions (unused because the filter skips updates where
            ``sorted_is_observed`` is False).
        sorted_is_observed ((N+M,) bool): True at training positions retained by
            the input mask; False everywhere else.
        sorted_is_test ((N+M,) int): 0 at training positions, 1 at test
            positions, ordered by the merge permutation.
        merge_perm ((N+M,) int): The permutation array such that the sorted
            views above are ``concat([train, test])[merge_perm]``. Caller can
            compute ``inv_perm = argsort(merge_perm)`` and gather
            ``inv_perm[N:]`` to recover test-point positions in caller order.
    """
    n_train = train_times.shape[0]
    n_test = test_times.shape[0]

    all_times = jnp.concatenate([train_times, test_times])
    all_targets = jnp.concatenate([centred_targets, jnp.zeros(n_test)])
    all_observed = jnp.concatenate([observation_mask, jnp.zeros(n_test, dtype=bool)])
    is_test = jnp.concatenate(
        [jnp.zeros(n_train, dtype=jnp.int32), jnp.ones(n_test, dtype=jnp.int32)]
    )
    # lexsort: last row is primary key.
    merge_perm = jnp.lexsort(jnp.stack([is_test, all_times]))

    sorted_times = all_times[merge_perm]
    sorted_targets = all_targets[merge_perm]
    sorted_is_observed = all_observed[merge_perm]
    sorted_is_test = is_test[merge_perm]
    return sorted_times, sorted_targets, sorted_is_observed, sorted_is_test, merge_perm


def _test_segment_operators(smoother_gains, sorted_is_test, num_test):
    r"""Cumulative smoother-gain products between consecutive test points.

    The RTS smoother cross-covariance recursion (Särkkä & Solin 2019 §12.2)
    gives :math:`\mathrm{Cov}(x_i, x_j \mid y_{1:T}) = G_i G_{i+1} \cdots
    G_{j-1} P_j^{\text{smooth}}` for merged-grid indices :math:`i < j`. This
    function computes, for each pair of *consecutive* test points in the
    merged grid, the cumulative gain product :math:`G_{p_a} \cdots
    G_{p_{a+1}-1}` connecting them — the reusable building block that
    :func:`_dense_test_covariance` chains together for arbitrary (not just
    adjacent) test-point pairs.

    A single forward pass over the merged grid accumulates the running
    product since the last test point was seen, resetting to the identity at
    each test point and emitting the accumulated product the next time a test
    point is reached. This costs :math:`O((N + M) d^3)` — linear in the
    training set size :math:`N`, not quadratic — since it never touches a
    pair of training points.

    Args:
        smoother_gains (Float[Array, "grid_size-1 state_dim state_dim"]): Per-step
            smoother gains from :func:`gpjax.state_space.inference.rts_smoother`
            with ``return_gains=True``, on the merged train-plus-test grid.
        sorted_is_test (Bool[Array, " grid_size"]): Merged-grid mask, ``True``
            at test positions.
        num_test (int): Number of test points ``M`` (static; the size of the
            ``True`` entries in ``sorted_is_test``).

    Returns:
        Float[Array, "num_test-1 state_dim state_dim"]: ``segments[a]`` is the
            operator connecting the ``a``-th and ``(a + 1)``-th test point in
            merged-grid order, so that
            :math:`\mathrm{Cov}(x_{p_a}, x_{p_{a+1}}) = \text{segments}[a]\,
            P_{p_{a+1}}^{\text{smooth}}`.
    """
    state_dim = smoother_gains.shape[-1]
    is_test_bool = sorted_is_test.astype(bool)
    identity = jnp.eye(state_dim)

    def step(carry, step_inputs):
        running_product, has_started = carry
        gain, is_test_here, is_test_next = step_inputs
        # Reset the running product to the identity right at a test point, so
        # it starts fresh with `gain` as its first factor.
        base = jnp.where(is_test_here, identity, running_product)
        running_product_next = base @ gain
        has_started_next = has_started | is_test_here
        # Emit only once accumulation has actually started (skips any
        # training-only prefix before the first test point).
        emit = is_test_next & has_started_next
        return (running_product_next, has_started_next), (running_product_next, emit)

    init_carry = (identity, jnp.array(False))
    step_inputs = (smoother_gains, is_test_bool[:-1], is_test_bool[1:])
    _, (candidates, emit_mask) = jax.lax.scan(step, init_carry, step_inputs)

    num_segments = num_test - 1
    emit_indices = jnp.nonzero(emit_mask, size=num_segments)[0]
    return candidates[emit_indices]


def _dense_test_covariance(segments, state_covs_at_test, observation_matrix):
    r"""Dense observation-space covariance across test points from smoother segments.

    Builds the full :math:`M \times M` covariance in the latent *state* space
    by chaining :func:`_test_segment_operators`' consecutive-pair operators —
    without ever inverting a gain product, which would be ill-conditioned for
    widely separated points since gains shrink towards zero with lag — then
    projects through the observation matrix. For each test point ``b``, a
    backward scan over the segments before it accumulates
    :math:`\mathrm{Cov}(x_a, x_b)` for every ``a < b`` in one pass, mirroring
    the RTS backward recursion itself but over the (typically far smaller)
    test-point chain rather than the full merged grid. Cost is
    :math:`O(M^2 d^3)`, independent of the training set size and no larger
    than the :math:`O(M^2)` already required to store the dense output.

    Args:
        segments (Float[Array, "num_test-1 state_dim state_dim"]): Consecutive-pair
            operators from :func:`_test_segment_operators`.
        state_covs_at_test (Float[Array, "num_test state_dim state_dim"]): Smoothed
            state covariances :math:`P_b^{\text{smooth}}` at each test point,
            in the same (merged-grid) order as ``segments``.
        observation_matrix (Float[Array, "1 state_dim"]): The SDE's observation
            matrix :math:`H`.

    Returns:
        Float[Array, "num_test num_test"]: The dense observation-space
            covariance matrix, in the same order as ``state_covs_at_test``.
    """
    num_test, state_dim, _ = state_covs_at_test.shape
    H_row = observation_matrix.reshape(-1)
    diag_variances = jnp.einsum("i,nij,j->n", H_row, state_covs_at_test, H_row)

    if num_test <= 1:
        # `segments` is empty (shape (0, state_dim, state_dim)); indexing it
        # even abstractly (as `column_for` would, under `jax.lax.scan`
        # tracing) is a static shape error in JAX, so skip it entirely.
        return jnp.diag(diag_variances)

    def column_for(b):
        def body(carry, a_rev):
            apply_segment = a_rev < b
            new_carry = jnp.where(apply_segment, segments[a_rev] @ carry, carry)
            return new_carry, new_carry

        a_rev_range = jnp.arange(num_test - 2, -1, -1)
        _, outputs = jax.lax.scan(body, state_covs_at_test[b], a_rev_range)
        column = jnp.zeros((num_test, state_dim, state_dim))
        return column.at[a_rev_range].set(outputs)

    # columns[b, a] = Cov(x_a, x_b) for a < b; entries with a >= b are unused.
    columns = jax.vmap(column_for)(jnp.arange(num_test))
    obs_cross = jnp.einsum("i,baij,j->ba", H_row, columns, H_row)

    lower_valid = jnp.tril(obs_cross, k=-1)
    off_diagonal = lower_valid + lower_valid.T
    return off_diagonal + jnp.diag(diag_variances)


def _dense_smoothed_test_covariance(
    smoother_gains,
    sorted_is_test,
    smoothed_Ls,
    test_positions_in_sorted,
    observation_matrix,
):
    r"""The dense test-test predictive covariance, in caller order.

    Combines :func:`_test_segment_operators` and :func:`_dense_test_covariance`,
    handling the sort into merged-grid order (both helpers above assume it)
    and the sort back into caller order.

    Args:
        smoother_gains (Float[Array, "grid_size-1 state_dim state_dim"]): Per-step
            smoother gains, as returned by ``rts_smoother(..., return_gains=True)``.
        sorted_is_test (Bool[Array, " grid_size"]): Merged-grid mask, ``True``
            at test positions.
        smoothed_Ls (Float[Array, "grid_size state_dim state_dim"]): Smoothed
            covariance square roots at every merged-grid position.
        test_positions_in_sorted (Int[Array, " num_test"]): For caller-order
            test index ``m``, the merged-grid position of that test point (as
            returned by the ``inv_perm`` construction in :func:`predict_smoothed`).
        observation_matrix (Float[Array, "1 state_dim"]): The SDE's observation
            matrix :math:`H`.

    Returns:
        Float[Array, "num_test num_test"]: The dense observation-space
            covariance matrix, in caller order.
    """
    num_test = test_positions_in_sorted.shape[0]
    sort_order = jnp.argsort(test_positions_in_sorted)
    grid_order_positions = test_positions_in_sorted[sort_order]
    unsort_order = jnp.argsort(sort_order)

    segments = _test_segment_operators(smoother_gains, sorted_is_test, num_test)
    state_covs_grid_order = jax.vmap(lambda L: L @ L.T)(
        smoothed_Ls[grid_order_positions]
    )
    covariance_grid_order = _dense_test_covariance(
        segments, state_covs_grid_order, observation_matrix
    )
    return covariance_grid_order[unsort_order][:, unsort_order]


def predict_smoothed(
    posterior,
    train_data,
    test_inputs,
    *,
    observation_mask=None,
    covariance: Literal["dense", "diagonal"] = "diagonal",
):
    """Smoothed-latent prediction for state-space GPs.

    Returns a ``GaussianDistribution`` over the M test points whose mean is
    the RTS-smoothed posterior mean. When ``covariance="diagonal"`` (the
    default) the scale is a ``lx.DiagonalLinearOperator`` carrying the
    smoothed marginal variance plus ``prior.jitter``. When
    ``covariance="dense"`` the scale is an ``lx.MatrixLinearOperator``
    carrying the full :math:`M \\times M` joint covariance, built from the
    RTS smoother's cross-covariance recursion
    (:func:`_dense_smoothed_test_covariance`) rather than a dense
    :math:`N \\times N` gram over the training set. Test inputs are returned
    in caller order regardless of sorting.

    See plans/2026-04-21-state-space-gps-design.md §Stage 4.
    """
    posterior = paramax.unwrap(posterior)
    prior = posterior.prior
    likelihood = posterior.likelihood

    train_X = train_data.X
    train_y = train_data.y
    train_times = train_X.squeeze(-1)
    train_targets = train_y.squeeze(-1)
    test_times = test_inputs.squeeze(-1)
    n_train = train_times.shape[0]

    mean_at_train = prior.mean_function(train_X).squeeze(-1)
    centred_targets = train_targets - mean_at_train

    sde = to_sde(prior.kernel)
    obs_variance = likelihood.obs_stddev**2
    sigma_eff = jnp.sqrt(obs_variance + prior.jitter)

    if observation_mask is None:
        observation_mask = jnp.ones(n_train, dtype=bool)

    (
        sorted_times,
        sorted_targets,
        sorted_is_observed,
        sorted_is_test,
        merge_perm,
    ) = _merge_grids(train_times, test_times, centred_targets, observation_mask)
    # Time-step deltas on the merged grid.
    sorted_time_steps = jnp.concatenate([jnp.zeros(1), jnp.diff(sorted_times)])

    forward_outputs, _ = _sqrt_filter_forward(
        sde,
        sorted_targets,
        sorted_time_steps,
        sorted_is_observed,
        sigma_eff,
    )
    if covariance == "dense":
        smoothed_means, smoothed_Ls, smoother_gains = rts_smoother(
            sde, forward_outputs, sorted_time_steps, return_gains=True
        )
    else:
        smoothed_means, smoothed_Ls = rts_smoother(
            sde, forward_outputs, sorted_time_steps
        )

    # Recover test-point positions in caller order via the inverse permutation.
    inv_perm = jnp.argsort(merge_perm)
    test_positions_in_sorted = inv_perm[n_train:]

    H = sde.observation_matrix
    test_smoothed_means = smoothed_means[test_positions_in_sorted]
    test_observation_means = jnp.einsum("ij,mj->mi", H, test_smoothed_means).squeeze(-1)

    # Re-add mean function at test points + add jitter.
    mean_at_test = prior.mean_function(test_inputs).squeeze(-1)
    test_predicted_means = test_observation_means + mean_at_test

    if covariance == "dense":
        test_covariance = _dense_smoothed_test_covariance(
            smoother_gains,
            sorted_is_test,
            smoothed_Ls,
            test_positions_in_sorted,
            H,
        )
        scale = lx.MatrixLinearOperator(add_jitter(test_covariance, prior.jitter))
    else:
        test_smoothed_Ls = smoothed_Ls[test_positions_in_sorted]
        test_observation_variances = jax.vmap(
            lambda L: (H @ (L @ L.T) @ H.T).squeeze()
        )(test_smoothed_Ls)
        test_predicted_variances = test_observation_variances + prior.jitter
        scale = lx.DiagonalLinearOperator(test_predicted_variances)

    return GaussianDistribution(
        loc=test_predicted_means,
        scale=scale,
    )


def predict_filtered(posterior, train_data, test_inputs, *, observation_mask=None):
    """Filtered (causal) prediction for state-space GPs.

    Each test point conditions only on training observations at timestamps
    less than or equal to the test timestamp (with train < test tie-break).
    Marginals are extracted from the forward filter trajectory rather than the
    smoother.

    See plans/2026-04-21-state-space-gps-design.md §Stage 4.
    """
    posterior = paramax.unwrap(posterior)
    prior = posterior.prior
    likelihood = posterior.likelihood

    train_X = train_data.X
    train_y = train_data.y
    train_times = train_X.squeeze(-1)
    train_targets = train_y.squeeze(-1)
    test_times = test_inputs.squeeze(-1)
    n_train = train_times.shape[0]

    mean_at_train = prior.mean_function(train_X).squeeze(-1)
    centred_targets = train_targets - mean_at_train

    sde = to_sde(prior.kernel)
    obs_variance = likelihood.obs_stddev**2
    sigma_eff = jnp.sqrt(obs_variance + prior.jitter)

    if observation_mask is None:
        observation_mask = jnp.ones(n_train, dtype=bool)

    (
        sorted_times,
        sorted_targets,
        sorted_is_observed,
        _sorted_is_test,
        merge_perm,
    ) = _merge_grids(train_times, test_times, centred_targets, observation_mask)
    sorted_time_steps = jnp.concatenate([jnp.zeros(1), jnp.diff(sorted_times)])

    (means_updated, Ls_updated, _means_predicted, _Ls_predicted), _ = (
        _sqrt_filter_forward(
            sde,
            sorted_targets,
            sorted_time_steps,
            sorted_is_observed,
            sigma_eff,
        )
    )

    inv_perm = jnp.argsort(merge_perm)
    test_positions_in_sorted = inv_perm[n_train:]

    H = sde.observation_matrix
    test_filtered_means = means_updated[test_positions_in_sorted]
    test_filtered_Ls = Ls_updated[test_positions_in_sorted]
    test_observation_means = jnp.einsum("ij,mj->mi", H, test_filtered_means).squeeze(-1)
    test_observation_variances = jax.vmap(lambda L: (H @ (L @ L.T) @ H.T).squeeze())(
        test_filtered_Ls
    )

    mean_at_test = prior.mean_function(test_inputs).squeeze(-1)
    test_predicted_means = test_observation_means + mean_at_test
    test_predicted_variances = test_observation_variances + prior.jitter

    return GaussianDistribution(
        loc=test_predicted_means,
        scale=lx.DiagonalLinearOperator(test_predicted_variances),
    )
