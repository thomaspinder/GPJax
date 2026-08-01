"""Merged-grid prediction for state-space GPs.

See plans/2026-04-21-state-space-gps-design.md §Stage 4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import paramax

from gpjax.distributions import GaussianDistribution
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


def predict_smoothed(posterior, train_data, test_inputs, *, observation_mask=None):
    """Smoothed-latent prediction for state-space GPs.

    Returns a ``GaussianDistribution`` over the M test points whose mean is
    the RTS-smoothed posterior mean and whose scale is a
    ``lx.DiagonalLinearOperator`` carrying the smoothed marginal variance plus
    ``prior.jitter``. Test inputs are returned in caller order regardless of
    sorting.

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
    # Time-step deltas on the merged grid.
    sorted_time_steps = jnp.concatenate([jnp.zeros(1), jnp.diff(sorted_times)])

    forward_outputs, _ = _sqrt_filter_forward(
        sde,
        sorted_targets,
        sorted_time_steps,
        sorted_is_observed,
        sigma_eff,
    )
    smoothed_means, smoothed_Ls = rts_smoother(sde, forward_outputs, sorted_time_steps)

    # Recover test-point positions in caller order via the inverse permutation.
    inv_perm = jnp.argsort(merge_perm)
    test_positions_in_sorted = inv_perm[n_train:]

    H = sde.observation_matrix
    test_smoothed_means = smoothed_means[test_positions_in_sorted]
    test_smoothed_Ls = smoothed_Ls[test_positions_in_sorted]
    test_observation_means = jnp.einsum("ij,mj->mi", H, test_smoothed_means).squeeze(-1)
    test_observation_variances = jax.vmap(lambda L: (H @ (L @ L.T) @ H.T).squeeze())(
        test_smoothed_Ls
    )

    # Re-add mean function at test points + add jitter.
    mean_at_test = prior.mean_function(test_inputs).squeeze(-1)
    test_predicted_means = test_observation_means + mean_at_test
    test_predicted_variances = test_observation_variances + prior.jitter

    return GaussianDistribution(
        loc=test_predicted_means,
        scale=lx.DiagonalLinearOperator(test_predicted_variances),
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
