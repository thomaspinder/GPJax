"""Tests for state-space prediction (smoothed and filtered)."""

import gpjax as gpx
from gpjax.distributions import GaussianDistribution
from gpjax.state_space.gps import StateSpacePrior
from gpjax.state_space.prediction import _merge_grids
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from tests.test_state_space.test_filter import _build_matern12_dataset


def test_merge_grids_returns_sorted_times():
    train_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    test_times = jnp.array([0.5, 2.5, 1.5])
    centred_targets = jnp.array([0.1, 0.2, 0.3, 0.4])
    observation_mask = jnp.array([True, True, True, True])
    sorted_times, _, _, _, _ = _merge_grids(
        train_times,
        test_times,
        centred_targets,
        observation_mask,
    )
    expected_sorted = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_allclose(np.asarray(sorted_times), expected_sorted, atol=1e-12)


def test_merge_grids_tie_break_train_before_test():
    """When a train timestamp duplicates a test timestamp, train sorts first."""
    train_times = jnp.array([0.0, 1.0, 2.0])
    test_times = jnp.array([1.0])  # duplicates train index 1
    centred_targets = jnp.array([0.1, 0.2, 0.3])
    observation_mask = jnp.array([True, True, True])
    _, _, sorted_is_observed, sorted_is_test, _ = _merge_grids(
        train_times,
        test_times,
        centred_targets,
        observation_mask,
    )
    # Sorted order: 0.0(train), 1.0(train), 1.0(test), 2.0(train).
    np.testing.assert_array_equal(np.asarray(sorted_is_test), np.array([0, 0, 1, 0]))
    np.testing.assert_array_equal(
        np.asarray(sorted_is_observed), np.array([True, True, False, True])
    )


def test_merge_grids_test_positions_recover_caller_order():
    """The inverse permutation must recover original caller order for test inputs."""
    train_times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    test_times = jnp.array([2.5, 0.5, 3.5])  # deliberately unsorted
    centred_targets = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    observation_mask = jnp.array([True, True, True, True, True])
    sorted_times, _, _, _, merge_perm = _merge_grids(
        train_times,
        test_times,
        centred_targets,
        observation_mask,
    )
    inv_perm = jnp.argsort(merge_perm)
    n = train_times.shape[0]
    test_positions_in_sorted = inv_perm[n:]  # (3,)
    # The j-th test position (caller order j) corresponds to test_times[j] in the sorted array.
    recovered_test_times = sorted_times[test_positions_in_sorted]
    np.testing.assert_allclose(
        np.asarray(recovered_test_times), np.asarray(test_times), atol=1e-12
    )


def test_merge_grids_train_observation_mask_propagates():
    """Train entries marked unobserved must remain unobserved post-merge."""
    train_times = jnp.array([0.0, 1.0, 2.0])
    test_times = jnp.array([1.5])
    centred_targets = jnp.array([0.1, 0.2, 0.3])
    observation_mask = jnp.array([True, False, True])  # mask out training index 1
    _, _, sorted_is_observed, sorted_is_test, _ = _merge_grids(
        train_times,
        test_times,
        centred_targets,
        observation_mask,
    )
    # Sorted indices: 0.0(train, observed), 1.0(train, masked), 1.5(test, ignore), 2.0(train, observed).
    np.testing.assert_array_equal(
        np.asarray(sorted_is_observed), np.array([True, False, False, True])
    )
    np.testing.assert_array_equal(np.asarray(sorted_is_test), np.array([0, 0, 1, 0]))


@pytest.mark.parametrize(
    "kernel_class",
    [
        gpx.kernels.Matern12,
        gpx.kernels.Matern32,
        gpx.kernels.Matern52,
    ],
)
@pytest.mark.parametrize("jitter", [0.0, 1e-6])
def test_state_space_posterior_predict_smoothed_matches_dense_gp(kernel_class, jitter):
    """StateSpaceConjugateModel.predict matches dense GP at training-and-test."""
    lengthscale = 1.5
    variance = 0.8
    obs_stddev = 0.2
    n_train = 30
    n_test = 10

    X_train, y_train = _build_matern12_dataset(
        n=n_train,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    Xtest = jnp.linspace(-0.5, 10.5, n_test).reshape(-1, 1)

    kernel = kernel_class(lengthscale=lengthscale, variance=variance)

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel,
        jitter=jitter,
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    ss_posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))

    ss_dist = ss_posterior.predict(Xtest, train_data)
    ss_means = np.asarray(ss_dist.mean)
    ss_variances = np.asarray(ss_dist.variance)

    # Dense reference.
    dense_kernel = kernel_class(lengthscale=lengthscale, variance=variance)
    dense_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=dense_kernel,
        jitter=jitter,
    )
    dense_posterior = dense_prior * likelihood
    dense_dist = dense_posterior.predict(Xtest, train_data, covariance="diagonal")
    dense_means = np.asarray(dense_dist.mean)
    dense_variances = np.asarray(dense_dist.variance)

    np.testing.assert_allclose(ss_means, dense_means, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(ss_variances, dense_variances, atol=1e-5, rtol=1e-6)


def test_state_space_posterior_predict_smoothed_returns_diagonal_distribution():
    """Returned scale must be a DiagonalLinearOperator (not dense)."""
    n = 10
    X, y = _build_matern12_dataset(n=n, lengthscale=1.0, variance=1.0, obs_stddev=0.2)
    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.2)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    Xtest = jnp.linspace(0.0, 10.0, 5).reshape(-1, 1)
    dist = posterior.predict(Xtest, train_data)
    assert isinstance(dist, GaussianDistribution)
    assert isinstance(dist.scale, lx.DiagonalLinearOperator)


def test_state_space_posterior_predict_observation_mask_equivalent_to_dropping():
    """Masked training entries are not conditioned on; equivalent to dropping them."""
    n_train = 25
    n_test = 8
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n_train,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    Xtest = jnp.linspace(0.0, 10.0, n_test).reshape(-1, 1)

    mask_indices = [3, 11, 17]
    is_observed = jnp.ones(n_train, dtype=bool)
    for i in mask_indices:
        is_observed = is_observed.at[i].set(False)

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior_with_mask = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))
    masked_dist = posterior_with_mask.predict(
        Xtest, train_data, observation_mask=is_observed
    )

    # Reference: drop the masked points entirely.
    keep = np.array(is_observed)
    X_kept = X_train[keep]
    y_kept = y_train[keep]
    train_data_kept = gpx.Dataset(X=X_kept.reshape(-1, 1), y=y_kept.reshape(-1, 1))
    likelihood_kept = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior_kept = ss_prior * likelihood_kept
    dropped_dist = posterior_kept.predict(Xtest, train_data_kept)

    np.testing.assert_allclose(
        np.asarray(masked_dist.mean), np.asarray(dropped_dist.mean), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(masked_dist.variance), np.asarray(dropped_dist.variance), atol=1e-9
    )


def test_state_space_posterior_predict_preserves_caller_order_unsorted_test_inputs():
    """Test inputs in arbitrary order — outputs must align to caller order."""
    n_train = 20
    lengthscale, variance, obs_stddev = 1.0, 1.0, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n_train,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )

    Xtest_caller_order = jnp.array([5.5, 0.5, 9.5, 2.5, 7.5]).reshape(-1, 1)
    Xtest_sorted = jnp.sort(Xtest_caller_order, axis=0)

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))

    dist_caller = posterior.predict(Xtest_caller_order, train_data)
    dist_sorted = posterior.predict(Xtest_sorted, train_data)

    sort_perm = np.argsort(np.asarray(Xtest_caller_order).squeeze())
    inv_sort = np.argsort(sort_perm)
    np.testing.assert_allclose(
        np.asarray(dist_caller.mean),
        np.asarray(dist_sorted.mean)[inv_sort],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(dist_caller.variance),
        np.asarray(dist_sorted.variance)[inv_sort],
        atol=1e-9,
    )


def test_state_space_posterior_predict_handles_test_at_train_timestamp():
    """Test point coinciding with a training timestamp: tie-break = train < test."""
    n = 5
    X_train = jnp.linspace(0.0, 4.0, n).reshape(-1, 1)
    y_train = jnp.zeros((n, 1))
    Xtest = jnp.array([[2.0]])  # exactly equals X_train[2]

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.1)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train, y=y_train)
    dist = posterior.predict(Xtest, train_data)
    assert jnp.isfinite(dist.mean).all()
    assert jnp.isfinite(dist.variance).all()
    assert dist.variance.shape == (1,)


def test_state_space_posterior_predict_filter_dense_raises():
    n = 5
    X, y = _build_matern12_dataset(n=n, lengthscale=1.0, variance=1.0, obs_stddev=0.2)
    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=1.0, variance=1.0),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=0.2)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))
    with pytest.raises(NotImplementedError, match=r"diagonal|dense"):
        posterior.predict_filter(jnp.array([[0.5]]), train_data, covariance="dense")


def test_state_space_posterior_predict_filter_runs_and_returns_diagonal():
    n = 20
    lengthscale, variance, obs_stddev = 1.0, 1.0, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    Xtest = jnp.linspace(0.0, 10.0, 7).reshape(-1, 1)
    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))
    dist = posterior.predict_filter(Xtest, train_data)

    assert isinstance(dist, GaussianDistribution)
    assert isinstance(dist.scale, lx.DiagonalLinearOperator)
    assert dist.mean.shape == (7,)
    assert dist.variance.shape == (7,)
    assert jnp.isfinite(dist.mean).all()
    assert jnp.isfinite(dist.variance).all()


def test_state_space_predict_filter_uses_only_past_observations():
    """A test point at time T conditions only on training points at t <= T.

    Reference: train a dense GP only on the prefix at t <= T, predict at T;
    the filtered prediction at T should match this prefix-conditioned dense GP.
    """
    n = 25
    lengthscale, variance, obs_stddev = 1.0, 1.0, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    test_time = 5.0
    Xtest = jnp.array([[test_time]])

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))
    filtered_dist = posterior.predict_filter(Xtest, train_data)

    # Prefix-conditioned dense GP.
    prefix_mask = X_train <= test_time
    X_prefix = X_train[prefix_mask].reshape(-1, 1)
    y_prefix = y_train[prefix_mask].reshape(-1, 1)
    prefix_data = gpx.Dataset(X=X_prefix, y=y_prefix)
    n_prefix = X_prefix.shape[0]
    likelihood_prefix = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    dense_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    dense_posterior = dense_prior * likelihood_prefix
    dense_dist = dense_posterior.predict(Xtest, prefix_data, covariance="diagonal")

    np.testing.assert_allclose(
        np.asarray(filtered_dist.mean), np.asarray(dense_dist.mean), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(filtered_dist.variance), np.asarray(dense_dist.variance), atol=1e-5
    )


def test_state_space_predict_smoothed_with_constant_mean_function():
    n = 30
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    constant_offset = 3.5
    y_train_offset = y_train + constant_offset
    Xtest = jnp.linspace(0.0, 10.0, 7).reshape(-1, 1)

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Constant(constant=jnp.array(constant_offset)),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train_offset.reshape(-1, 1))
    ss_dist = posterior.predict(Xtest, train_data)
    ss_means = np.asarray(ss_dist.mean)

    # Dense reference with same constant mean.
    dense_prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(constant=jnp.array(constant_offset)),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    dense_posterior = dense_prior * likelihood
    dense_dist = dense_posterior.predict(Xtest, train_data, covariance="diagonal")
    dense_means = np.asarray(dense_dist.mean)

    np.testing.assert_allclose(ss_means, dense_means, atol=1e-5, rtol=1e-6)


def test_state_space_predict_filter_with_constant_mean_function():
    n = 30
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    constant_offset = -1.2
    y_train_offset = y_train + constant_offset
    Xtest = jnp.linspace(0.0, 10.0, 7).reshape(-1, 1)

    ss_prior = StateSpacePrior(
        mean_function=gpx.mean_functions.Constant(constant=jnp.array(constant_offset)),
        kernel=gpx.kernels.Matern12(lengthscale=lengthscale, variance=variance),
    )
    likelihood = gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    posterior = ss_prior * likelihood
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train_offset.reshape(-1, 1))
    dist = posterior.predict_filter(Xtest, train_data)
    means = np.asarray(dist.mean)
    assert np.all(np.isfinite(means))
    # At least one mean should be on the same side of 0 as constant_offset.
    assert means[len(means) // 2] < 0  # given offset = -1.2
