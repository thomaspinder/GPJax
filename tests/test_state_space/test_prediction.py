"""Tests for state-space prediction (smoothed and filtered)."""

import gpjax as gpx
from gpjax.distributions import GaussianDistribution
from gpjax.state_space.gps import StateSpacePrior
from gpjax.state_space.inference import rts_smoother
from gpjax.state_space.prediction import _merge_grids
import jax
import jax.numpy as jnp
import jax.random as jr
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


# ---------------------------------------------------------------------------
# Dense joint predictive covariance (issue #651)
# ---------------------------------------------------------------------------


def _build_smoothed_dense_comparison(kernel_class, jitter, n_train=30, n_test=10):
    """Shared setup for the dense-vs-dense-GP comparison tests below."""
    lengthscale, variance, obs_stddev = 1.5, 0.8, 0.2
    X_train, y_train = _build_matern12_dataset(
        n=n_train,
        lengthscale=lengthscale,
        variance=variance,
        obs_stddev=obs_stddev,
    )
    Xtest = jnp.linspace(-0.5, 10.5, n_test).reshape(-1, 1)
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))

    ss_posterior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel_class(lengthscale=lengthscale, variance=variance),
        jitter=jitter,
    ) * gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    dense_posterior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=kernel_class(lengthscale=lengthscale, variance=variance),
        jitter=jitter,
    ) * gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    return ss_posterior, dense_posterior, train_data, Xtest


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
@pytest.mark.parametrize("jitter", [0.0, 1e-6])
def test_state_space_posterior_predict_smoothed_dense_matches_dense_gp_joint(
    kernel_class, jitter
):
    """Acceptance criterion (issue #651): the dense joint predictive — mean
    *and* full covariance — must match the dense ``ConjugateModel`` predictive
    to ~1e-8."""
    ss_posterior, dense_posterior, train_data, Xtest = _build_smoothed_dense_comparison(
        kernel_class, jitter
    )

    ss_dist = ss_posterior.predict(Xtest, train_data, covariance="dense")
    dense_dist = dense_posterior.predict(Xtest, train_data, covariance="dense")

    assert isinstance(ss_dist.scale, lx.MatrixLinearOperator)
    np.testing.assert_allclose(
        np.asarray(ss_dist.mean), np.asarray(dense_dist.mean), atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(ss_dist.covariance_matrix),
        np.asarray(dense_dist.covariance_matrix),
        atol=1e-8,
    )


def test_state_space_posterior_predict_smoothed_dense_diagonal_matches_diagonal_mode():
    """The diagonal of the dense covariance must equal the diagonal-mode variances."""
    ss_posterior, _, train_data, Xtest = _build_smoothed_dense_comparison(
        gpx.kernels.Matern32, jitter=1e-6
    )

    dense_dist = ss_posterior.predict(Xtest, train_data, covariance="dense")
    diagonal_dist = ss_posterior.predict(Xtest, train_data, covariance="diagonal")

    np.testing.assert_allclose(
        np.diag(np.asarray(dense_dist.covariance_matrix)),
        np.asarray(diagonal_dist.variance),
        atol=1e-10,
    )


def test_state_space_posterior_predict_smoothed_dense_preserves_caller_order():
    """Unsorted test inputs must produce a covariance matrix permuted the same
    way as sorted test inputs — this exercises the grid-order/caller-order
    bookkeeping unique to the dense path."""
    ss_posterior, _, train_data, _ = _build_smoothed_dense_comparison(
        gpx.kernels.Matern32, jitter=1e-6
    )
    Xtest_caller_order = jnp.array([5.5, 0.5, 9.5, 2.5, 7.5]).reshape(-1, 1)
    sort_perm = np.argsort(np.asarray(Xtest_caller_order).squeeze())
    Xtest_sorted = Xtest_caller_order[sort_perm]

    dist_caller = ss_posterior.predict(
        Xtest_caller_order, train_data, covariance="dense"
    )
    dist_sorted = ss_posterior.predict(Xtest_sorted, train_data, covariance="dense")

    cov_caller = np.asarray(dist_caller.covariance_matrix)
    cov_sorted = np.asarray(dist_sorted.covariance_matrix)
    np.testing.assert_allclose(
        cov_caller,
        cov_sorted[np.ix_(np.argsort(sort_perm), np.argsort(sort_perm))],
        atol=1e-9,
    )


def test_state_space_posterior_predict_smoothed_dense_single_test_point():
    """M=1 is the degenerate case with no cross-covariance segments at all."""
    ss_posterior, dense_posterior, train_data, _ = _build_smoothed_dense_comparison(
        gpx.kernels.Matern12, jitter=1e-6
    )
    Xtest = jnp.array([[4.2]])

    ss_dist = ss_posterior.predict(Xtest, train_data, covariance="dense")
    dense_dist = dense_posterior.predict(Xtest, train_data, covariance="dense")

    assert ss_dist.covariance_matrix.shape == (1, 1)
    np.testing.assert_allclose(
        np.asarray(ss_dist.covariance_matrix),
        np.asarray(dense_dist.covariance_matrix),
        atol=1e-8,
    )


def test_state_space_posterior_predict_smoothed_dense_joint_sampling_smoke():
    """Joint samples must be finite and their empirical covariance must be
    consistent with the analytic dense covariance."""
    ss_posterior, _, train_data, Xtest = _build_smoothed_dense_comparison(
        gpx.kernels.Matern32, jitter=1e-6, n_test=6
    )
    dist = ss_posterior.predict(Xtest, train_data, covariance="dense")

    samples = dist.sample(jr.key(0), sample_shape=(4000,))
    assert samples.shape == (4000, 6)
    assert bool(jnp.all(jnp.isfinite(samples)))

    empirical_cov = np.cov(np.asarray(samples).T)
    analytic_cov = np.asarray(dist.covariance_matrix)
    # Monte-Carlo tolerance for 4000 draws, not the ~1e-8 exact-match bar above.
    np.testing.assert_allclose(empirical_cov, analytic_cov, atol=0.05)


def test_state_space_posterior_predict_smoothed_dense_is_jittable():
    lengthscale, variance, obs_stddev = 1.2, 0.9, 0.15
    X_train, y_train = _build_matern12_dataset(
        n=20, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))
    Xtest = jnp.linspace(0.0, 10.0, 5).reshape(-1, 1)

    def build_and_predict(kernel_lengthscale):
        posterior = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.Matern32(
                lengthscale=kernel_lengthscale, variance=variance
            ),
            jitter=1e-6,
        ) * gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
        dist = posterior.predict(Xtest, train_data, covariance="dense")
        return dist.mean, dist.covariance_matrix

    jitted_mean, jitted_cov = jax.jit(build_and_predict)(jnp.asarray(lengthscale))
    eager_mean, eager_cov = build_and_predict(jnp.asarray(lengthscale))

    np.testing.assert_allclose(
        np.asarray(jitted_mean), np.asarray(eager_mean), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(jitted_cov), np.asarray(eager_cov), atol=1e-10
    )


def test_state_space_posterior_predict_smoothed_dense_is_differentiable():
    lengthscale, variance, obs_stddev = 1.2, 0.9, 0.15
    X_train, y_train = _build_matern12_dataset(
        n=20, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))
    Xtest = jnp.linspace(0.0, 10.0, 5).reshape(-1, 1)

    def loss(kernel_lengthscale):
        posterior = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.Matern32(
                lengthscale=kernel_lengthscale, variance=variance
            ),
            jitter=1e-6,
        ) * gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
        dist = posterior.predict(Xtest, train_data, covariance="dense")
        return jnp.sum(dist.covariance_matrix) + jnp.sum(dist.mean**2)

    gradient = jax.grad(loss)(jnp.asarray(lengthscale))
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


def test_rts_smoother_return_gains_shape_is_linear_in_grid_size():
    """The gains that feed the cross-covariance recursion are one array of
    shape ``(grid_size - 1, state_dim, state_dim)`` — never an
    ``(N, N)``-shaped object — which is what keeps the dense predictive
    linear in the training set size rather than quadratic."""
    from gpjax.state_space.inference import _sqrt_filter_forward
    from gpjax.state_space.sde import Matern32SDE

    lengthscale, variance, obs_stddev = 1.0, 1.0, 0.2
    n = 40
    X, y = _build_matern12_dataset(
        n=n, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    sde = Matern32SDE(lengthscale=lengthscale, variance=variance)
    time_steps = jnp.concatenate([jnp.array([0.0]), jnp.diff(X)])
    is_observed = jnp.ones(n, dtype=bool)
    forward_outputs, _ = _sqrt_filter_forward(
        sde, y, time_steps, is_observed, jnp.asarray(obs_stddev)
    )
    _, _, smoother_gains = rts_smoother(
        sde, forward_outputs, time_steps, return_gains=True
    )
    assert smoother_gains.shape == (n - 1, sde.state_dim, sde.state_dim)


def test_state_space_posterior_predict_smoothed_dense_scales_to_larger_training_set():
    """Soft robustness check: the dense predictive stays finite and correct
    as the training set grows well past the small sizes used elsewhere in
    this file, with a small, fixed number of test points."""
    lengthscale, variance, obs_stddev = 1.0, 1.0, 0.2
    n_train = 300
    X_train, y_train = _build_matern12_dataset(
        n=n_train, lengthscale=lengthscale, variance=variance, obs_stddev=obs_stddev
    )
    train_data = gpx.Dataset(X=X_train.reshape(-1, 1), y=y_train.reshape(-1, 1))
    Xtest = jnp.array([[1.5], [4.5], [8.5]])

    posterior = StateSpacePrior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern32(lengthscale=lengthscale, variance=variance),
        jitter=1e-6,
    ) * gpx.likelihoods.Gaussian(obs_stddev=obs_stddev)
    dist = posterior.predict(Xtest, train_data, covariance="dense")

    cov = np.asarray(dist.covariance_matrix)
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


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
