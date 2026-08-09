"""Tests for ``StateSpaceConjugateModel.condition`` and ``StateSpacePosterior``.

These pin the v1.0 conditioning contract for state-space models: conditioning
must reach the Kalman recursions, never the dense ``ExactPosterior`` that
``ConjugateModel`` would otherwise supply by inheritance. The dense fallback is
silent and O(N^3), so every assertion below that looks like plumbing is really
guarding against that regression.
"""

import gpjax as gpx
from gpjax.conditioning import ExactPosterior, Posterior
from gpjax.distributions import GaussianDistribution
from gpjax.state_space.conditioning import StateSpacePosterior
from gpjax.state_space.gps import StateSpacePrior
from gpjax.state_space.objectives import state_space_mll
from gpjax.state_space.prediction import predict_filtered, predict_smoothed
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from tests.test_state_space.test_filter import _build_matern12_dataset

_LENGTHSCALE = 1.5
_VARIANCE = 0.8
_OBS_STDDEV = 0.2


def _build_model(kernel_class=gpx.kernels.Matern32, mean_function=None, jitter=1e-6):
    """A state-space joint model with the shared hyperparameters."""
    return StateSpacePrior(
        mean_function=mean_function or gpx.mean_functions.Zero(),
        kernel=kernel_class(lengthscale=_LENGTHSCALE, variance=_VARIANCE),
        jitter=jitter,
    ) * gpx.likelihoods.Gaussian(obs_stddev=_OBS_STDDEV)


def _build_train_data(n=25):
    X, y = _build_matern12_dataset(
        n=n,
        lengthscale=_LENGTHSCALE,
        variance=_VARIANCE,
        obs_stddev=_OBS_STDDEV,
    )
    return gpx.Dataset(X=X.reshape(-1, 1), y=y.reshape(-1, 1))


_TEST_INPUTS = jnp.linspace(0.5, 9.5, 11).reshape(-1, 1)


# ---------------------------------------------------------------------------
# The returned object: Kalman-backed, never the dense ExactPosterior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
def test_condition_returns_state_space_posterior_not_exact(kernel_class):
    """``condition`` must not fall through to ``ConjugateModel``'s dense path."""
    model = _build_model(kernel_class)
    train_data = _build_train_data()

    conditioned = model.condition(train_data)

    assert isinstance(conditioned, StateSpacePosterior)
    assert isinstance(conditioned, Posterior)
    assert not isinstance(conditioned, ExactPosterior)


def test_or_sugar_returns_state_space_posterior_not_exact():
    """``model | D`` is ``model.condition(D)``, so it must not build a dense one."""
    model = _build_model()
    train_data = _build_train_data()

    conditioned = model | train_data

    assert isinstance(conditioned, StateSpacePosterior)
    assert not isinstance(conditioned, ExactPosterior)
    assert type(conditioned) is type(model.condition(train_data))


def test_conditioned_posterior_exposes_model_and_train_data():
    model = _build_model()
    train_data = _build_train_data()

    conditioned = model | train_data

    assert conditioned.model is model
    assert conditioned.train_data is train_data
    assert conditioned.prior is model.prior
    assert conditioned.likelihood is model.likelihood
    assert conditioned.observation_mask is None


def test_or_sugar_matches_condition_numerically():
    model = _build_model()
    train_data = _build_train_data()

    from_or = (model | train_data)(_TEST_INPUTS)
    from_condition = model.condition(train_data)(_TEST_INPUTS)

    np.testing.assert_array_equal(
        np.asarray(from_or.mean), np.asarray(from_condition.mean)
    )
    np.testing.assert_array_equal(
        np.asarray(from_or.variance), np.asarray(from_condition.variance)
    )


# ---------------------------------------------------------------------------
# Numerical agreement with the Kalman prediction path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
def test_conditioned_call_matches_predict_smoothed(kernel_class):
    """The whole point of the fix: ``condition(D)(t)`` *is* the Kalman smoother."""
    model = _build_model(kernel_class)
    train_data = _build_train_data()

    conditioned = model.condition(train_data)(_TEST_INPUTS, covariance="diagonal")
    reference = predict_smoothed(model, train_data, _TEST_INPUTS)

    np.testing.assert_allclose(
        np.asarray(conditioned.mean), np.asarray(reference.mean), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(conditioned.variance), np.asarray(reference.variance), atol=1e-12
    )


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
def test_conditioned_call_dense_matches_predict_smoothed(kernel_class):
    """Same plumbing identity as above, for the dense joint covariance."""
    model = _build_model(kernel_class)
    train_data = _build_train_data()

    conditioned = model.condition(train_data)(_TEST_INPUTS, covariance="dense")
    reference = predict_smoothed(model, train_data, _TEST_INPUTS, covariance="dense")

    assert isinstance(conditioned.scale, lx.MatrixLinearOperator)
    np.testing.assert_allclose(
        np.asarray(conditioned.mean), np.asarray(reference.mean), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(conditioned.covariance_matrix),
        np.asarray(reference.covariance_matrix),
        atol=1e-12,
    )


def test_conditioned_call_defaults_to_diagonal():
    """``covariance`` defaults to diagonal; the v1 contract has no dense form."""
    model = _build_model()
    train_data = _build_train_data()
    conditioned = model.condition(train_data)

    default_call = conditioned(_TEST_INPUTS)
    explicit_call = conditioned(_TEST_INPUTS, covariance="diagonal")

    assert isinstance(default_call, GaussianDistribution)
    assert isinstance(default_call.scale, lx.DiagonalLinearOperator)
    np.testing.assert_array_equal(
        np.asarray(default_call.variance), np.asarray(explicit_call.variance)
    )


def test_conditioned_filtered_matches_predict_filtered():
    model = _build_model()
    train_data = _build_train_data()

    conditioned = model.condition(train_data).filtered(
        _TEST_INPUTS, covariance="diagonal"
    )
    reference = predict_filtered(model, train_data, _TEST_INPUTS)

    np.testing.assert_allclose(
        np.asarray(conditioned.mean), np.asarray(reference.mean), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(conditioned.variance), np.asarray(reference.variance), atol=1e-12
    )


def test_conditioned_smoothed_and_filtered_differ():
    """A guard against ``filtered`` silently aliasing the smoothed query."""
    model = _build_model()
    train_data = _build_train_data()
    conditioned = model.condition(train_data)

    smoothed = conditioned(_TEST_INPUTS)
    filtered = conditioned.filtered(_TEST_INPUTS)

    # The smoother conditions on the future too, so it is strictly sharper away
    # from the final timestamp.
    assert not np.allclose(np.asarray(smoothed.variance), np.asarray(filtered.variance))
    assert np.all(np.asarray(smoothed.variance) <= np.asarray(filtered.variance) + 1e-9)


def test_condition_threads_the_observation_mask():
    """Masking through ``condition`` must equal dropping the points entirely."""
    model = _build_model()
    n = 25
    train_data = _build_train_data(n=n)

    observation_mask = jnp.ones(n, dtype=bool)
    for index in (3, 11, 17):
        observation_mask = observation_mask.at[index].set(False)

    masked = model.condition(train_data, observation_mask=observation_mask)(
        _TEST_INPUTS
    )

    keep = np.asarray(observation_mask)
    kept_data = gpx.Dataset(X=train_data.X[keep], y=train_data.y[keep])
    dropped = model.condition(kept_data)(_TEST_INPUTS)

    np.testing.assert_allclose(
        np.asarray(masked.mean), np.asarray(dropped.mean), atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(masked.variance), np.asarray(dropped.variance), atol=1e-9
    )


# ---------------------------------------------------------------------------
# The dense joint predictive (smoothed), and the still-rejected filtered one
# ---------------------------------------------------------------------------


def test_conditioned_call_dense_returns_matrix_operator():
    model = _build_model()
    train_data = _build_train_data()
    conditioned = model.condition(train_data)

    dist = conditioned(_TEST_INPUTS, covariance="dense")

    assert isinstance(dist, GaussianDistribution)
    assert isinstance(dist.scale, lx.MatrixLinearOperator)
    assert dist.covariance_matrix.shape == (
        _TEST_INPUTS.shape[0],
        _TEST_INPUTS.shape[0],
    )
    np.testing.assert_allclose(
        np.asarray(dist.covariance_matrix),
        np.asarray(dist.covariance_matrix).T,
        atol=1e-10,
    )


def test_conditioned_predict_dense_matches_call():
    """``predict`` is sugar for ``__call__``; the dense mode is no exception."""
    model = _build_model()
    train_data = _build_train_data()
    conditioned = model.condition(train_data)

    sugar = conditioned.predict(_TEST_INPUTS, covariance="dense")
    explicit = conditioned(_TEST_INPUTS, covariance="dense")

    np.testing.assert_array_equal(
        np.asarray(sugar.covariance_matrix), np.asarray(explicit.covariance_matrix)
    )


def test_conditioned_filtered_dense_raises_with_actionable_message():
    """Unlike the smoothed predictive, ``filtered`` has no dense joint form:
    each test point conditions on a different information set, so a "joint"
    filtered covariance is not comparable to the dense conjugate predictive
    the smoothed path now matches. See ``StateSpacePosterior``'s docstring."""
    model = _build_model()
    train_data = _build_train_data()
    conditioned = model.condition(train_data)

    with pytest.raises(NotImplementedError) as excinfo:
        conditioned.filtered(_TEST_INPUTS, covariance="dense")

    message = str(excinfo.value)
    assert "predict_filter" in message
    assert "covariance='diagonal'" in message


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
def test_log_marginal_likelihood_matches_state_space_mll(kernel_class):
    """The posterior's evidence is the model's Kalman evidence, not a re-derivation."""
    model = _build_model(kernel_class)
    train_data = _build_train_data()

    from_posterior = (model | train_data).log_marginal_likelihood
    from_objective = state_space_mll(model, train_data)

    np.testing.assert_allclose(
        np.float64(from_posterior), np.float64(from_objective), rtol=1e-12, atol=0.0
    )


def test_log_marginal_likelihood_honours_the_observation_mask():
    model = _build_model()
    n = 25
    train_data = _build_train_data(n=n)
    observation_mask = jnp.ones(n, dtype=bool).at[jnp.array([2, 9, 20])].set(False)

    conditioned = model.condition(train_data, observation_mask=observation_mask)

    np.testing.assert_allclose(
        np.float64(conditioned.log_marginal_likelihood),
        np.float64(
            state_space_mll(model, train_data, observation_mask=observation_mask)
        ),
        rtol=1e-12,
        atol=0.0,
    )


def test_log_marginal_likelihood_matches_dense_conjugate_mll():
    """Kalman evidence must equal the dense conjugate evidence for a Matern12 GP."""
    train_data = _build_train_data()
    model = _build_model(gpx.kernels.Matern12, jitter=0.0)

    dense_model = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.Matern12(lengthscale=_LENGTHSCALE, variance=_VARIANCE),
        jitter=0.0,
    ) * gpx.likelihoods.Gaussian(obs_stddev=_OBS_STDDEV)

    np.testing.assert_allclose(
        np.float64((model | train_data).log_marginal_likelihood),
        np.float64(gpx.objectives.conjugate_mll(dense_model, train_data)),
        atol=1e-6,
        rtol=1e-8,
    )


# ---------------------------------------------------------------------------
# The sugar identity: predict is condition(D)(t), never a second implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_class",
    [gpx.kernels.Matern12, gpx.kernels.Matern32, gpx.kernels.Matern52],
)
def test_model_predict_is_condition_then_call(kernel_class):
    model = _build_model(kernel_class)
    train_data = _build_train_data()

    sugar = model.predict(_TEST_INPUTS, train_data)
    explicit = model.condition(train_data)(_TEST_INPUTS)

    np.testing.assert_array_equal(np.asarray(sugar.mean), np.asarray(explicit.mean))
    np.testing.assert_array_equal(
        np.asarray(sugar.variance), np.asarray(explicit.variance)
    )


def test_model_call_is_condition_then_call():
    model = _build_model()
    train_data = _build_train_data()

    sugar = model(_TEST_INPUTS, train_data)
    explicit = model.condition(train_data)(_TEST_INPUTS)

    np.testing.assert_array_equal(np.asarray(sugar.mean), np.asarray(explicit.mean))
    np.testing.assert_array_equal(
        np.asarray(sugar.variance), np.asarray(explicit.variance)
    )


def test_model_predict_filter_is_condition_then_filtered():
    model = _build_model()
    train_data = _build_train_data()

    sugar = model.predict_filter(_TEST_INPUTS, train_data)
    explicit = model.condition(train_data).filtered(_TEST_INPUTS)

    np.testing.assert_array_equal(np.asarray(sugar.mean), np.asarray(explicit.mean))
    np.testing.assert_array_equal(
        np.asarray(sugar.variance), np.asarray(explicit.variance)
    )


def test_posterior_predict_ignores_train_data_argument():
    """``Posterior.predict`` keeps the pre-v1.0 signature but is already conditioned."""
    model = _build_model()
    train_data = _build_train_data()
    conditioned = model.condition(train_data)

    with_data = conditioned.predict(_TEST_INPUTS, _build_train_data(n=5))
    without_data = conditioned.predict(_TEST_INPUTS)

    np.testing.assert_array_equal(
        np.asarray(with_data.mean), np.asarray(without_data.mean)
    )


# ---------------------------------------------------------------------------
# jit
# ---------------------------------------------------------------------------


def test_condition_and_query_are_jittable():
    model = _build_model()
    train_data = _build_train_data()

    @jax.jit
    def query(model, train_data, test_inputs):
        return model.condition(train_data)(test_inputs).mean

    np.testing.assert_allclose(
        np.asarray(query(model, train_data, _TEST_INPUTS)),
        np.asarray(model.condition(train_data)(_TEST_INPUTS).mean),
        atol=1e-12,
    )


def test_log_marginal_likelihood_is_differentiable_through_condition():
    train_data = _build_train_data()

    def loss(lengthscale):
        model = StateSpacePrior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=gpx.kernels.Matern32(lengthscale=lengthscale, variance=_VARIANCE),
        ) * gpx.likelihoods.Gaussian(obs_stddev=_OBS_STDDEV)
        return -(model | train_data).log_marginal_likelihood

    gradient = jax.grad(loss)(jnp.asarray(_LENGTHSCALE))
    assert jnp.isfinite(gradient)
    assert gradient != 0.0
