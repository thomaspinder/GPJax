"""Conformance tests: every variational family conditions through gpjax.conditioning.

The v1.0 contract for Gaussian-output variational families:

- ``condition()`` returns a :class:`gpjax.conditioning.Posterior`;
- ``predict`` is one-line sugar over ``condition`` and agrees with it exactly;
- ``prior_kl()`` returns a finite scalar;
- the posterior's diagonal-covariance path agrees with the dense marginals.

The collapsed family is the one whose maths needs data: ``condition`` and
``prior_kl`` take the training set, mirroring ``model.condition(train_data)``.
"""

import gpjax as gpx
from gpjax.conditioning import (
    CollapsedPosterior,
    Posterior,
    SparsePosterior,
)
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    DualVariationalGaussian,
    GraphVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

from tests._dual_helpers import random_dual_sites

_NUM_INDUCING = 5


def _nontrivial_moments(num_inducing: int):
    index = jnp.arange(num_inducing, dtype=jnp.float64)
    mean = (0.3 * jnp.cos(1.7 * index) - 0.15 * index).reshape(-1, 1)
    row, col = index[:, None], index[None, :]
    off_diagonal = 0.2 * jnp.sin(0.9 * row + 1.3 * col)
    diagonal = 0.5 + 0.4 * jnp.cos(0.6 * index)
    root = jnp.tril(off_diagonal, -1) + jnp.diag(diagonal)
    return mean, root


def _euclidean_model():
    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Constant(jnp.array(0.4)),
        kernel=gpx.kernels.RBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.3)),
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.array(0.37))


def _graph_model():
    graph = nx.barbell_graph(10, 0)
    laplacian = jnp.asarray(nx.laplacian_matrix(graph).toarray(), dtype=jnp.float64)
    kernel = gpx.kernels.GraphKernel(
        laplacian=laplacian, lengthscale=2.3, variance=3.2, smoothness=6.1
    )
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Constant(), kernel=kernel)
    return prior * gpx.likelihoods.Bernoulli()


def _train_data():
    rng = np.random.default_rng(42)
    x = jnp.asarray(np.sort(rng.uniform(-2.0, 2.0, (20, 1)), axis=0))
    y = jnp.sin(3.0 * x) + 0.1 * jnp.asarray(rng.normal(size=(20, 1)))
    return gpx.Dataset(X=x, y=y)


def _build(family_cls):
    """Build a family with non-trivial variational state, plus its test inputs."""
    if family_cls is GraphVariationalGaussian:
        model = _graph_model()
        inducing = jnp.arange(0, 20, 4).reshape(-1, 1).astype(jnp.int64)
        mean, root = _nontrivial_moments(_NUM_INDUCING)
        family = family_cls(
            model=model,
            inducing_inputs=inducing,
            variational_mean=mean,
            variational_root_covariance=root,
        )
        test_inputs = jnp.arange(1, 20, 3).reshape(-1, 1).astype(jnp.int64)
        return family, test_inputs

    model = _euclidean_model()
    inducing = jnp.linspace(-2.0, 2.0, _NUM_INDUCING).reshape(-1, 1)
    test_inputs = jnp.linspace(-2.5, 2.5, 9).reshape(-1, 1)

    if family_cls is DualVariationalGaussian:
        dual_vector, dual_matrix = random_dual_sites(3, _NUM_INDUCING)
        family = family_cls(
            model=model,
            inducing_inputs=inducing,
            dual_vector=dual_vector,
            dual_matrix=dual_matrix,
        )
    elif family_cls is CollapsedVariationalGaussian:
        family = family_cls(model=model, inducing_inputs=inducing)
    else:
        mean, root = _nontrivial_moments(_NUM_INDUCING)
        family = family_cls(
            model=model,
            inducing_inputs=inducing,
            variational_mean=mean,
            variational_root_covariance=root,
        )
    return family, test_inputs


def _condition(family):
    """Condition the family, passing data only where its maths requires it."""
    if isinstance(family, CollapsedVariationalGaussian):
        return family.condition(_train_data())
    return family.condition()


_FAMILIES = [
    VariationalGaussian,
    WhitenedVariationalGaussian,
    GraphVariationalGaussian,
    CollapsedVariationalGaussian,
    DualVariationalGaussian,
]


@pytest.mark.parametrize("family_cls", _FAMILIES, ids=lambda cls: cls.__name__)
def test_condition_returns_a_conditioning_posterior(family_cls):
    family, _ = _build(family_cls)
    posterior = _condition(family)
    assert isinstance(posterior, Posterior)
    expected_mode = (
        CollapsedPosterior
        if family_cls is CollapsedVariationalGaussian
        else SparsePosterior
    )
    assert isinstance(posterior, expected_mode)


@pytest.mark.parametrize("family_cls", _FAMILIES, ids=lambda cls: cls.__name__)
def test_predict_is_sugar_over_condition(family_cls):
    """``q.predict(x)`` must agree with ``q.condition()(x)`` exactly."""
    family, test_inputs = _build(family_cls)
    if family_cls is CollapsedVariationalGaussian:
        data = _train_data()
        via_predict = family.predict(test_inputs, data)
        via_condition = family.condition(data)(test_inputs)
    else:
        via_predict = family.predict(test_inputs)
        via_condition = family.condition()(test_inputs)

    np.testing.assert_array_equal(
        np.asarray(via_predict.mean), np.asarray(via_condition.mean)
    )
    np.testing.assert_array_equal(
        np.asarray(via_predict.covariance()), np.asarray(via_condition.covariance())
    )


@pytest.mark.parametrize("family_cls", _FAMILIES, ids=lambda cls: cls.__name__)
def test_prior_kl_is_a_finite_scalar(family_cls):
    family, _ = _build(family_cls)
    if family_cls is CollapsedVariationalGaussian:
        kl = family.prior_kl(_train_data())
    else:
        kl = family.prior_kl()
    assert kl.shape == ()
    assert jnp.isfinite(kl)
    assert kl >= -1e-12


@pytest.mark.parametrize("family_cls", _FAMILIES, ids=lambda cls: cls.__name__)
def test_diagonal_marginals_match_dense(family_cls):
    """The new diagonal path must agree with the dense covariance's diagonal."""
    family, test_inputs = _build(family_cls)
    posterior = _condition(family)

    dense = posterior(test_inputs, covariance="dense")
    diagonal = posterior(test_inputs, covariance="diagonal")

    np.testing.assert_allclose(
        np.asarray(diagonal.mean), np.asarray(dense.mean), rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(diagonal.variance),
        np.asarray(jnp.diagonal(dense.covariance())),
        rtol=0.0,
        atol=1e-10,
    )
