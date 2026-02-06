# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections.abc import Callable

import gpjax as gpx
from gpjax.gps import AbstractPosterior
from gpjax.variational_families import (
    AbstractVariationalFamily,
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    GraphVariationalGaussian,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import networkx as nx
import numpy as np
import numpyro.distributions as npd
from numpyro.distributions import Distribution as NumpyroDistribution
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_abstract_variational_family():
    # Test that the abstract class cannot be instantiated.
    with pytest.raises(TypeError):
        AbstractVariationalFamily()

    # Create a dummy variational family class with abstract methods implemented.
    class DummyPosterior:
        @property
        def __class__(self) -> type:
            return AbstractPosterior

    class DummyVariationalFamily(AbstractVariationalFamily):
        def predict(self, x: Float[Array, "N D"]) -> npd.MultivariateNormal:
            return npd.MultivariateNormal(loc=x, covariance_matrix=jnp.eye(x.shape[1]))

    # Test that the dummy variational family can be instantiated.
    dummy_variational_family = DummyVariationalFamily(posterior=DummyPosterior())
    assert isinstance(dummy_variational_family, AbstractVariationalFamily)


# Functions to test variational family parameter shapes upon initialisation.
def vector_shape(n_inducing: int) -> tuple[int, int]:
    """Shape of a vector with n_inducing rows and 1 column."""
    return (n_inducing, 1)


def matrix_shape(n_inducing: int) -> tuple[int, int]:
    """Shape of a matrix with n_inducing rows and 1 column."""
    return (n_inducing, n_inducing)


# Functions to test variational parameter values upon initialisation.
def vector_val(val: float) -> Callable[[int], Float[Array, "n_inducing 1"]]:
    """Vector of shape (n_inducing, 1) filled with val."""

    def vector_val_fn(n_inducing: int):
        return val * jnp.ones(vector_shape(n_inducing))

    return vector_val_fn


def diag_matrix_val(
    val: float,
) -> Callable[[int], Float[Array, "n_inducing n_inducing"]]:
    """Diagonal matrix of shape (n_inducing, n_inducing) filled with val."""

    def diag_matrix_fn(n_inducing: int) -> Float[Array, "n_inducing n_inducing"]:
        return jnp.eye(n_inducing) * val

    return diag_matrix_fn


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize(
    "variational_family",
    [
        VariationalGaussian,
        WhitenedVariationalGaussian,
        NaturalVariationalGaussian,
        ExpectationVariationalGaussian,
    ],
)
def test_variational_gaussians(
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
) -> None:
    # Initialise variational family:
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )
    likelihood = gpx.likelihoods.Gaussian(123)
    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)

    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)

    posterior = prior * likelihood
    q = variational_family(posterior=posterior, inducing_inputs=inducing_inputs)

    # Test init:
    assert q.num_inducing == n_inducing
    assert isinstance(q, AbstractVariationalFamily)

    if isinstance(q, (VariationalGaussian, WhitenedVariationalGaussian)):
        assert q.variational_mean[...].shape == vector_shape(n_inducing)
        assert q.variational_root_covariance[...].shape == matrix_shape(n_inducing)
        assert (q.variational_mean[...] == vector_val(0.0)(n_inducing)).all()
        assert (
            q.variational_root_covariance[...] == diag_matrix_val(1.0)(n_inducing)
        ).all()

    elif isinstance(q, NaturalVariationalGaussian):
        assert q.natural_vector[...].shape == vector_shape(n_inducing)
        assert q.natural_matrix[...].shape == matrix_shape(n_inducing)
        assert (q.natural_vector[...] == vector_val(0.0)(n_inducing)).all()
        assert (q.natural_matrix[...] == diag_matrix_val(-0.5)(n_inducing)).all()

    elif isinstance(q, ExpectationVariationalGaussian):
        assert q.expectation_vector[...].shape == vector_shape(n_inducing)
        assert q.expectation_matrix[...].shape == matrix_shape(n_inducing)
        assert (q.expectation_vector[...] == vector_val(0.0)(n_inducing)).all()
        assert (q.expectation_matrix[...] == diag_matrix_val(1.0)(n_inducing)).all()

    # Test KL
    kl = q.prior_kl()
    assert isinstance(kl, jnp.ndarray)
    assert kl.shape == ()
    assert kl >= 0.0

    # Test predictions
    predictive_dist = q(test_inputs)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize("n_test", [10, 20])
@pytest.mark.parametrize("n_inducing", [10, 20])
@pytest.mark.parametrize(
    "variational_family",
    [
        GraphVariationalGaussian,
    ],
)
def test_graph_variational_gaussian(
    n_test: int,
    n_inducing: int,
    variational_family: AbstractVariationalFamily,
) -> None:
    G = nx.barbell_graph(100, 0)
    L = nx.laplacian_matrix(G).toarray()

    kernel = gpx.kernels.GraphKernel(
        laplacian=L,
        lengthscale=2.3,
        variance=3.2,
        smoothness=6.1,
    )
    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Bernoulli(num_datapoints=G.number_of_nodes())

    inducing_inputs = jnp.array(
        np.random.randint(low=1, high=100, size=(n_inducing, 1))
    ).astype(jnp.int64)

    test_inputs = jnp.array(np.random.randint(low=0, high=1, size=(n_test, 1))).astype(
        jnp.int64
    )

    posterior = prior * likelihood
    q = variational_family(posterior=posterior, inducing_inputs=inducing_inputs)
    # Test KL
    kl = q.prior_kl()
    assert isinstance(kl, jnp.ndarray)
    assert kl.shape == ()
    assert kl >= 0.0

    # Test predictions
    predictive_dist = q(test_inputs)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)


@pytest.mark.parametrize("n_test", [1, 10])
@pytest.mark.parametrize("n_datapoints", [1, 10])
@pytest.mark.parametrize("n_inducing", [1, 10, 20])
@pytest.mark.parametrize("point_dim", [1, 2])
def test_collapsed_variational_gaussian(
    n_test: int, n_inducing: int, n_datapoints: int, point_dim: int
) -> None:
    x = jnp.linspace(-5.0, 5.0, n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.key(123), shape=x.shape) * 0.1
    x = jnp.hstack([x] * point_dim)
    D = gpx.Dataset(X=x, y=y)

    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(), mean_function=gpx.mean_functions.Constant()
    )

    inducing_inputs = jnp.linspace(-5.0, 5.0, n_inducing).reshape(-1, 1)
    inducing_inputs = jnp.hstack([inducing_inputs] * point_dim)
    test_inputs = jnp.linspace(-5.0, 5.0, n_test).reshape(-1, 1)
    test_inputs = jnp.hstack([test_inputs] * point_dim)

    posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=D.n)

    variational_family = CollapsedVariationalGaussian(
        posterior=posterior,
        inducing_inputs=inducing_inputs,
    )

    # We should raise an error for non-Gaussian likelihoods:
    with pytest.raises(TypeError):
        CollapsedVariationalGaussian(
            posterior=prior * gpx.likelihoods.Bernoulli(num_datapoints=D.n),
            inducing_inputs=inducing_inputs,
        )

    # Test init
    assert variational_family.num_inducing == n_inducing
    assert (variational_family.inducing_inputs[...] == inducing_inputs).all()
    assert variational_family.posterior.likelihood.obs_stddev[...] == 1.0

    # Test predictions
    predictive_dist = variational_family(test_inputs, D)
    assert isinstance(predictive_dist, NumpyroDistribution)

    mu = predictive_dist.mean
    sigma = predictive_dist.covariance()

    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)
    assert mu.shape == (n_test,)
    assert sigma.shape == (n_test, n_test)
