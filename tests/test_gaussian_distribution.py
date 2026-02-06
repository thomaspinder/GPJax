# %% [markdown]
# Copyright 2022 The Jax Linear Operator Contributors All Rights Reserved.
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


from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

from gpjax.distributions import GaussianDistribution
from gpjax.linalg import PSD
from gpjax.linalg.operators import (
    Dense,
    Diagonal,
)

_key = jr.key(seed=42)

from numpyro.distributions import MultivariateNormal
from numpyro.distributions.kl import kl_divergence


def approx_equal(res: jnp.ndarray, actual: jnp.ndarray) -> bool:
    """Check if two arrays are approximately equal."""
    return jnp.linalg.norm(res - actual) < 1e-5


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_array_arguments(n: int) -> None:
    key_mean, key_sqrt = jr.split(_key, 2)
    mean = jr.uniform(key_mean, shape=(n,))
    sqrt = jr.uniform(key_sqrt, shape=(n, n))
    covariance = sqrt @ sqrt.T
    # check that cholesky does not error
    _L = jnp.linalg.cholesky(covariance)

    dist = GaussianDistribution(loc=mean, scale=PSD(Dense(covariance)))

    assert approx_equal(dist.mean, mean)
    assert approx_equal(dist.variance, covariance.diagonal())
    assert approx_equal(dist.stddev(), jnp.sqrt(covariance.diagonal()))
    assert approx_equal(dist.covariance(), covariance)

    assert isinstance(dist.scale, Dense)
    assert PSD in dist.scale.annotations

    y = jr.uniform(_key, shape=(n,))

    tfp_dist = MultivariateNormal(loc=mean, covariance_matrix=covariance)

    assert approx_equal(dist.log_prob(y), tfp_dist.log_prob(y))
    assert approx_equal(dist.kl_divergence(dist), 0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_diag_linear_operator(n: int) -> None:
    key_mean, key_diag = jr.split(_key, 2)
    mean = jr.uniform(key_mean, shape=(n,))
    diag = jr.uniform(key_diag, shape=(n,))
    diag_covariance = jnp.diag(diag**2)

    # We purosely forget to add a PSD annotation to the diagonal matrix.
    dist_diag = GaussianDistribution(loc=mean, scale=Diagonal(diag**2))
    npt_dist = MultivariateNormal(loc=mean, covariance_matrix=diag_covariance)

    # We check that the PSD annotation is added automatically.
    assert isinstance(dist_diag.scale, Diagonal)
    assert PSD in dist_diag.scale.annotations

    assert approx_equal(dist_diag.mean, npt_dist.mean)
    assert approx_equal(dist_diag.entropy(), npt_dist.entropy())
    assert approx_equal(dist_diag.variance, npt_dist.variance)
    assert approx_equal(dist_diag.covariance(), npt_dist.covariance_matrix)

    gpjax_samples = dist_diag.sample(key=_key, sample_shape=(10,))
    npt_samples = npt_dist.sample(key=_key, sample_shape=(10,))
    assert approx_equal(gpjax_samples, npt_samples)

    y = jr.uniform(_key, shape=(n,))

    assert approx_equal(dist_diag.log_prob(y), npt_dist.log_prob(y))

    assert approx_equal(dist_diag.kl_divergence(dist_diag), 0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_dense_linear_operator(n: int) -> None:
    key_mean, key_sqrt = jr.split(_key, 2)
    mean = jr.uniform(key_mean, shape=(n,))
    sqrt = jr.uniform(key_sqrt, shape=(n, n))
    covariance = sqrt @ sqrt.T

    sqrt = jnp.linalg.cholesky(covariance + jnp.eye(n) * 1e-10)

    dist_dense = GaussianDistribution(loc=mean, scale=PSD(Dense(covariance)))
    npt_dist = MultivariateNormal(loc=mean, covariance_matrix=covariance)

    assert approx_equal(dist_dense.mean, npt_dist.mean)
    assert approx_equal(dist_dense.entropy(), npt_dist.entropy())
    assert approx_equal(dist_dense.variance, npt_dist.variance)
    assert approx_equal(dist_dense.covariance(), npt_dist.covariance_matrix)

    y = jr.uniform(_key, shape=(n,))

    assert approx_equal(dist_dense.log_prob(y), npt_dist.log_prob(y))
    assert approx_equal(dist_dense.kl_divergence(dist_dense), 0.0)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_kl_divergence(n: int) -> None:
    key_a, key_b = jr.split(_key, 2)
    mean_a = jr.uniform(key_a, shape=(n,))
    mean_b = jr.uniform(key_b, shape=(n,))
    sqrt_a = jr.uniform(key_a, shape=(n, n))
    sqrt_b = jr.uniform(key_b, shape=(n, n))
    covariance_a = sqrt_a @ sqrt_a.T
    covariance_b = sqrt_b @ sqrt_b.T

    dist_a = GaussianDistribution(loc=mean_a, scale=PSD(Dense(covariance_a)))
    dist_b = GaussianDistribution(loc=mean_b, scale=PSD(Dense(covariance_b)))

    npt_dist_a = MultivariateNormal(loc=mean_a, covariance_matrix=covariance_a)
    npt_dist_b = MultivariateNormal(loc=mean_b, covariance_matrix=covariance_b)

    assert approx_equal(
        dist_a.kl_divergence(dist_b), kl_divergence(npt_dist_a, npt_dist_b)
    )
