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

from gpjax.likelihoods import (
    Bernoulli,
    Gaussian,
    Poisson,
    inv_probit,
)
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
)
import numpy as np
import numpyro.distributions as npd
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
_initialise_key = jr.key(123)


def _compute_latent_dist(
    n: int,
) -> tuple[npd.MultivariateNormal, Float[Array, " N"], Float[Array, "N N"]]:
    k1, k2 = jr.split(_initialise_key)
    latent_mean = jr.uniform(k1, shape=(n,))
    latent_sqrt = jr.uniform(k2, shape=(n, n))
    latent_cov = jnp.matmul(latent_sqrt, latent_sqrt.T)
    latent_dist = npd.MultivariateNormal(loc=latent_mean, covariance_matrix=latent_cov)
    return latent_dist, latent_mean, latent_cov


@pytest.mark.parametrize("n", [1, 2, 10, 3])
@pytest.mark.parametrize("obs_stddev", [0.1, 0.5, 1.0, 0.0])
def test_gaussian_likelihood(n: int, obs_stddev: float):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Gaussian(num_datapoints=n, obs_stddev=obs_stddev)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.Normal)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, npd.MultivariateNormal)

    # Check predictive mean and variance.
    assert (pred_dist.mean == latent_mean).all()
    noise_matrix = jnp.eye(likelihood.num_datapoints) * likelihood.obs_stddev[...] ** 2
    assert np.allclose(
        pred_dist.scale_tril, jnp.linalg.cholesky(latent_cov + noise_matrix)
    )


@pytest.mark.parametrize("n", [1, 2, 10])
def test_bernoulli_likelihood(n: int):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Bernoulli(num_datapoints=n)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.BernoulliProbs)

    # Construct latent function distribution.
    latent_dist, latent_mean, latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, npd.BernoulliProbs)

    # Check predictive mean and variance.
    p = inv_probit(latent_mean / jnp.sqrt(1.0 + jnp.diagonal(latent_cov)))
    assert (pred_dist.mean == p).all()
    assert (pred_dist.variance == p * (1.0 - p)).all()


@pytest.mark.parametrize("n", [1, 2, 10])
def test_poisson_likelihood(n: int):
    x = jnp.linspace(-3.0, 3.0).reshape(-1, 1)
    likelihood = Poisson(num_datapoints=n)

    assert isinstance(likelihood.link_function, Callable)
    assert isinstance(likelihood.link_function(x), npd.Poisson)

    # Construct latent function distribution.
    latent_dist, latent_mean, _latent_cov = _compute_latent_dist(n)
    pred_dist = likelihood(latent_dist)
    assert isinstance(pred_dist, npd.Poisson)

    # Check predictive mean and variance.
    rate = jnp.exp(latent_mean)
    assert (pred_dist.mean == rate).all()


class TestMultiOutputGaussian:
    def test_init_scalar_noise(self):
        """Scalar obs_stddev broadcasts to [P] vector."""
        from gpjax.likelihoods import MultiOutputGaussian

        lik = MultiOutputGaussian(num_datapoints=10, num_outputs=3, obs_stddev=0.5)
        assert lik.obs_stddev[...].shape == (3,)
        assert jnp.allclose(lik.obs_stddev[...], jnp.full(3, 0.5))

    def test_init_vector_noise(self):
        """Vector obs_stddev is accepted directly."""
        from gpjax.likelihoods import MultiOutputGaussian

        noise = jnp.array([0.1, 0.2, 0.3])
        lik = MultiOutputGaussian(num_datapoints=10, num_outputs=3, obs_stddev=noise)
        assert jnp.allclose(lik.obs_stddev[...], noise)

    def test_noise_vector_shape(self):
        """noise_vector() returns [NP] in output-major order."""
        from gpjax.likelihoods import MultiOutputGaussian

        lik = MultiOutputGaussian(num_datapoints=10, num_outputs=3, obs_stddev=1.0)
        nv = lik.noise_vector(10)
        assert nv.shape == (30,)

    def test_noise_vector_ordering(self):
        """noise_vector() repeats per-output variance in output-major order."""
        from gpjax.likelihoods import MultiOutputGaussian

        noise = jnp.array([1.0, 2.0])
        lik = MultiOutputGaussian(num_datapoints=3, num_outputs=2, obs_stddev=noise)
        nv = lik.noise_vector(3)
        # Output-major: [σ₁² repeated N, σ₂² repeated N]
        expected = jnp.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
        assert jnp.allclose(nv, expected)

    def test_is_gaussian_subclass(self):
        """MultiOutputGaussian is a Gaussian subclass for posterior dispatch."""
        from gpjax.likelihoods import MultiOutputGaussian

        lik = MultiOutputGaussian(num_datapoints=10, num_outputs=2)
        assert isinstance(lik, Gaussian)
