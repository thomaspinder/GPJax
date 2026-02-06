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


from gpjax.integrators import GHQuadratureIntegrator
from gpjax.likelihoods import (
    Bernoulli,
    Gaussian,
)
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("num_points", [10, 20, 30])
def test_quadrature(jit: bool, num_points: int):
    def test():
        def fun(x, y):
            """In practice, the first argument will be the latent function values"""
            return x**2 + y

        mean = jnp.array([[2.0]])
        variance = jnp.array([[1.0]])
        fn_val = GHQuadratureIntegrator(num_points=num_points).integrate(
            fun=fun,
            mean=mean,
            variance=variance,
            y=jnp.ones_like(mean),
            likelihood=None,
        )
        return fn_val.squeeze().round(1)

    if jit:
        test = jax.jit(test)
    assert test() == 6.0


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize(
    "params", [(0.5, -4.22579135), (1.0, -1.91893853), (0.01, -9996.31376835)]
)
def test_analytical_gaussian(jit: bool, params: tuple[float, float]):
    obs_stddev, expected = params
    likelihood = Gaussian(num_datapoints=1, obs_stddev=jnp.array([obs_stddev]))
    mu = jnp.array([[0.0]])
    variance = jnp.array([[1.0]])
    y = jnp.array([[1.0]])

    if jit:
        ell_fn = jax.jit(likelihood.expected_log_likelihood)
    else:
        ell_fn = likelihood.expected_log_likelihood
    ell = ell_fn(y=y, mean=mu, variance=variance)
    np.testing.assert_almost_equal(ell, expected)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("params", [(0.25, 0.5, -0.65437282), (0.5, 1.0, -0.61716802)])
def test_bernoulli_quadrature(jit: bool, params: tuple[float, float]):
    mu, variance, expected = params
    mu = jnp.atleast_2d(mu)
    variance = jnp.atleast_2d(variance)
    likelihood = Bernoulli(num_datapoints=1)
    y = jnp.array([[1.0]])

    if jit:
        ell_fn = jax.jit(likelihood.expected_log_likelihood)
    else:
        ell_fn = likelihood.expected_log_likelihood
    ell = ell_fn(y=y, mean=mu, variance=variance)
    np.testing.assert_almost_equal(ell, expected)
