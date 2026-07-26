# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
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
"""Tests for the spectral measures of stationary kernels (issue #612)."""

from itertools import pairwise

from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
)
from gpjax.kernels.stationary.base import StationaryKernel
import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
import pytest

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

SPECTRAL_KERNELS = [RBF, Matern12, Matern32, Matern52]


@pytest.mark.parametrize("kernel", SPECTRAL_KERNELS)
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("variance", [1.0, 3.0])
def test_spectral_measure_satisfies_bochner(
    kernel: type[StationaryKernel],
    n_dims: int,
    lengthscale: float,
    variance: float,
):
    """Bochner's theorem: k(τ) = σ² · E_{p(ω)}[cos(ωᵀτ)].

    This pins the spectral measure's parameterisation without reference to any
    Fourier-transform normalisation convention. A measure that ignores the
    lengthscale (issue #612) fails this for every ℓ != 1.
    """
    base_kernel = kernel(n_dims=n_dims, lengthscale=lengthscale, variance=variance)
    measure = base_kernel.spectral_density

    omega = measure.sample(jr.key(42), (200_000,))
    assert omega.shape == (200_000, n_dims)

    # A handful of separations at which to check the identity.
    tau = jnp.array([[0.0], [0.3], [0.7], [1.5]]) * jnp.ones((1, n_dims))

    monte_carlo = variance * jnp.mean(jnp.cos(omega @ tau.T), axis=0)
    exact = jax.vmap(lambda t: base_kernel(jnp.zeros(n_dims), t))(tau)

    assert jnp.allclose(monte_carlo, exact, atol=2e-2)


@pytest.mark.parametrize("kernel", SPECTRAL_KERNELS)
def test_spectral_density_depends_on_lengthscale(kernel: type[StationaryKernel]):
    """The reporter's reproducer: densities for different ℓ must not coincide.

    Previously every lengthscale returned the same standardised measure, so the
    curves were superimposed.
    """
    omega = jnp.linspace(-5.0, 5.0, 101).reshape(-1, 1)

    log_probs = [
        kernel(n_dims=1, lengthscale=ell).spectral_density.log_prob(omega)
        for ell in (0.5, 1.0, 2.0)
    ]

    for a, b in pairwise(log_probs):
        assert not jnp.allclose(a, b)


@pytest.mark.parametrize("kernel", SPECTRAL_KERNELS)
def test_spectral_measure_is_multivariate(kernel: type[StationaryKernel]):
    """The measure must live on R^D, not be hard-coded to 1-D."""
    n_dims = 3
    measure = kernel(n_dims=n_dims).spectral_density

    assert measure.event_shape == (n_dims,)

    omega = jnp.ones((7, n_dims))
    assert measure.log_prob(omega).shape == (7,)


@pytest.mark.parametrize("kernel", SPECTRAL_KERNELS)
def test_spectral_measure_scales_inversely_with_lengthscale(
    kernel: type[StationaryKernel],
):
    """A longer lengthscale must concentrate the measure near the origin."""
    short = kernel(n_dims=1, lengthscale=0.5).spectral_density
    long = kernel(n_dims=1, lengthscale=4.0).spectral_density

    key = jr.key(0)
    spread_short = jnp.mean(jnp.abs(short.sample(key, (20_000,))))
    spread_long = jnp.mean(jnp.abs(long.sample(key, (20_000,))))

    assert spread_short > spread_long


@pytest.mark.parametrize("kernel", SPECTRAL_KERNELS)
def test_spectral_measure_is_anisotropic_under_ard(kernel: type[StationaryKernel]):
    """An ARD lengthscale must give a per-dimension spectral scale."""
    base_kernel = kernel(lengthscale=jnp.array([0.5, 4.0]))
    measure = base_kernel.spectral_density

    omega = measure.sample(jr.key(1), (20_000,))
    spread = jnp.mean(jnp.abs(omega), axis=0)

    # Dimension 0 has the shorter lengthscale, so the wider spectrum.
    assert spread[0] > spread[1]


@pytest.mark.parametrize("kernel", SPECTRAL_KERNELS)
def test_spectral_measure_is_independent_of_variance(
    kernel: type[StationaryKernel],
):
    """σ² is the measure's total mass, not part of its shape.

    `spectral_density` returns a normalised probability measure, so the variance
    must not appear in it; Bochner's theorem carries σ² as a separate factor.
    """
    unit = kernel(n_dims=1, variance=1.0).spectral_density
    scaled = kernel(n_dims=1, variance=5.0).spectral_density

    omega = jnp.linspace(-3.0, 3.0, 51).reshape(-1, 1)
    assert jnp.allclose(unit.log_prob(omega), scaled.log_prob(omega))
