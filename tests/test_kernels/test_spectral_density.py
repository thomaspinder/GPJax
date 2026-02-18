"""Tests for the SpectralDensity class and kernel spectral density evaluation."""

from gpjax.kernels.stationary import RBF, Matern12, Matern32, Matern52
from gpjax.kernels.stationary.utils import SpectralDensity
from jax import config
import jax.numpy as jnp
import jax.random as jr
import numpy.testing as npt
import pytest

config.update("jax_enable_x64", True)

UNIT_VARIANCE = jnp.array(1.0)
UNIT_LENGTHSCALE = jnp.array(1.0)
TEST_FREQUENCIES = jnp.array([0.0, 1.0, 5.0])


# ──────────────────────────────────────────────────────────────────────
# SpectralDensity interface
# ──────────────────────────────────────────────────────────────────────


def test_spectral_density_has_sample_method():
    """SpectralDensity must expose sample() for RFF compatibility."""
    spectral_density = RBF(n_dims=1).spectral_density
    assert isinstance(spectral_density, SpectralDensity)
    samples = spectral_density.sample(key=jr.key(0), sample_shape=(10, 1))
    assert samples.shape == (10, 1)


def test_spectral_density_is_callable():
    """SpectralDensity must be callable with (omega, variance, lengthscale)."""
    spectral_density = RBF(n_dims=1).spectral_density
    frequencies = jnp.array([0.5, 1.0, 2.0])
    values = spectral_density(frequencies, UNIT_VARIANCE, UNIT_LENGTHSCALE)
    assert values.shape == (3,)
    assert jnp.all(values > 0)


# ──────────────────────────────────────────────────────────────────────
# Closed-form spectral density formulae
# ──────────────────────────────────────────────────────────────────────


def test_rbf_spectral_density_matches_closed_form():
    """S(w) = variance * sqrt(2*pi) * lengthscale * exp(-0.5 * lengthscale^2 * w^2)."""
    variance = jnp.array(2.0)
    lengthscale = jnp.array(0.5)
    spectral_density = RBF(n_dims=1, variance=2.0, lengthscale=0.5).spectral_density

    result = spectral_density(TEST_FREQUENCIES, variance, lengthscale)
    expected = (
        variance
        * jnp.sqrt(2 * jnp.pi)
        * lengthscale
        * jnp.exp(-0.5 * lengthscale**2 * TEST_FREQUENCIES**2)
    )
    npt.assert_allclose(result, expected, atol=1e-12)


def test_rbf_spectral_density_peaks_at_zero_and_decays():
    """RBF spectral density peaks at omega=0 and decays monotonically."""
    spectral_density = RBF(n_dims=1).spectral_density
    frequencies = jnp.linspace(0, 10, 100)
    values = spectral_density(frequencies, UNIT_VARIANCE, UNIT_LENGTHSCALE)
    assert values[0] == jnp.max(values)
    assert jnp.all(jnp.diff(values) <= 0)


def test_matern12_spectral_density_matches_closed_form():
    """S(w) = variance * 2/ell * 1/(1/ell^2 + w^2)."""
    variance = jnp.array(1.5)
    lengthscale = jnp.array(0.8)
    spectral_density = Matern12(
        n_dims=1, variance=1.5, lengthscale=0.8
    ).spectral_density

    result = spectral_density(TEST_FREQUENCIES, variance, lengthscale)
    expected = (
        variance * (2.0 / lengthscale) / (1.0 / lengthscale**2 + TEST_FREQUENCIES**2)
    )
    npt.assert_allclose(result, expected, atol=1e-12)


def test_matern32_spectral_density_matches_closed_form():
    """S(w) = variance * 4*(sqrt(3)/ell)^3 / (3/ell^2 + w^2)^2."""
    variance = jnp.array(2.0)
    lengthscale = jnp.array(1.5)
    spectral_density = Matern32(
        n_dims=1, variance=2.0, lengthscale=1.5
    ).spectral_density

    result = spectral_density(TEST_FREQUENCIES, variance, lengthscale)
    alpha = jnp.sqrt(3.0) / lengthscale
    expected = (
        variance * 4.0 * alpha**3 / (3.0 / lengthscale**2 + TEST_FREQUENCIES**2) ** 2
    )
    npt.assert_allclose(result, expected, atol=1e-12)


def test_matern52_spectral_density_matches_closed_form():
    """S(w) = variance * (16/3)*(sqrt(5)/ell)^5 / (5/ell^2 + w^2)^3."""
    variance = jnp.array(3.0)
    lengthscale = jnp.array(2.0)
    spectral_density = Matern52(
        n_dims=1, variance=3.0, lengthscale=2.0
    ).spectral_density

    result = spectral_density(TEST_FREQUENCIES, variance, lengthscale)
    alpha = jnp.sqrt(5.0) / lengthscale
    expected = (
        variance
        * (16.0 / 3.0)
        * alpha**5
        / (5.0 / lengthscale**2 + TEST_FREQUENCIES**2) ** 3
    )
    npt.assert_allclose(result, expected, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Positivity across all stationary kernels
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("KernelClass", [RBF, Matern12, Matern32, Matern52])
def test_spectral_density_is_positive_everywhere(KernelClass):
    """All spectral densities must be positive for all omega."""
    spectral_density = KernelClass(n_dims=1).spectral_density
    frequencies = jnp.linspace(0, 20, 200)
    values = spectral_density(frequencies, UNIT_VARIANCE, UNIT_LENGTHSCALE)
    assert jnp.all(values > 0)
