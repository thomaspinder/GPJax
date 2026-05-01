"""Objective meso-benchmarks: conjugate_mll, elbo, collapsed_elbo, heteroscedastic_elbo, oilmm_predict."""

from __future__ import annotations

import gpjax as gpx
from gpjax import objectives
from gpjax.likelihoods import HeteroscedasticGaussian
from gpjax.variational_families import HeteroscedasticVariationalFamily
import jax.numpy as jnp
import jax.random as jr

from benchmarks._setup import KEY, make_inputs, make_outputs, realise


def _gaussian_dataset(n: int, key=KEY) -> gpx.Dataset:
    X = make_inputs(n)
    y = make_outputs(n, key=jr.fold_in(key, 1))
    return gpx.Dataset(X=X, y=y)


def _conjugate_posterior(n: int):
    data = _gaussian_dataset(n)
    kernel = gpx.kernels.RBF()
    mean = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
    return prior * likelihood, data


class ConjugateMllSuite:
    params = ([200, 1000],)
    param_names = ("n",)

    def setup(self, n):
        self.posterior, self.data = _conjugate_posterior(n)
        realise(objectives.conjugate_mll(self.posterior, self.data))

    def time_conjugate_mll(self, n):
        realise(objectives.conjugate_mll(self.posterior, self.data))


class ElboSuite:
    """Sparse VFE ELBO."""

    params = ([500],)
    param_names = ("n",)

    def setup(self, n):
        data = _gaussian_dataset(n)
        kernel = gpx.kernels.RBF()
        prior = gpx.gps.Prior(kernel=kernel, mean_function=gpx.mean_functions.Zero())
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
        posterior = prior * likelihood
        Z = data.X[:50]
        self.q = gpx.variational_families.VariationalGaussian(
            posterior=posterior, inducing_inputs=Z
        )
        self.data = data
        realise(objectives.elbo(self.q, self.data))

    def time_elbo(self, n):
        realise(objectives.elbo(self.q, self.data))


class CollapsedElboSuite:
    params = ([500],)
    param_names = ("n",)

    def setup(self, n):
        data = _gaussian_dataset(n)
        kernel = gpx.kernels.RBF()
        prior = gpx.gps.Prior(kernel=kernel, mean_function=gpx.mean_functions.Zero())
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
        posterior = prior * likelihood
        Z = data.X[:50]
        self.q = gpx.variational_families.CollapsedVariationalGaussian(
            posterior=posterior, inducing_inputs=Z
        )
        self.data = data
        realise(objectives.collapsed_elbo(self.q, self.data))

    def time_collapsed_elbo(self, n):
        realise(objectives.collapsed_elbo(self.q, self.data))


class HeteroscedasticElboSuite:
    params = ([200],)
    param_names = ("n",)

    def setup(self, n):
        data = _gaussian_dataset(n)
        signal_kernel = gpx.kernels.RBF()
        noise_kernel = gpx.kernels.RBF()
        signal_prior = gpx.gps.Prior(
            kernel=signal_kernel, mean_function=gpx.mean_functions.Zero()
        )
        noise_prior = gpx.gps.Prior(
            kernel=noise_kernel, mean_function=gpx.mean_functions.Zero()
        )
        likelihood = HeteroscedasticGaussian(num_datapoints=n, noise_prior=noise_prior)
        posterior = signal_prior * likelihood
        Z = data.X[:50]
        self.q = HeteroscedasticVariationalFamily(
            posterior=posterior, inducing_inputs=Z, inducing_inputs_g=Z
        )
        self.data = data
        realise(objectives.heteroscedastic_elbo(self.q, self.data))

    def time_heteroscedastic_elbo(self, n):
        realise(objectives.heteroscedastic_elbo(self.q, self.data))


class OilmmPredictSuite:
    """Replaces the wall-clock assertion in tests/test_models_oilmm_performance.py."""

    params = ([1, 2, 3],)
    param_names = ("m",)

    def setup(self, m):
        N, P = 50, 6
        key = jr.key(42)
        model = gpx.models.create_oilmm(num_outputs=P, num_latent_gps=m, key=key)
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)
        self.posterior = model.condition_on_observations(dataset)
        self.X_test = jnp.linspace(0.1, 0.9, 20).reshape(-1, 1)
        realise(self.posterior.predict(self.X_test))

    def time_predict(self, m):
        realise(self.posterior.predict(self.X_test))
