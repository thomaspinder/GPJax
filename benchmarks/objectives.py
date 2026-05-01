"""Objective meso-benchmarks.

Three suites — ConjugateMllSuite, SvgpElboSuite, VfeElboSuite — share
``ALIGNED_NS`` on the n-axis and the same underlying data drawn from
``make_shared_dataset``. ASV's dashboard plots them on a common N axis,
which is the headline sparse-vs-full comparison for users.

SvgpElboSuite (uncollapsed VariationalGaussian + stochastic ELBO) varies
both M (inducing count) and batch_size; VfeElboSuite (collapsed analytic
ELBO) varies M only — the collapsed objective requires the full dataset.

VariationalParametrisationSuite compares the four variational Gaussian
parameterisations (standard, whitened, natural, expectation) at one
fixed (n, M) so users can see the per-step cost of the choice.

HeteroscedasticElboSuite and OilmmPredictSuite are independent — they
do not participate in the alignment because their model structure
differs.
"""

from __future__ import annotations

import gpjax as gpx
from gpjax import objectives
from gpjax.dataset import Dataset
from gpjax.likelihoods import HeteroscedasticGaussian
from gpjax.variational_families import (
    CollapsedVariationalGaussian,
    ExpectationVariationalGaussian,
    HeteroscedasticVariationalFamily,
    NaturalVariationalGaussian,
    VariationalGaussian,
    WhitenedVariationalGaussian,
)
import jax.numpy as jnp
import jax.random as jr

from benchmarks._setup import (
    ALIGNED_NS,
    M_INDUCING,
    M_INDUCING_GRID,
    SVGP_BATCH_SIZES,
    make_shared_dataset,
    realise,
)


def _conjugate_posterior(n: int):
    data = make_shared_dataset(n)
    kernel = gpx.kernels.RBF()
    mean = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
    return prior * likelihood, data


def _sparse_setup(n: int, m: int = M_INDUCING):
    """Posterior + dataset + first ``m`` inducing points."""
    posterior, data = _conjugate_posterior(n)
    Z = data.X[:m]
    return posterior, data, Z


class ConjugateMllSuite:
    params = (list(ALIGNED_NS),)
    param_names = ("n",)

    def setup(self, n):
        self.posterior, self.data = _conjugate_posterior(n)
        realise(objectives.conjugate_mll(self.posterior, self.data))

    def time_conjugate_mll(self, n):
        realise(objectives.conjugate_mll(self.posterior, self.data))


class SvgpElboSuite:
    """Stochastic Variational GP ELBO with a non-conjugate-style minibatch.

    The variational family is the uncollapsed ``VariationalGaussian``.
    ``elbo`` rescales the variational expectation by ``num_datapoints /
    data.n`` so passing a batch slice gives the standard SVGP estimator.
    """

    params = (list(ALIGNED_NS), list(M_INDUCING_GRID), list(SVGP_BATCH_SIZES))
    param_names = ("n", "m", "batch_size")

    def setup(self, n, m, batch_size):
        posterior, data, Z = _sparse_setup(n, m=m)
        self.q = VariationalGaussian(posterior=posterior, inducing_inputs=Z)
        self.batch = Dataset(X=data.X[:batch_size], y=data.y[:batch_size])
        realise(objectives.elbo(self.q, self.batch))

    def time_elbo(self, n, m, batch_size):
        realise(objectives.elbo(self.q, self.batch))


class VfeElboSuite:
    """Variational Free Energy (collapsed sparse GP) ELBO.

    Uses ``CollapsedVariationalGaussian``, which marginalises q(u)
    analytically and therefore operates on the full dataset — there is
    no mini-batch axis to vary.
    """

    params = (list(ALIGNED_NS), list(M_INDUCING_GRID))
    param_names = ("n", "m")

    def setup(self, n, m):
        posterior, self.data, Z = _sparse_setup(n, m=m)
        self.q = CollapsedVariationalGaussian(posterior=posterior, inducing_inputs=Z)
        realise(objectives.collapsed_elbo(self.q, self.data))

    def time_collapsed_elbo(self, n, m):
        realise(objectives.collapsed_elbo(self.q, self.data))


_VARIATIONAL_FAMILIES = {
    "standard": VariationalGaussian,
    "whitened": WhitenedVariationalGaussian,
    "natural": NaturalVariationalGaussian,
    "expectation": ExpectationVariationalGaussian,
}


class VariationalParametrisationSuite:
    """Per-step ELBO cost across the four variational Gaussian families.

    All four parameterise the same q(u); the differences are in how the
    KL term and predictive moments are computed. Holding (n, M) fixed
    isolates the parameterisation cost.
    """

    params = (list(_VARIATIONAL_FAMILIES),)
    param_names = ("family",)

    def setup(self, family):
        n = 1000
        posterior, self.data, Z = _sparse_setup(n)
        self.q = _VARIATIONAL_FAMILIES[family](posterior=posterior, inducing_inputs=Z)
        realise(objectives.elbo(self.q, self.data))

    def time_elbo(self, family):
        realise(objectives.elbo(self.q, self.data))


class HeteroscedasticElboSuite:
    params = ([200],)
    param_names = ("n",)

    def setup(self, n):
        data = make_shared_dataset(n)
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
        Z = data.X[:M_INDUCING]
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
