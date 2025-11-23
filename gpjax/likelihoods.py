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
from __future__ import annotations

import abc
from dataclasses import dataclass

import beartype.typing as tp
from flax import nnx
import jax
from jax import vmap
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float
import numpy as np
import numpyro.distributions as npd

from gpjax.distributions import GaussianDistribution
from gpjax.integrators import (
    AbstractIntegrator,
    AnalyticalGaussianIntegrator,
    GHQuadratureIntegrator,
)
from gpjax.parameters import (
    NonNegativeReal,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)


@dataclass
class NoiseMoments:
    log_variance: Array
    inv_variance: Array
    variance: Array


jax.tree_util.register_pytree_node(
    NoiseMoments,
    lambda x: ((x.log_variance, x.inv_variance, x.variance), None),
    lambda _, x: NoiseMoments(*x),
)


class AbstractLikelihood(nnx.Module):
    r"""Abstract base class for likelihoods.

    All likelihoods must inherit from this class and implement the `predict` and
    `link_function` methods.
    """

    def __init__(
        self,
        num_datapoints: int,
        integrator: AbstractIntegrator = GHQuadratureIntegrator(),
    ):
        """Initializes the likelihood.

        Args:
            num_datapoints (int): the number of data points.
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`.
        """
        self.num_datapoints = num_datapoints
        self.integrator = integrator

    def __call__(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            dist: The predictive distribution to evaluate the likelihood at.

        Returns:
            The predictive distribution.
        """
        return self.predict(dist)

    @abc.abstractmethod
    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Distribution:
        r"""Evaluate the likelihood function at a given predictive distribution.

        Args:
            dist: The predictive distribution to evaluate the likelihood at.

        Returns:
            npd.Distribution: The predictive distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def link_function(self, f: Float[Array, "..."]) -> npd.Distribution:
        r"""Return the link function of the likelihood function.

        Args:
            f (Float[Array, "..."]): the latent Gaussian process values.

        Returns:
            npd.Distribution: The distribution of observations, y, given values of the
                Gaussian process, f.
        """
        raise NotImplementedError

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        **_: tp.Any,
    ) -> Float[Array, " N"]:
        r"""Compute the expected log likelihood.

        For a variational distribution $q(f)\sim\mathcal{N}(m, s)$ and a likelihood
        $p(y|f)$, compute the expected log likelihood:
        ```math
        \mathbb{E}_{q(f)}\left[\log p(y|f)\right]
        ```

        Args:
            y (Float[Array, 'N D']): The observed response variable.
            mean (Float[Array, 'N D']): The variational mean.
            variance (Float[Array, 'N D']): The variational variance.
            mean_g (Float[Array, 'N D']): Optional moments of the latent noise
                process for heteroscedastic likelihoods.
            variance_g (Float[Array, 'N D']): Optional moments of the latent noise
                process for heteroscedastic likelihoods.
            **_: Unused extra arguments for compatibility with specialised
                likelihoods.

        Returns:
            ScalarFloat: The expected log likelihood.
        """
        log_prob = vmap(lambda f, y: self.link_function(f).log_prob(y))
        return self.integrator(
            fun=log_prob, y=y, mean=mean, variance=variance, likelihood=self
        )


class AbstractNoiseTransform(nnx.Module):
    """Abstract base class for noise transformations."""

    @abc.abstractmethod
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform the input noise signal."""
        raise NotImplementedError

    @abc.abstractmethod
    def moments(
        self, mean: Float[Array, "..."], variance: Float[Array, "..."]
    ) -> NoiseMoments:
        """Compute the moments of the transformed noise signal."""
        raise NotImplementedError


class LogNormalTransform(AbstractNoiseTransform):
    """Log-normal noise transformation."""

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.exp(x)

    def moments(
        self, mean: Float[Array, "..."], variance: Float[Array, "..."]
    ) -> NoiseMoments:
        expected_variance = jnp.exp(mean + 0.5 * variance)
        expected_log_variance = mean
        expected_inv_variance = jnp.exp(-mean + 0.5 * variance)
        return NoiseMoments(
            log_variance=expected_log_variance,
            inv_variance=expected_inv_variance,
            variance=expected_variance,
        )


class SoftplusTransform(AbstractNoiseTransform):
    """Softplus noise transformation."""

    def __init__(self, num_points: int = 20):
        self.num_points = num_points

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnn.softplus(x)

    def moments(
        self, mean: Float[Array, "..."], variance: Float[Array, "..."]
    ) -> NoiseMoments:
        quad_x, quad_w = np.polynomial.hermite.hermgauss(self.num_points)
        quad_w = jnp.asarray(quad_w / jnp.sqrt(jnp.pi))
        quad_x = jnp.asarray(quad_x)

        std = jnp.sqrt(variance)
        samples = mean[..., None] + jnp.sqrt(2.0) * std[..., None] * quad_x
        sigma2 = self(samples)
        log_sigma2 = jnp.log(sigma2)
        inv_sigma2 = 1.0 / sigma2

        expected_variance = jnp.sum(sigma2 * quad_w, axis=-1)
        expected_log_variance = jnp.sum(log_sigma2 * quad_w, axis=-1)
        expected_inv_variance = jnp.sum(inv_sigma2 * quad_w, axis=-1)

        return NoiseMoments(
            log_variance=expected_log_variance,
            inv_variance=expected_inv_variance,
            variance=expected_variance,
        )


class AbstractHeteroscedasticLikelihood(AbstractLikelihood):
    r"""Base class for heteroscedastic likelihoods with latent noise processes."""

    def __init__(
        self,
        num_datapoints: int,
        noise_prior,
        noise_transform: tp.Union[
            AbstractNoiseTransform,
            tp.Callable[[Float[Array, "..."]], Float[Array, "..."]],
        ] = SoftplusTransform(),
        integrator: AbstractIntegrator = GHQuadratureIntegrator(),
    ):
        self.noise_prior = noise_prior

        if isinstance(noise_transform, AbstractNoiseTransform):
            self.noise_transform = noise_transform
        else:
            transform_name = getattr(noise_transform, "__name__", "")
            if noise_transform is jnp.exp or transform_name == "exp":
                self.noise_transform = LogNormalTransform()
            else:
                # Default to SoftplusTransform for softplus or unknown callables (legacy behavior used quadrature)
                # Note: If an unknown callable is passed, we technically use SoftplusTransform which applies softplus.
                # Users should implement AbstractNoiseTransform for custom transforms.
                self.noise_transform = SoftplusTransform()

        super().__init__(num_datapoints=num_datapoints, integrator=integrator)

    def __call__(
        self,
        dist: tp.Union[npd.MultivariateNormal, GaussianDistribution],
        noise_dist: tp.Optional[
            tp.Union[npd.MultivariateNormal, GaussianDistribution]
        ] = None,
    ) -> npd.Distribution:
        return self.predict(dist, noise_dist)

    def supports_tight_bound(self) -> bool:
        """Return whether the tighter bound from Lázaro-Gredilla & Titsias (2011)
        is applicable."""
        return False

    def noise_statistics(
        self, mean: Float[Array, "N D"], variance: Float[Array, "N D"]
    ) -> NoiseMoments:
        r"""Moment matching of the transformed noise process.

        Args:
            mean: Mean of the latent noise GP.
            variance: Variance of the latent noise GP.

        Returns:
            NoiseMoments: Expected log variance, inverse variance, and variance.
        """
        return self.noise_transform.moments(mean, variance)

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        **kwargs: tp.Any,
    ) -> Float[Array, " N"]:
        raise NotImplementedError


class Gaussian(AbstractLikelihood):
    r"""Gaussian likelihood object."""

    def __init__(
        self,
        num_datapoints: int,
        obs_stddev: tp.Union[ScalarFloat, Float[Array, "#N"], NonNegativeReal] = 1.0,
        integrator: AbstractIntegrator = AnalyticalGaussianIntegrator(),
    ):
        r"""Initializes the Gaussian likelihood.

        Args:
            num_datapoints (int): the number of data points.
            obs_stddev (Union[ScalarFloat, Float[Array, "#N"]]): the standard deviation
                of the Gaussian observation noise.
            integrator (AbstractIntegrator): The integrator to be used for computing expected log
                likelihoods. Must be an instance of `AbstractIntegrator`. For the Gaussian likelihood, this defaults to
                the `AnalyticalGaussianIntegrator`, as the expected log likelihood can be computed analytically.
        """
        if not isinstance(obs_stddev, NonNegativeReal):
            obs_stddev = NonNegativeReal(jnp.asarray(obs_stddev))
        self.obs_stddev = obs_stddev

        super().__init__(num_datapoints, integrator)

    def link_function(self, f: Float[Array, "..."]) -> npd.Normal:
        r"""The link function of the Gaussian likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Normal: The likelihood function.
        """
        return npd.Normal(loc=f, scale=self.obs_stddev.value.astype(f.dtype))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.MultivariateNormal:
        r"""Evaluate the Gaussian likelihood.

        Evaluate the Gaussian likelihood function at a given predictive
        distribution. Computationally, this is equivalent to summing the
        observation noise term to the diagonal elements of the predictive
        distribution's covariance matrix.

        Args:
            dist (npd.Distribution): The Gaussian process posterior,
                evaluated at a finite set of test points.

        Returns:
            npd.Distribution: The predictive distribution.
        """
        n_data = dist.event_shape[0]
        cov = dist.covariance_matrix
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(self.obs_stddev.value**2)

        return npd.MultivariateNormal(dist.mean, noisy_cov)


class HeteroscedasticGaussian(AbstractHeteroscedasticLikelihood):
    def predict(
        self,
        dist: tp.Union[npd.MultivariateNormal, GaussianDistribution],
        noise_dist: tp.Optional[
            tp.Union[npd.MultivariateNormal, GaussianDistribution]
        ] = None,
    ) -> npd.MultivariateNormal:
        if noise_dist is None:
            raise ValueError(
                "noise_dist must be provided for heteroscedastic prediction."
            )

        n_data = dist.event_shape[0]
        noise_mean = noise_dist.mean
        noise_variance = jnp.diag(noise_dist.covariance_matrix)
        noise_stats = self.noise_statistics(
            noise_mean[..., None], noise_variance[..., None]
        )

        cov = dist.covariance_matrix
        noisy_cov = cov.at[jnp.diag_indices(n_data)].add(noise_stats.variance.squeeze())

        return npd.MultivariateNormal(dist.mean, noisy_cov)

    def link_function(self, f: Float[Array, "..."]) -> npd.Normal:
        sigma2 = self.noise_transform(jnp.zeros_like(f))
        return npd.Normal(loc=f, scale=jnp.sqrt(sigma2))

    def expected_log_likelihood(
        self,
        y: Float[Array, "N D"],
        mean: Float[Array, "N D"],
        variance: Float[Array, "N D"],
        mean_g: tp.Optional[Float[Array, "N D"]] = None,
        variance_g: tp.Optional[Float[Array, "N D"]] = None,
        noise_stats: tp.Optional[NoiseMoments] = None,
        return_parts: bool = False,
        **_: tp.Any,
    ) -> tp.Union[Float[Array, " N"], tuple[Float[Array, " N"], NoiseMoments]]:
        if mean_g is None or variance_g is None:
            raise ValueError(
                "mean_g and variance_g must be provided for heteroscedastic models."
            )

        if noise_stats is None:
            noise_stats = self.noise_statistics(mean_g, variance_g)
        sq_error = jnp.square(y - mean)
        log2pi = jnp.log(2.0 * jnp.pi)
        expected = -0.5 * (
            log2pi
            + noise_stats.log_variance
            + (sq_error + variance) * noise_stats.inv_variance
        )
        expected_sum = jnp.sum(expected, axis=1)
        if return_parts:
            return expected_sum, noise_stats
        return expected_sum

    def supports_tight_bound(self) -> bool:
        return True


class Bernoulli(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> npd.BernoulliProbs:
        r"""The probit link function of the Bernoulli likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Bernoulli: The likelihood function.
        """
        return npd.Bernoulli(probs=inv_probit(f))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.BernoulliProbs:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist ([npd.MultivariateNormal, GaussianDistribution].): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.Bernoulli: The pointwise predictive distribution.
        """
        variance = jnp.diag(dist.covariance_matrix)
        mean = dist.mean.ravel()
        return self.link_function(mean / jnp.sqrt(1.0 + variance))


class Poisson(AbstractLikelihood):
    def link_function(self, f: Float[Array, "..."]) -> npd.Poisson:
        r"""The link function of the Poisson likelihood.

        Args:
            f (Float[Array, "..."]): Function values.

        Returns:
            npd.Poisson: The likelihood function.
        """
        return npd.Poisson(rate=jnp.exp(f))

    def predict(
        self, dist: tp.Union[npd.MultivariateNormal, GaussianDistribution]
    ) -> npd.Poisson:
        r"""Evaluate the pointwise predictive distribution.

        Evaluate the pointwise predictive distribution, given a Gaussian
        process posterior and likelihood parameters.

        Args:
            dist (tp.Union[npd.MultivariateNormal, GaussianDistribution]): The Gaussian
                process posterior, evaluated at a finite set of test points.

        Returns:
            npd.Poisson: The pointwise predictive distribution.
        """
        return self.link_function(dist.mean)


def inv_probit(x: Float[Array, " *N"]) -> Float[Array, " *N"]:
    r"""Compute the inverse probit function.

    Args:
        x (Float[Array, "*N"]): A vector of values.

    Returns
    -------
        Float[Array, "*N"]: The inverse probit of the input vector.
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter


NonGaussian = tp.Union[Poisson, Bernoulli]

__all__ = [
    "AbstractLikelihood",
    "NonGaussian",
    "Gaussian",
    "AbstractHeteroscedasticLikelihood",
    "HeteroscedasticGaussian",
    "Bernoulli",
    "Poisson",
    "inv_probit",
    "NoiseMoments",
    "AbstractNoiseTransform",
    "LogNormalTransform",
    "SoftplusTransform",
]
