from typing import Optional

import jax.numpy as jnp
import jax.random as jr
from objax import Module
from tensorflow_probability.substrates.jax import distributions as tfd

from ..kernel import Kernel
from ..likelihoods import Likelihood, Bernoulli, Gaussian
from .posteriors import PosteriorExact, PosteriorApprox
from multipledispatch import dispatch
from ..mean_functions import MeanFunction, ZeroMean


class Prior(Module):
    r"""
    The base class for Gaussian process priors. Considering a set :math:`X` and  function :math:`f`, the GP induces the
    prior :math:`p(f)\sim\mathcal{GP}(m, k)` where :math:`m: X \rightarrow \mathbb{R}` is a mean function and kernel
    :math:`k: X \times X \rightarrow \mathbb{R}`.
    """
    def __init__(self,
                 kernel: Kernel,
                 mean_function: Optional[MeanFunction] = ZeroMean(),
                 jitter: Optional[float] = 1e-6):
        """
        Args:
            kernel: The Gaussian process model's kernel, or covariance, function.
            mean_function: The prior mean function. This is optional and will default to a zero-mean function.
            jitter: A small amount of noise to stabilise the prior covariance matrix.
        """
        self.meanf = mean_function
        self.kernel = kernel
        self.jitter = jitter

    def sample(self,
               X: jnp.ndarray,
               key,
               n_samples: Optional[int] = 1) -> jnp.ndarray:
        """
        Draw a set of n samples from the GP prior at a set of input points.

        Args:
            X: The finite set of points from where we wish to sample from the GP prior.
            key: The Jax key to ensure reproducibility
            n_samples: The number of samples to be drawn.

        Returns: A Jax array of samples.

        """
        mu = self.meanf(X)
        cov = self.kernel(X, X)
        if cov.shape[0] == cov.shape[1]:
            Inn = jnp.eye(cov.shape[0]) * self.jitter
            cov += Inn
        return jr.multivariate_normal(key,
                                      mu.squeeze(),
                                      cov,
                                      shape=(n_samples, ))

    def __mul__(self, other: Likelihood):
        """
        The posterior distribution is proportional to the product of the prior and the data's likelihood. This magic
        method enables this mathematical behaviour to be represented computationally.

        Args:
            other: A likelihood distribution.

        Returns: A Gaussian process posterior.
        """
        return create_posterior(self, other)

    def __repr__(self):
        return "Gaussian process prior with {} kernel and {} mean function.".format(self.kernel.name, self.meanf.name)

    def __str__(self):
        return "Gaussian process prior with {} kernel and {} mean function.".format(self.kernel.name, self.meanf.name)


@dispatch(Prior, Gaussian)
def create_posterior(prior: Prior, likelihood: Gaussian):
    return PosteriorExact(prior, likelihood)


@dispatch(Prior, Bernoulli)
def create_posterior(prior: Prior, likelihood: Bernoulli):
    return PosteriorApprox(prior, likelihood)
