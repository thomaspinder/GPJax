from __future__ import annotations
import typing

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular
from objax import Module
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Tuple

from ..likelihoods import Gaussian, Likelihood
from ..parameters import Parameter
from ..transforms import Identity


if typing.TYPE_CHECKING:
    from .priors import Prior


class Posterior(Module):
    """
    A base class for GP posterior
    """
    def __init__(self, prior: Prior, likelihood: Likelihood):
        self.kernel = prior.kernel
        self.meanf = prior.meanf
        self.likelihood = likelihood
        self.jitter = prior.jitter

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Here we compute :math:`\log p(y | x, \theta)
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A multivariate normal distribution

        """
        raise NotImplementedError

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        In gradient based optimisation of the marginal log-likelihood (MLL), we minimise the negative MLL.
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A unary float array
        """
        raise NotImplementedError

    def predict(self, Xstar: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Given a set of test points Xstar, comute the predictive mean and variance of the GP here, conditional upon
        the training data.

        Args:
            Xstar: The test point inputs
            X: The training inputs
            y: The training ouputs

        Returns:
            A tuple containing the predictive mean and predictive variance.
        """
        raise NotImplementedError

    def __eq__(self, other: Posterior) -> bool:
        """
        Compare if two Gaussian process posteriors are equal.

        Args:
            other: A second Gaussian process posterior

        Returns: Boolean value

        """
        return list(self.vars().items()) == list(other.vars().items())

    @property
    def n_vars(self) -> int:
        """
        Calculate the total number of parameters within a model.

        Returns:
            Integer value of the number of total number of parameters.
        """
        return sum([v.n_values for v in self.vars().values()])


class PosteriorExact(Posterior):
    """
    A Gaussian process posterior whereby the likelihood function is a Gaussian distribution.
    """
    def __init__(self, prior: Prior, likelihood: Gaussian):
        super().__init__(prior, likelihood)

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Here we compute :math:`\log p(y | x, \theta) for a conjugate, or exact, Gaussian process.
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A multivariate normal distribution

        """
        Inn = jnp.eye(X.shape[0])
        mu = self.meanf(X)
        cov = self.kernel(X, X) + self.jitter * Inn
        cov += self.likelihood.noise.untransform * Inn
        L = jnp.linalg.cholesky(cov)
        return tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        In gradient based optimisation of the marginal log-likelihood (MLL), we minimise the negative MLL.
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A unary float array

        """
        rv = self.marginal_ll(X, y)
        log_prior_density = jnp.sum(jnp.array([v.log_density for k, v in self.vars().items()]))
        return -(rv.log_prob(y.squeeze()).mean()+log_prior_density)

    def predict(self, Xstar: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Conditional upon the GP posterior, compute the predictive posterior given a set of new and unseen test points.
        Args:
            Xstar: The new inputs that correspond to the locations that we'd like to sample the GP predictive posterior at
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A predictive mean and predictive variance

        """
        # Compute covariance matrices and jitter matrix
        Kff = self.kernel(X, X)
        Kfx = self.kernel(Xstar, X)
        Kxx = self.kernel(Xstar, Xstar)
        Inn = jnp.eye(X.shape[0])
        # Compute prior mean
        mu_f = self.meanf(X)
        # Realise the current estimate of the observational noise
        sigma = self.likelihood.noise.untransform
        # Compute the lower Cholesky decomposition
        L = cho_factor(Kff + Inn * sigma, lower=True)
        err = y.reshape(-1, 1) - mu_f.reshape(-1, 1)
        weights = cho_solve(L, err)
        # Compute the predictive mean
        mu = jnp.dot(Kfx, weights)
        # Compute the predictive variance
        v = cho_solve(L, Kfx.T)
        cov = Kxx - jnp.dot(Kfx, v)
        return mu, cov


class PosteriorApprox(Posterior):
    """
    A Gaussian process posterior for cases where the likelihood function of the data is non-Gaussian.
    """
    def __init__(self, prior: Prior, likelihood: Likelihood):
        super().__init__(prior=prior, likelihood=likelihood)
        self.n = None
        self.nu = None
        self.latent_init = False

    def _initialise_latent_values(self, n: int):
        """
        To facilitate the multiplie-dispatch style posterior computation, the latent values of the GP may only be
        initialised once we've observed some data; a step which usually does not happen until optimisation commences.
        Args:
            n: The number of training points
        """
        self.n = n
        self.nu = Parameter(jnp.zeros(shape=(self.n, 1)), transform=Identity(), prior=tfd.Normal(loc=0., scale=1.))
        self.latent_init = True

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the marginal log-likelihood of the non-conjugate Gaussian process

        Args:
            X: Input training points
            y: Output training points

        Returns: A 1-dimensional array containing the marginal log-likelihood's value
        """
        # There is a slight issue here in that the full dataset must be realised before optimisation, otherwise the
        # Objax optimiser does not correctly initialise nu due to the stochasticity of the optimisation.
        if not self.latent_init:
            self._initialise_latent_values(X.shape[0])
        mu = self.meanf(X)
        Kxx = self.kernel(X, X)
        Inn = jnp.eye(self.n) * self.jitter
        L = jnp.linalg.cholesky(Kxx + Inn)
        F = jnp.matmul(L, self.nu.untransform) + mu
        ll = jnp.sum(self.likelihood.log_density(F, y))
        lpd = 0
        for k, v in self.vars().items():
            lpd += jnp.sum(v.log_density).reshape()
        # lpd = jnp.sum(jnp.array([v.log_density for k, v in self.vars().items()]))
        return ll + lpd

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Return the negative marginal log-likelihood. This is only used for optimisation purposes where we minimise
        the negative mll in order to actually maximise its value.

        Args:
            X: Input training points
            y: Output training points

        Returns: A 1-dimensional array containing the negative marginal log-likelihood's value
        """
        return -self.marginal_ll(X, y)

    def predict(self, Xstar: jnp.ndarray, X:jnp.ndarray, y:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Conditional upon the GP posterior, compute the predictive posterior given a set of new and unseen test points.
        Args:
            Xstar: The new inputs that correspond to the locations that we'd like to sample the GP predictive posterior at
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A predictive mean and predictive variance

        """
        if not self.latent_init:
            self._initialise_latent_values(X.shape[0])
        Inn = jnp.eye(X.shape[0])*self.jitter
        Kff = self.kernel(X, X) + Inn
        Kfx = self.kernel(X, Xstar)
        Kxx = jnp.diag(self.kernel(Xstar, Xstar))
        L = jnp.linalg.cholesky(Kff)
        A = solve_triangular(L, Kfx, lower=True)
        latent_var = Kxx - jnp.sum(jnp.square(A), -2)
        latent_mean = jnp.matmul(A.T, self.nu.untransform)
        mu, sigma = self.likelihood.predictive_moments(latent_mean, latent_var)
        return mu, sigma
