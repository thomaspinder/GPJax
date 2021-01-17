from objax import Module
import jax.numpy as jnp
from .parameters import Parameter
from typing import Optional, Tuple
from tensorflow_probability.substrates.jax import distributions as tfd


class Likelihood(Module):
    """
    Base class for all likelihood functions. By inheriting the `Module` class from Objax, seamless interaction with
    model parameters is provided.
    """
    def __init__(self, name: Optional[str] = "Likelihood"):
        """
        Args:
            name: Optional naming of the likelihood.
        """
        self.name = name

    def log_density(self, F, Y) -> jnp.ndarray:
        raise NotImplementedError

    def predictive_moments(self, latent_mean,
                           latent_variance) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the predictive mean and predictive variance using the mean and variance of GP's latent function.

        Args:
            mean: Mean of the latent function
            variance: Variance of the latent function

        Returns:
            Tuple containing the predictive mean and variance of the process.
        """
        raise NotImplementedError


class Gaussian(Likelihood):
    """
    Gaussian likelihood function 
    """
    def __init__(self, noise: jnp.array = jnp.array([1.0]), name="Gaussian"):
        super().__init__(name=name)
        self.noise = Parameter(noise)
        self.random_variable = tfd.Normal

    def log_density(self, F, Y) -> jnp.ndarray:
        raise NotImplementedError

    def predictive_moments(self, latent_mean,
                           latent_variance) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


class Bernoulli(Likelihood):
    """
    Bernoulli likelihood that is used for observations that take values in {0, 1}.
    """
    def __init__(self):
        """
        Initialiser for the Bernoulli likelihood class
        """
        super().__init__(name="Bernoulli")
        self.random_variable = tfd.ProbitBernoulli

    def log_density(self, F, Y) -> jnp.ndarray:
        """
        Compute the log-density of a Bernoulli random variable parameterised by F w.r.t some observations Y

        Args:
            F: Latent process of the GP
            Y: Observations Y

        Returns:
            An array of probability values
        """
        return self.random_variable(F).log_prob(Y)

    def predictive_moments(self, latent_mean,
                           latent_variance) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Using the predictive mean and variance of the latent function :math:`f^{\star}`, we can analytically compute
        the averaged predictive probability (eq. 3.25 [Rasmussen and Williams (2006)]) as we've used the Probit link
        function. For a full derivation, see Appendix 3.9 of Rasmussen and Williams (2006). However, the first moment
        is given by
        .. math::
            \mathbb{E}_q[x] = \mu + \frac{\sigma^2 \mathcal{N}(z)}{\Phi(z)\nu\sqrt{1 + \frac{\sigma^2}{\nu^2}}}

        and the second moment by
        .. math::
            \mathbb{V}[x] = \mathbb{E}_q[x^2] - [\mathbb{E}_q[x]]^2.

        Args:
            mean: Mean of the latent function
            variance: Variance of the latent function

        Returns:
            Tuple containing the predictive mean and variance of the process.
        """
        rv = self.random_variable(latent_mean.ravel() /
                                  jnp.sqrt(1 + latent_variance.ravel()))
        return rv.mean(), rv.variance()
