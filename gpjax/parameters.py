from objax import TrainVar
from objax.typing import JaxArray
from objax.variable import reduce_mean
import jax.numpy as jnp
from typing import Optional, Callable
from .transforms import Transform, Softplus
from tensorflow_probability.substrates.jax import distributions as tfd


class Parameter(TrainVar):
    """
    Base parameter class. This is a simple extension of the `TrainVar` class in Objax that enables parameter transforms
    and, in the future, prior distributions to be placed on the parameter in question.
    """
    def __init__(self,
                 tensor: JaxArray,
                 reduce: Optional[Callable[[JaxArray],
                                           JaxArray]] = reduce_mean,
                 transform: Transform = Softplus(),
                 prior: tfd.Distribution = None):
        """
        Args:
            tensor: The initial value of the parameter
            reduce: A helper function for parallelisable calls.
            transform: The bijective transformation that should be applied to the parameter.
        """
        super().__init__(transform.forward(tensor), reduce)
        self.fn = transform
        self.prior = prior

    @property
    def untransform(self) -> JaxArray:
        """
        Return the paramerter's transformed valued that exists on constrained R.
        """
        return self.fn.backward(self.value)

    @property
    def log_density(self) -> JaxArray:
        """
        Return the log prior density of the parameter's constrained parameter value.
        """
        if self.prior is None:
            lpd = jnp.zeros_like(self.value)
        else:
            lpd = self.prior.log_prob(self.untransform)
        return lpd

    @property
    def n_values(self) -> int:
        return self.value.shape[0]
