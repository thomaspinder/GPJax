import jax.numpy as jnp
from objax import Module
from typing import Callable, Optional
from jax import vmap
from ..parameters import Parameter
from .base import Kernel


class WhiteNoise(Kernel):
    """
    The White kernel with a noise level on the diagonal.
    """
    def __init__(self,
                 noise_level: Optional[jnp.ndarray] = jnp.array([0.1]),
                 name: Optional[str] = "WhiteNoise"):
        """
        Args:
            noise_level: The noise level used along the diagonal.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.noise_level = Parameter(noise_level)

    def diag(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.noise_level.untransform * jnp.eye(X.shape[0], dtyle=X.dtype)
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)