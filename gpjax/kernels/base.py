import jax.numpy as jnp
from objax import Module
from typing import Callable, Optional
from jax import vmap
from ..parameters import Parameter


class Kernel(Module):
    """
    Base class for all kernel functions. By inheriting the `Module` class from Objax, seamless interaction with model parameters is provided.
    """
    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Optional naming of the kernel.
        """
        self.name = name
        self.spectral = False

    @staticmethod
    def gram(func: Callable, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the kernel's gram matrix given two, possibly identical, Jax arrays.

        Args:
            func: The kernel function to be called for any two values in x and y.
            x: An NxD vector of inputs.
            y: An MXE vector of inputs.

        Returns:
            An NxM gram matrix.
        """
        mapx1 = vmap(lambda x, y: func(x=x, y=y),
                     in_axes=(0, None),
                     out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
        return mapx2(x, y)

    @staticmethod
    def dist(x: jnp.array, y: jnp.array, power: int) -> jnp.ndarray:
        """
        Compute the squared distance matrix between two inputs.

        Args:
            x: A 1xN vector
            y: A 1xM vector

        Returns:
            A float value quantifying the distance between x and y.
        """
        return jnp.sum(jnp.power((x - y), power))

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        raise NotImplementedError

