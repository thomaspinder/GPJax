import jax.numpy as jnp
from objax import Module
from typing import Callable, Optional
from jax import vmap
from .parameters import Parameter


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
    def dist(x: jnp.array, y: jnp.array) -> float:
        """
        Compute the squared distance matrix between two inputs.

        Args:
            x: A 1xN vector
            y: A 1xM vector

        Returns:
            A float value quantifying the distance between x and y.
        """
        return jnp.sum((x - y)**2)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        raise NotImplementedError


class Stationary(Kernel):
    """
    A base class for stationary kernels. That is, kernels whereby the Gram matrix's values are invariant to the value of the inputs, and instead depend only on the distance between the inputs.
    """
    def __init__(self,
                 lengthscale: Optional[jnp.ndarray] = jnp.array([1.]),
                 variance: Optional[jnp.ndarray] = jnp.array([1.]),
                 name: Optional[str] = "Stationary"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The inital value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RBF(Stationary):
    """
    The radial basis function kernel.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "RBF"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def feature_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the RBF specific function
        """
        # :math:`k(x,y)=\sigma^2 \exp\left( \frac{-0.5 \tau}{2\ell^2}\right) ` where  :math:`\tau = \lVert x-y \rVert_{2}^{2}`.
        ell = self.lengthscale.untransform
        sigma = self.variance.untransform
        tau = self.dist(x / ell, y / ell)
        return sigma * jnp.exp(-0.5 * tau)

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.feature_map, X, Y).squeeze()
