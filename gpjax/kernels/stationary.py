import jax.numpy as jnp
from objax import Module
from typing import Callable, Optional
from jax import vmap
from ..parameters import Parameter
from .base import Kernel


class RBF(Kernel):
    """
    The radial basis function kernel.
    """
    def __init__(self,
                 lengthscale: Optional[jnp.ndarray] = jnp.array([1.]),
                 variance: Optional[jnp.ndarray] = jnp.array([1.]),
                 name: Optional[str] = "RBF"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The inital value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)
    def scaled_distance(self, x: jnp.ndarray, y: jnp.ndarray, power: int) -> jnp.ndarray:
        ell = self.lengthscale.untransform
        tau = self.dist(x / ell, y / ell, power=power)
        return tau
    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the RBF specific function :math:`k(x,y)=\sigma^2 \exp\left( \frac{-0.5 \tau}{2\ell^2}\right) ` where  :math:`\tau = \lVert x-y \rVert_{2}^{2}`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=2)
        return sigma * jnp.exp(-0.5 * tau)
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.kernel_func, X, Y).squeeze()


class Matern12(Kernel):
    """
    The Matern kernel with a smoothness parameter of 1/2.
    """
    def __init__(self,
                 lengthscale: Optional[jnp.ndarray] = jnp.array([1.]),
                 variance: Optional[jnp.ndarray] = jnp.array([1.]),
                 name: Optional[str] = "Matern 1/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The inital value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)
    def scaled_distance(self, x: jnp.ndarray, y: jnp.ndarray, power: int) -> jnp.ndarray:
        ell = self.lengthscale.untransform
        tau = self.dist(x / ell, y / ell, power=power)
        return tau
    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 \exp\left( \frac{-\tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        return sigma * jnp.exp(-0.5 * tau)
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.kernel_func, X, Y).squeeze()


class Matern32(Kernel):
    """
    The Matern kernel with a smoothness parameter of 3/2.
    """
    def __init__(self,
                 lengthscale: Optional[jnp.ndarray] = jnp.array([1.]),
                 variance: Optional[jnp.ndarray] = jnp.array([1.]),
                 name: Optional[str] = "Matern 3/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The inital value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)
    def scaled_distance(self, x: jnp.ndarray, y: jnp.ndarray, power: int) -> jnp.ndarray:
        ell = self.lengthscale.untransform
        tau = self.dist(x / ell, y / ell, power=power)
        return tau
    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 (1 + \frac{\sqrt{3} \tau}{\ell}) \exp\left( \frac{\sqrt{3} \tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        return sigma*(1+(jnp.sqrt(3.)*tau))*jnp.exp(-jnp.sqrt(3.)*tau)
    def scaled_distance(self, x: jnp.ndarray, y: jnp.ndarray, power: int) -> jnp.ndarray:
        ell = self.lengthscale.untransform
        tau = self.dist(x / ell, y / ell, power=power)
        return tau
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.kernel_func, X, Y).squeeze()


class Matern52(Kernel):
    """
    The Matern kernel with a smoothness parameter of 5/2.
    """
    def __init__(self,
                 lengthscale: Optional[jnp.ndarray] = jnp.array([1.]),
                 variance: Optional[jnp.ndarray] = jnp.array([1.]),
                 name: Optional[str] = "Matern 5/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The inital value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(name=name)
        self.lengthscale = Parameter(lengthscale)
        self.variance = Parameter(variance)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 (1 + \frac{\sqrt{5} \tau}{\ell} + \frac{2.5*\tau**2}{\ell**2}) \exp\left( \frac{\sqrt{5} \tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        sqrt5tau = jnp.sqrt(5.0) * tau
        return sigma*(1.0 + sqrt5tau + jnp.power(tau, 2) / 3.0)*jnp.exp(-sqrt5tau)
    def scaled_distance(self, x: jnp.ndarray, y: jnp.ndarray, power: int) -> jnp.ndarray:
        ell = jnp.power(self.lengthscale.untransform, 2)
        tau = self.dist(x / ell, y / ell, power=power)
        return tau
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return self.gram(self.kernel_func, X, Y).squeeze()
