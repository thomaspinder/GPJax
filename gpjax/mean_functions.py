from objax import Module
import jax.numpy as jnp


class MeanFunction(Module):
    def __init__(self, name: str = "Mean Function"):
        self.name = name

    def __call__(self, X: jnp.ndarray):
        raise NotImplementedError


class ZeroMean(MeanFunction):
    def __init__(self, name: str = "Zero Mean"):
        super().__init__(name=name)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((X.shape[0], 1), dtype=X.dtype)
