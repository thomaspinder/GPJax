from typing import Optional, Union

import jax.numpy as jnp
from chex import dataclass
from jax.interpreters.ad import JVPTracer
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.interpreters.pxla import ShardedDeviceArray

NoneType = type(None)
Array = (
    jnp.ndarray,
    ShardedDeviceArray,
    jnp.DeviceArray,
    JVPTracer,
    DynamicJaxprTracer,
)
Arrays = Union[
    jnp.ndarray, ShardedDeviceArray, jnp.DeviceArray, JVPTracer, DynamicJaxprTracer
]


@dataclass(repr=False)
class Dataset:
    X: Arrays
    y: Arrays

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}"
        )

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        return self.y.shape[1]


def verify_dataset(ds: Dataset) -> None:
    assert (
        ds.X.ndim == 2
    ), f"2-dimensional training inputs are required. Current dimension: {ds.X.ndim}."
    if ds.y is not None:
        assert (
            ds.y.ndim == 2
        ), f"2-dimensional training outputs are required. Current dimension: {ds.y.ndim}."
        assert (
            ds.X.shape[0] == ds.y.shape[0]
        ), f"Number of inputs must equal the number of outputs. \nCurrent counts:\n- X: {ds.X.shape[0]}\n- y: {ds.y.shape[0]}"


@dataclass(repr=False)
class SparseDataset(Dataset):
    """
    Base class for the inducing inputs used for sparse GP schemes
    """
    Z: Arrays = None

    @property
    def n_inducing(self) -> int:
        return self.Z.shape[0]

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}\n- Number of inducing points: {self.n_inducing}"
        )