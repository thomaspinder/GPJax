import jax.random as jr
from chex import dataclass
from .types import Arrays
from typing import Callable


def random_sample(key: Arrays, X: Arrays, n: int):
    row_idxs = jr.randint(key=key, shape=(n, ), minval=0, maxval=X.shape[0])
    return X[row_idxs, :]


