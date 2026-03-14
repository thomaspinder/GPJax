import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import numpyro.distributions.transforms as npt
from paramax import AbstractUnwrappable


class FillTriangularTransform(npt.Transform):
    """Transform: vector of length n(n+1)/2 -> n x n lower triangular matrix."""

    def __call__(self, x):
        L = x.shape[-1]
        n = int((-1 + math.sqrt(1 + 8 * L)) // 2)
        if n * (n + 1) // 2 != L:
            raise ValueError("Last dimension must equal n(n+1)/2 for some integer n.")

        def fill_single(vec):
            out = jnp.zeros((n, n), dtype=vec.dtype)
            row, col = jnp.tril_indices(n)
            return out.at[row, col].set(vec)

        if x.ndim == 1:
            return fill_single(x)
        batch_shape = x.shape[:-1]
        flat_x = x.reshape((-1, L))
        out = jax.vmap(fill_single)(flat_x)
        return out.reshape((*batch_shape, n, n))

    def _inverse(self, y):
        if y.ndim < 2:
            raise ValueError("Input to inverse must be at least two-dimensional.")
        n = y.shape[-1]
        if y.shape[-2] != n:
            raise ValueError(f"Input matrix must be square; got shape {y.shape[-2:]}")
        row, col = jnp.tril_indices(n)

        def inv_single(mat):
            return mat[row, col]

        if y.ndim == 2:
            return inv_single(y)
        batch_shape = y.shape[:-2]
        flat_y = y.reshape((-1, n, n))
        out = jax.vmap(inv_single)(flat_y)
        return out.reshape((*batch_shape, n * (n + 1) // 2))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.zeros(x.shape[:-1])

    @property
    def sign(self):
        return 1.0

    def tree_flatten(self):
        return (), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


_fill_triangular = FillTriangularTransform()


def inv_softplus(x):
    """Inverse of jax.nn.softplus: log(exp(x) - 1)."""
    return jnp.log(jnp.expm1(x))


class PositiveReal(AbstractUnwrappable):
    """Strictly positive parameter.

    Stored unconstrained via inverse-softplus; unwrap() applies softplus.
    Softplus is used rather than exp() because its gradient does not
    saturate for large values.
    """

    _unconstrained: jax.Array
    prior: dist.Distribution | None = eqx.field(static=True, default=None)

    def __init__(self, value, *, prior=None):
        self._unconstrained = inv_softplus(jnp.asarray(value, dtype=jnp.float64))
        self.prior = prior

    def unwrap(self):
        return jax.nn.softplus(self._unconstrained)


class NonNegativeReal(AbstractUnwrappable):
    """Non-negative parameter (semantically allows zero, e.g. jitter, noise floor).

    Uses the same softplus bijection as PositiveReal. The distinction is
    semantic: NonNegativeReal signals that zero is a meaningful boundary.
    """

    _unconstrained: jax.Array
    prior: dist.Distribution | None = eqx.field(static=True, default=None)

    def __init__(self, value, *, prior=None):
        self._unconstrained = inv_softplus(jnp.asarray(value, dtype=jnp.float64))
        self.prior = prior

    def unwrap(self):
        return jax.nn.softplus(self._unconstrained)


class Real(AbstractUnwrappable):
    """Unconstrained parameter. unwrap() returns self.value unchanged."""

    value: jax.Array
    prior: dist.Distribution | None = eqx.field(static=True, default=None)

    def __init__(self, value, *, prior=None):
        self.value = jnp.asarray(value, dtype=jnp.float64)
        self.prior = prior

    def unwrap(self):
        return self.value


class SigmoidBounded(AbstractUnwrappable):
    """Parameter bounded to [low, high] via sigmoid bijection."""

    _unconstrained: jax.Array
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)
    prior: dist.Distribution | None = eqx.field(static=True, default=None)

    def __init__(self, value, *, low=0.0, high=1.0, prior=None):
        value = jnp.asarray(value, dtype=jnp.float64)
        self._unconstrained = jax.scipy.special.logit((value - low) / (high - low))
        self.low = low
        self.high = high
        self.prior = prior

    def unwrap(self):
        return self.low + (self.high - self.low) * jax.nn.sigmoid(self._unconstrained)


class LowerTriangular(AbstractUnwrappable):
    """Lower-triangular matrix parameter, stored as a flat vector."""

    _flat: jax.Array
    prior: dist.Distribution | None = eqx.field(static=True, default=None)

    def __init__(self, value, *, prior=None):
        value = jnp.asarray(value, dtype=jnp.float64)
        self._flat = _fill_triangular._inverse(value)
        self.prior = prior

    def unwrap(self):
        return _fill_triangular(self._flat)


class CoregionalizationMatrix(eqx.Module):
    """Parameterises a PSD output-correlation matrix B = WW^T + diag(kappa)."""

    num_outputs: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    W: Real
    kappa: PositiveReal

    def __init__(self, num_outputs: int, rank: int, key: jax.Array):
        self.num_outputs = num_outputs
        self.rank = rank
        self.W = Real(jr.normal(key, (num_outputs, rank)) * 0.1)
        self.kappa = PositiveReal(jnp.ones(num_outputs))

    @property
    def B(self) -> jnp.ndarray:
        w = self.W.unwrap()
        k = self.kappa.unwrap()
        return w @ w.T + jnp.diag(k)
