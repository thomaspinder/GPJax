import math

import equinox as eqx
import jax
from jax.nn import softplus
import jax.numpy as jnp
import jax.random as jr
from numpyro.distributions import biject_to, constraints
from numpyro.distributions.transforms import SoftplusLowerCholeskyTransform
from paramax import AbstractUnwrappable

# numpyro's biject_to is a ConstraintRegistry that maps constraints to
# bijective transforms for unconstrained optimisation:
#
#   biject_to(constraints.softplus_positive) -> SoftplusTransform
#   biject_to(constraints.interval(a, b))    -> SigmoidTransform scaled to [a, b]
#   biject_to(constraints.softplus_lower_cholesky)
#       -> SoftplusLowerCholeskyTransform (fill triangular + softplus on diagonal)
#
# Each parameter class stores a constraint and resolves the bijection via
# biject_to(self._constraint).


class _DtypePreservingSoftplusLowerCholeskyTransform(SoftplusLowerCholeskyTransform):
    """Dtype-preserving variant of numpyro's SoftplusLowerCholeskyTransform.

    TEMPORARY WORKAROUND. numpyro's upstream ``__call__`` routes through
    ``vec_to_tril_matrix`` (``numpyro/distributions/util.py``) and
    ``jnp.identity(n)``, both of which allocate arrays without a dtype
    kwarg. That pulls JAX's default float (``float64`` when
    ``jax_enable_x64`` is on) and forces the whole transform to
    ``float64`` regardless of input dtype -- breaking ``float32`` round
    trips via ``lax.scatter_add`` type-mismatch errors.

    This subclass overrides only the forward pass so allocations inherit
    ``x.dtype``. The inverse is already dtype-preserving upstream.

    Remove this class once the upstream allocations accept ``dtype``.
    See https://github.com/thomaspinder/GPJax/discussions/628.
    """

    def __call__(self, x):
        n = round((math.sqrt(1 + 8 * x.shape[-1]) - 1) / 2)
        off_diag = x[..., :-n]
        diag = softplus(x[..., -n:])

        z = jnp.zeros((*off_diag.shape[:-1], n, n), dtype=x.dtype)
        row, col = jnp.tril_indices(n, k=-1)
        z = z.at[..., row, col].set(off_diag)

        return z + jnp.expand_dims(diag, axis=-1) * jnp.eye(n, dtype=x.dtype)


_dtype_preserving_lower_cholesky = _DtypePreservingSoftplusLowerCholeskyTransform()


class PositiveReal(AbstractUnwrappable):
    """Strictly positive parameter.

    Stored unconstrained via the inverse of the softplus transform;
    ``unwrap()`` applies softplus to recover the constrained value.
    """

    _constraint = constraints.softplus_positive
    _unconstrained: jax.Array

    def __init__(self, value):
        transform = biject_to(self._constraint)
        self._unconstrained = transform.inv(jnp.asarray(value))

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)


class NonNegativeReal(AbstractUnwrappable):
    """Non-negative parameter (semantically allows zero, e.g. jitter, noise floor).

    Uses the same softplus bijection as PositiveReal. The distinction is
    semantic: NonNegativeReal signals that zero is a meaningful boundary.
    """

    _constraint = constraints.softplus_positive
    _unconstrained: jax.Array

    def __init__(self, value):
        transform = biject_to(self._constraint)
        self._unconstrained = transform.inv(jnp.asarray(value))

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)


class Real(AbstractUnwrappable):
    """Unconstrained parameter. unwrap() returns the value unchanged."""

    _constraint = constraints.real
    value: jax.Array

    def __init__(self, value):
        self.value = jnp.asarray(value)

    def unwrap(self):
        return self.value


class SigmoidBounded(AbstractUnwrappable):
    """Parameter bounded to [low, high] via sigmoid bijection."""

    _unconstrained: jax.Array
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __init__(self, value, *, low=0.0, high=1.0):
        value = jnp.asarray(value)
        transform = biject_to(constraints.interval(low, high))
        self._unconstrained = transform.inv(value)
        self.low = low
        self.high = high

    @property
    def _constraint(self):
        return constraints.interval(self.low, self.high)

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)


class LowerTriangular(AbstractUnwrappable):
    """Lower-triangular matrix parameter with positive diagonal (Cholesky factor).

    Stored as a flat vector; ``unwrap()`` fills a lower-triangular matrix
    with softplus applied to the diagonal entries.
    """

    _constraint = constraints.softplus_lower_cholesky
    _flat: jax.Array

    def __init__(self, value):
        value = jnp.asarray(value)
        self._flat = _dtype_preserving_lower_cholesky.inv(value)

    def unwrap(self):
        return _dtype_preserving_lower_cholesky(self._flat)


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
        w = self.W.unwrap() if isinstance(self.W, AbstractUnwrappable) else self.W
        k = (
            self.kappa.unwrap()
            if isinstance(self.kappa, AbstractUnwrappable)
            else self.kappa
        )
        return w @ w.T + jnp.diag(k)
