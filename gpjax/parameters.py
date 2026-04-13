import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from numpyro.distributions import biject_to, constraints
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


class PositiveReal(AbstractUnwrappable):
    """Strictly positive parameter.

    Stored unconstrained via the inverse of the softplus transform;
    ``unwrap()`` applies softplus to recover the constrained value.
    """

    _constraint = constraints.softplus_positive
    _unconstrained: jax.Array

    def __init__(self, value):
        transform = biject_to(self._constraint)
        self._unconstrained = transform.inv(jnp.asarray(value, dtype=jnp.float64))

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
        self._unconstrained = transform.inv(jnp.asarray(value, dtype=jnp.float64))

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)


class Real(AbstractUnwrappable):
    """Unconstrained parameter. unwrap() returns the value unchanged."""

    _constraint = constraints.real
    value: jax.Array

    def __init__(self, value):
        self.value = jnp.asarray(value, dtype=jnp.float64)

    def unwrap(self):
        return self.value


class SigmoidBounded(AbstractUnwrappable):
    """Parameter bounded to [low, high] via sigmoid bijection."""

    _unconstrained: jax.Array
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __init__(self, value, *, low=0.0, high=1.0):
        value = jnp.asarray(value, dtype=jnp.float64)
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
        value = jnp.asarray(value, dtype=jnp.float64)
        transform = biject_to(self._constraint)
        self._flat = transform.inv(value)

    def unwrap(self):
        return biject_to(self._constraint)(self._flat)


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
