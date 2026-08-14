import math

import equinox as eqx
import jax
from jax.nn import softplus
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as npd
from numpyro.distributions import biject_to, constraints
from numpyro.distributions.transforms import SoftplusLowerCholeskyTransform
import paramax
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


def val(x):
    """Return a parameter's constrained value.

    Call this wherever a parameter meets arithmetic. Models are always held in
    their wrapped form -- including the model ``fit`` returns -- so ``val`` is
    the single point at which the constraining bijection is applied.

    Safe to apply to anything: a parameter is resolved to its constrained value
    (recursively, so nested wrappers such as ``paramax.non_trainable`` are
    handled), while a plain array or float is returned unchanged.

    Args:
        x: A parameter, or any value that does not need unwrapping.

    Returns:
        The constrained value of ``x`` if it is a parameter, else ``x`` itself.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.parameters import PositiveReal, val
        >>> float(val(PositiveReal(jnp.array(2.0))))
        2.0
        >>> float(val(jnp.array(2.0)))
        2.0
    """
    return paramax.unwrap(x) if isinstance(x, AbstractUnwrappable) else x


class _OpaquePrior:
    """Non-pytree holder for a prior distribution.

    A NumPyro ``Distribution`` is a pytree whose hyperparameters are usually
    JAX arrays. Holding the distribution in a plain object, which is not registered as
    a pytree, prevents its parameters from being trained. Solely marking the NumPyro
    ``Distribution`` itself as a static Equinox field does not work as Equinox
    complains about having JAX arrays present in the static field.
    """

    def __init__(self, distribution: npd.Distribution):
        self.distribution = distribution

    def __repr__(self) -> str:
        return repr(self.distribution)


class Parameter(AbstractUnwrappable):
    """Base class for GPJax parameters, carrying an optional prior.

    The prior is static and stored as an _OpaquePrior, which is not registered as a
    pytree node. This prevents the prior's parameters from being optimised.

    Attach one by passing ``prior=`` to any parameter and read it back from the
    ``prior`` property::

        PositiveReal(1.0, prior=numpyro.distributions.LogNormal(0.0, 1.0))

    :func:`collect_log_prior` sums these over a model, and
    :func:`gpjax.objectives.with_log_prior` turns that sum into
    MAP-regularised fitting. Note that the prior is placed in *constrained* space.
    """

    _prior: _OpaquePrior | None = eqx.field(static=True)

    @property
    def prior(self) -> npd.Distribution | None:
        """The prior distribution attached to this parameter, if any."""
        return None if self._prior is None else self._prior.distribution


class PositiveReal(Parameter):
    """Strictly positive parameter.

    Stored unconstrained via the inverse of the softplus transform;
    ``unwrap()`` applies softplus to recover the constrained value.
    """

    _constraint = constraints.softplus_positive
    _unconstrained: jax.Array

    def __init__(self, value, prior=None):
        transform = biject_to(self._constraint)
        self._unconstrained = transform.inv(jnp.asarray(value))
        self._prior = None if prior is None else _OpaquePrior(prior)

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)


class NonNegativeReal(Parameter):
    """Non-negative parameter (semantically allows zero, e.g. jitter, noise floor).

    Uses the same softplus bijection as PositiveReal. The distinction is
    semantic: NonNegativeReal signals that zero is a meaningful boundary.
    """

    _constraint = constraints.softplus_positive
    _unconstrained: jax.Array

    def __init__(self, value, prior=None):
        transform = biject_to(self._constraint)
        self._unconstrained = transform.inv(jnp.asarray(value))
        self._prior = None if prior is None else _OpaquePrior(prior)

    def unwrap(self):
        return biject_to(self._constraint)(self._unconstrained)


class Real(Parameter):
    """Unconstrained parameter. unwrap() returns the value unchanged."""

    _constraint = constraints.real
    value: jax.Array

    def __init__(self, value, prior=None):
        self.value = jnp.asarray(value)
        self._prior = None if prior is None else _OpaquePrior(prior)

    def unwrap(self):
        return self.value


class SigmoidBounded(Parameter):
    """Parameter bounded to [low, high] via sigmoid bijection."""

    _unconstrained: jax.Array
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __init__(self, value, *, low=0.0, high=1.0, prior=None):
        value = jnp.asarray(value)
        transform = biject_to(constraints.interval(low, high))
        self._unconstrained = transform.inv(value)
        self.low = low
        self.high = high
        self._prior = None if prior is None else _OpaquePrior(prior)

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
        w = val(self.W)
        k = val(self.kappa)
        return w @ w.T + jnp.diag(k)


def collect_log_prior(model) -> jax.Array:
    r"""Sum the log-densities of every prior attached to a model's parameters.

    Walks ``model`` for :class:`Parameter` leaves carrying a ``prior`` and
    accumulates :math:`\sum_i \log p(\theta_i)`, evaluated at each parameter's
    *constrained* value. Parameters without a prior contribute nothing.

    Must be called on a model whose parameters are still wrapped --
    ``paramax.unwrap`` replaces :class:`Parameter` leaves with bare arrays and
    so discards the priors along with them.

    Args:
        model: Any pytree containing :class:`Parameter` leaves.

    Returns:
        Scalar total log-prior density; ``0.0`` when no parameter has a prior.
    """
    total = jnp.asarray(0.0)

    def is_param(leaf):
        return isinstance(leaf, Parameter)

    for leaf in jax.tree.leaves(model, is_leaf=is_param):
        if is_param(leaf) and leaf.prior is not None:
            total = total + jnp.sum(leaf.prior.log_prob(val(leaf)))
    return total


__all__ = [
    "CoregionalizationMatrix",
    "LowerTriangular",
    "NonNegativeReal",
    "Parameter",
    "PositiveReal",
    "Real",
    "SigmoidBounded",
    "collect_log_prior",
    "val",
]
