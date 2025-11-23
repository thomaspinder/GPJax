import typing as tp

from flax import nnx
from jax.experimental import checkify
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.typing import ArrayLike
import numpyro.distributions.transforms as npt

from gpjax.numpyro_extras import FillTriangularTransform

T = tp.TypeVar("T", bound=tp.Union[ArrayLike, list[float]])
ParameterTag = str


def transform(
    params: nnx.State,
    params_bijection: tp.Dict[str, npt.Transform],
    inverse: bool = False,
) -> nnx.State:
    r"""Transforms parameters using a bijector.

    Example:
        >>> from gpjax.parameters import PositiveReal, transform
        >>> import jax.numpy as jnp
        >>> import numpyro.distributions.transforms as npt
        >>> from flax import nnx
        >>> params = nnx.State(
        ...     {
        ...         "a": PositiveReal(jnp.array([1.0])),
        ...         "b": PositiveReal(jnp.array([2.0])),
        ...     }
        ... )
        >>> params_bijection = {'positive': npt.SoftplusTransform()}
        >>> transformed_params = transform(params, params_bijection)
        >>> print(transformed_params["a"].value)
        [1.3132617]

    Args:
        params: A nnx.State object containing parameters to be transformed.
        params_bijection: A dictionary mapping parameter types to bijectors.
        inverse: Whether to apply the inverse transformation.

    Returns:
        State: A new nnx.State object containing the transformed parameters.
    """

    def _inner(param):
        bijector = params_bijection.get(param.tag, npt.IdentityTransform())
        if inverse:
            transformed_value = bijector.inv(param.value)
        else:
            transformed_value = bijector(param.value)

        param = param.replace(transformed_value)
        return param

    gp_params, *other_params = params.split(Parameter, ...)

    # Transform each parameter in the state
    transformed_gp_params: nnx.State = jtu.tree_map(
        lambda x: _inner(x) if isinstance(x, Parameter) else x,
        gp_params,
        is_leaf=lambda x: isinstance(x, Parameter),
    )
    return nnx.State.merge(transformed_gp_params, *other_params)


class Parameter(nnx.Variable[T]):
    """Parameter base class.

    All trainable parameters in GPJax should inherit from this class.

    """

    def __init__(self, value: T, tag: ParameterTag, **kwargs):
        _check_is_arraylike(value)

        super().__init__(value=jnp.asarray(value), **kwargs)

        # nnx.Variable metadata must be set via set_metadata (direct setattr is disallowed).
        self.set_metadata(tag=tag)

    @property
    def tag(self) -> ParameterTag:
        """Return the parameter's constraint tag."""
        return self.metadata.get("tag", "real")


class NonNegativeReal(Parameter[T]):
    """Parameter that is non-negative."""

    def __init__(self, value: T, tag: ParameterTag = "non_negative", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)
        _safe_assert(_check_is_non_negative, self.value)


class PositiveReal(Parameter[T]):
    """Parameter that is strictly positive."""

    def __init__(self, value: T, tag: ParameterTag = "positive", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)
        _safe_assert(_check_is_positive, self.value)


class Real(Parameter[T]):
    """Parameter that can take any real value."""

    def __init__(self, value: T, tag: ParameterTag = "real", **kwargs):
        super().__init__(value, tag, **kwargs)


class SigmoidBounded(Parameter[T]):
    """Parameter that is bounded between 0 and 1."""

    def __init__(self, value: T, tag: ParameterTag = "sigmoid", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        # Only perform validation in non-JIT contexts
        if (
            not isinstance(value, jnp.ndarray)
            or getattr(value, "aval", None) is not None
        ):
            _safe_assert(
                _check_in_bounds,
                self.value,
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            )


class LowerTriangular(Parameter[T]):
    """Parameter that is a lower triangular matrix."""

    def __init__(self, value: T, tag: ParameterTag = "lower_triangular", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        # Only perform validation in non-JIT contexts
        if (
            not isinstance(value, jnp.ndarray)
            or getattr(value, "aval", None) is not None
        ):
            _safe_assert(_check_is_square, self.value)
            _safe_assert(_check_is_lower_triangular, self.value)


DEFAULT_BIJECTION = {
    "positive": npt.SoftplusTransform(),
    "non_negative": npt.SoftplusTransform(),
    "real": npt.IdentityTransform(),
    "sigmoid": npt.SigmoidTransform(),
    "lower_triangular": FillTriangularTransform(),
}


def _check_is_arraylike(value: T) -> None:
    """Check if a value is array-like.

    Args:
        value: The value to check.

    Raises:
        TypeError: If the value is not array-like.
    """
    if not isinstance(value, (ArrayLike, list)):
        raise TypeError(
            f"Expected parameter value to be an array-like type. Got {value}."
        )


@checkify.checkify
def _check_is_non_negative(value):
    checkify.check(
        jnp.all(value >= 0), "value needs to be non-negative, got {value}", value=value
    )


@checkify.checkify
def _check_is_positive(value):
    checkify.check(
        jnp.all(value > 0), "value needs to be positive, got {value}", value=value
    )


@checkify.checkify
def _check_is_square(value: T) -> None:
    """Check if a value is a square matrix.

    Args:
        value: The value to check.

    Raises:
        ValueError: If the value is not a square matrix.
    """
    checkify.check(
        value.shape[0] == value.shape[1],
        "value needs to be a square matrix, got {value}",
        value=value,
    )


@checkify.checkify
def _check_is_lower_triangular(value: T) -> None:
    """Check if a value is a lower triangular matrix.

    Args:
        value: The value to check.

    Raises:
        ValueError: If the value is not a lower triangular matrix.
    """
    checkify.check(
        jnp.all(jnp.tril(value) == value),
        "value needs to be a lower triangular matrix, got {value}",
        value=value,
    )


@checkify.checkify
def _check_in_bounds(value: T, low: T, high: T) -> None:
    """Check if a value is bounded between low and high.

    Args:
        value: The value to check.
        low: The lower bound.
        high: The upper bound.

    Raises:
        ValueError: If any element of value is outside the bounds.
    """
    checkify.check(
        jnp.all((value >= low) & (value <= high)),
        "value needs to be bounded between {low} and {high}, got {value}",
        value=value,
        low=low,
        high=high,
    )


def _safe_assert(fn: tp.Callable[[tp.Any], None], value: T, **kwargs) -> None:
    error, _ = fn(value, **kwargs)
    checkify.check_error(error)
