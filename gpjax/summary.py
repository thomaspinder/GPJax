"""Human-readable model summaries for GPJax pytrees.

Render any GPJax model (kernel, prior, posterior, likelihood, variational
family, ...) as a flat ``rich`` table -- one row per parameter -- in the
spirit of GPflow's ``print_summary``. Use the free function :func:`summarise`,
or rely on the ``__rich__`` / ``_repr_mimebundle_`` hooks attached to the
user-facing abstract bases for ``rich.print`` and notebook auto-rendering.
"""

import io

import beartype.typing as tp
import jax
import numpy as np
import paramax
from numpyro.distributions import biject_to
from paramax import AbstractUnwrappable
from rich import box
from rich.console import Console
from rich.table import Table

from gpjax.parameters import SigmoidBounded

__all__ = ["summarise"]

# Default GPflow-style column set, in display order.
_DEFAULT_COLUMNS: tuple[str, ...] = (
    "Parameter",
    "Class",
    "Value",
    "Bijector",
    "Prior",
    "Trainable",
    "Shape",
    "Dtype",
)

# Friendly labels for the numpyro transforms returned by ``biject_to``.
_BIJECTOR_NAMES: dict[str, str] = {
    "SoftplusTransform": "Softplus",
    "IdentityTransform": "Identity",
    "SoftplusLowerCholeskyTransform": "LowerCholesky",
}


class ParamRecord(tp.NamedTuple):
    """A single rendered parameter row (rendering-agnostic)."""

    name: str
    cls: str
    value: tp.Any
    bijector: str
    prior: str
    trainable: bool
    shape: tuple[int, ...]
    dtype: str


def _is_param_leaf(x: tp.Any) -> bool:
    """Stop pytree traversal at parameter objects."""
    return isinstance(x, AbstractUnwrappable)


def _is_frozen(leaf: tp.Any) -> bool:
    """True iff ``leaf``'s subtree contains a ``paramax.NonTrainable``."""
    leaves = jax.tree_util.tree_leaves(
        leaf, is_leaf=lambda y: isinstance(y, paramax.NonTrainable)
    )
    return any(isinstance(x, paramax.NonTrainable) for x in leaves)


def _bijector_name(leaf: tp.Any) -> str:
    """Friendly bijector label for a parameter leaf."""
    if isinstance(leaf, SigmoidBounded):
        return f"Sigmoid[{leaf.low:g}, {leaf.high:g}]"
    constraint = getattr(leaf, "_constraint", None)
    if constraint is None:
        return "Identity"
    raw = type(biject_to(constraint)).__name__
    return _BIJECTOR_NAMES.get(raw, raw)


def _short_dtype(dtype: tp.Any) -> str:
    """Abbreviate a dtype name, e.g. float64 -> f64."""
    name = np.dtype(dtype).name
    return name.replace("float", "f").replace("complex", "c").replace("int", "i")


def _is_traced(value: tp.Any) -> bool:
    """True for abstract values seen under a ``jax.jit`` trace."""
    return isinstance(value, jax.core.Tracer)


def _format_value(value: tp.Any, *, max_array: int, precision: int) -> str:
    """Format a parameter value; never crashes on traced values."""
    if _is_traced(value):
        return f"<traced {_short_dtype(value.dtype)}{list(value.shape)}>"
    arr = np.asarray(value)
    if arr.ndim == 0:
        return f"{float(arr):.{precision}g}"
    flat = arr.reshape(-1)
    shown = ", ".join(f"{float(v):.{precision}g}" for v in flat[:max_array])
    if flat.size > max_array:
        shown += ", ..."
    return f"[{shown}]"
