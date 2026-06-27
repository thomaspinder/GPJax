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
    return (
        name.replace("float", "f")
        .replace("complex", "c")
        .replace("uint", "u")
        .replace("int", "i")
    )


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


def _collect(model: tp.Any) -> list[ParamRecord]:
    """Walk ``model`` and produce one :class:`ParamRecord` per parameter."""
    records: list[ParamRecord] = []
    paths_leaves, _ = jax.tree_util.tree_flatten_with_path(
        model, is_leaf=_is_param_leaf
    )
    for path, leaf in paths_leaves:
        is_param = _is_param_leaf(leaf)
        if not (is_param or isinstance(leaf, (jax.Array, np.ndarray))):
            continue
        value = paramax.unwrap(leaf) if is_param else leaf
        # A frozen *bare* array stops traversal at a NonTrainable wrapper; label
        # it like its unfrozen form ("Array"), not "NonTrainable".
        if not is_param or isinstance(leaf, paramax.NonTrainable):
            cls = "Array"
        else:
            cls = type(leaf).__name__
        records.append(
            ParamRecord(
                name=jax.tree_util.keystr(path).lstrip("."),
                cls=cls,
                value=value,
                bijector=_bijector_name(leaf) if is_param else "Identity",
                prior="-",
                trainable=not _is_frozen(leaf),
                shape=tuple(getattr(value, "shape", ())),
                dtype=_short_dtype(value.dtype) if hasattr(value, "dtype") else "?",
            )
        )
    return records


def _render(
    records: list[ParamRecord],
    *,
    columns: tp.Sequence[str] = _DEFAULT_COLUMNS,
    title: str | None = None,
    max_array: int = 4,
    precision: int = 3,
) -> Table:
    """Render collected records into a ``rich.Table``."""
    table = Table(title=title, box=box.ROUNDED, title_justify="left")
    for column in columns:
        table.add_column(column, overflow="fold")

    accessors: dict[str, tp.Callable[[ParamRecord], str]] = {
        "Parameter": lambda r: r.name,
        "Class": lambda r: r.cls,
        "Value": lambda r: _format_value(
            r.value, max_array=max_array, precision=precision
        ),
        "Bijector": lambda r: r.bijector,
        "Prior": lambda r: r.prior,
        "Trainable": lambda r: (
            "[green]yes[/green]" if r.trainable else "[dim red]no[/dim red]"
        ),
        "Shape": lambda r: str(r.shape),
        "Dtype": lambda r: r.dtype,
    }

    n_trainable = 0
    for record in records:
        cells = [accessors[column](record) for column in columns]
        table.add_row(*cells, style=None if record.trainable else "dim")
        n_trainable += int(record.trainable)

    table.caption = f"{len(records)} parameters, {n_trainable} trainable"
    table.caption_justify = "left"
    return table
