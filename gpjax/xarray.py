# Copyright 2026 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Bridge between labelled xarray data and GPJax.

A GP in GPJax takes a :class:`~gpjax.dataset.Dataset` of flat ``(N, D)`` inputs
and ``(N, 1)`` outputs. Gridded data arrives as labelled xarray objects instead.
This module converts between the two at the edges of a workflow and nowhere
else:

.. code-block:: python

    data, spec = from_xarray(ds, target="t2m", inputs=["lat", "lon", "elevation"])
    posterior = model.condition(data)
    test_inputs, test_spec = spec.inputs_for(grid)
    out = test_spec.to_xarray(posterior(test_inputs))

``data`` is an ordinary :class:`~gpjax.dataset.Dataset`. Everything needed to
rebuild the grid lives on the :class:`GridSpec`, which never enters ``fit``,
``condition`` or a traced computation.

Requires the optional ``xarray`` extra: ``pip install "gpjax[xarray]"``.
"""

from dataclasses import dataclass

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import Float
import lineax as lx
import numpy as np

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.typing import (
    Array,
    KeyArray,
)

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "gpjax.xarray requires xarray; install it with pip install 'gpjax[xarray]'"
    ) from error


@dataclass(frozen=True, repr=False)
class GridSpec:
    r"""The labelled grid behind a flattened :class:`~gpjax.dataset.Dataset`.

    Row ``i`` of the flattened data is the ``i``-th kept cell of the grid in C
    order over ``dims``. That correspondence is all this object records, and all
    :meth:`inputs_for` and :meth:`to_xarray` need.

    Attributes:
        target: Name of the modelled variable.
        target_attrs: The target's attributes (units, long_name, ...), carried
            onto predictions.
        inputs: Input names, in the column order of ``X``.
        dims: The grid's dims, in the target's order.
        coords: The grid's coordinates, used to rebuild labelled output.
        time_origins: For each datetime input, the timestamp encoded as day 0.
        mask: Boolean array over the full grid; ``True`` marks a cell that has a
            row in the flattened data.
        n_dropped: Number of grid cells dropped for containing NaN.
    """

    target: str
    target_attrs: dict[str, tp.Any]
    inputs: tuple[str, ...]
    dims: tuple[str, ...]
    coords: tp.Mapping[tp.Hashable, xr.DataArray]
    time_origins: dict[str, np.datetime64]
    mask: np.ndarray
    n_dropped: int

    @property
    def n_kept(self) -> int:
        r"""Number of grid cells with a row in the flattened data."""
        return int(self.mask.sum())

    def __repr__(self) -> str:
        r"""Summarise the grid without printing its coordinates or mask."""
        grid = dict(zip(self.dims, self.mask.shape, strict=True))
        return (
            f"GridSpec(target={self.target!r}, inputs={self.inputs}, grid={grid}, "
            f"kept={self.n_kept}, dropped={self.n_dropped})"
        )

    def inputs_for(
        self, obj: tp.Union[xr.Dataset, xr.DataArray]
    ) -> tuple[Float[Array, "M D"], "GridSpec"]:
        r"""Build prediction inputs on a new grid, encoded as in training.

        The new grid is the broadcast of this spec's inputs as found in ``obj``;
        no target is needed. Its dims follow the training grid's order, with any
        new dims after them. Datetime inputs reuse the training time origins, so
        a date maps to the same number here as it did in training. Cells where an
        input is NaN get no row, and come back as NaN from :meth:`to_xarray`.

        Args:
            obj: Labelled data holding every input named by this spec.

        Returns:
            The ``(M, D)`` prediction inputs and the ``GridSpec`` of the new grid,
            which carries this spec's target name and attributes.

        Raises:
            ValueError: If an input is missing from ``obj``.
            TypeError: If an input is non-numeric, or is a datetime now but was
                not in training (or the reverse).
        """
        dataset = _as_dataset(obj)
        variables = _resolve(dataset, self.inputs)
        input_dims = dict.fromkeys(dim for var in variables for dim in var.dims)
        dims = tuple(
            [dim for dim in self.dims if dim in input_dims]
            + [dim for dim in input_dims if dim not in self.dims]
        )
        grid = xr.broadcast(*variables)[0].transpose(*dims)
        input_matrix = _input_matrix(variables, grid, dims, self.time_origins)
        keep = np.isfinite(input_matrix).all(axis=1)
        test_spec = GridSpec(
            target=self.target,
            target_attrs=dict(self.target_attrs),
            inputs=self.inputs,
            dims=dims,
            coords=_grid_coords(dataset, dims),
            time_origins=self.time_origins,
            mask=keep.reshape(grid.shape),
            n_dropped=int(keep.size - keep.sum()),
        )
        return jnp.asarray(input_matrix[keep]), test_spec

    def to_xarray(
        self,
        dist: GaussianDistribution,
        *,
        num_samples: tp.Optional[int] = None,
        key: tp.Optional[KeyArray] = None,
    ) -> xr.Dataset:
        r"""Map a predictive distribution back onto the labelled grid.

        Args:
            dist: A distribution over exactly the cells this spec kept, in
                flattened order -- e.g. ``posterior(test_inputs)`` for inputs
                built by :meth:`inputs_for`.
            num_samples: If given, return this many joint draws from ``dist``
                instead of its mean and variance.
            key: PRNG key for the draws; required with ``num_samples``.

        Returns:
            By default, an ``xr.Dataset`` holding ``{target}_mean`` and
            ``{target}_variance`` over the grid. With ``num_samples``, one
            variable ``{target}`` over ``("sample", *dims)``. Dropped cells are
            NaN either way.

        Raises:
            ValueError: If ``dist`` does not match the number of kept cells, if
                ``num_samples`` is given without ``key``, or if samples are
                requested from a distribution holding only marginal variances.
        """
        n_points = dist.mean.shape[0]
        if n_points != self.n_kept:
            raise ValueError(
                f"distribution has {n_points} points but this spec expects "
                f"{self.n_kept}; was it built from a different grid?"
            )
        if num_samples is not None:
            return self._samples(dist, num_samples, key)
        variance_attrs = dict(self.target_attrs)
        if "units" in variance_attrs:
            variance_attrs["units"] = f"({variance_attrs['units']})^2"
        return xr.Dataset(
            {
                f"{self.target}_mean": self._scatter(dist.mean, self.target_attrs),
                f"{self.target}_variance": self._scatter(dist.variance, variance_attrs),
            }
        )

    def _samples(
        self,
        dist: GaussianDistribution,
        num_samples: int,
        key: tp.Optional[KeyArray],
    ) -> xr.Dataset:
        r"""Joint draws from ``dist``, scattered onto the grid per sample."""
        if key is None:
            raise ValueError("num_samples requires a PRNG key, e.g. key=jr.key(0)")
        if lx.is_diagonal(dist.scale):
            # Independent draws would ignore the spatial correlation, so any
            # aggregate over cells (a region, a portfolio) would be overconfident.
            raise ValueError(
                "this distribution holds only marginal variances, so it cannot "
                'give joint samples; predict with covariance="dense" to keep '
                "the correlation between cells"
            )
        attrs = {
            **self.target_attrs,
            "description": "joint posterior predictive draws",
        }
        draws = self._scatter(dist.sample(key, (num_samples,)), attrs, ("sample",))
        return xr.Dataset({self.target: draws})

    def _scatter(
        self,
        values: Float[Array, "... N"],
        attrs: dict[str, tp.Any],
        leading_dims: tuple[str, ...] = (),
    ) -> xr.DataArray:
        r"""Place per-kept-cell ``values`` into a NaN-filled labelled grid.

        Any leading axes of ``values`` (e.g. samples) are kept as
        ``leading_dims`` in front of the grid's dims.
        """
        values = np.asarray(values)
        leading_shape = values.shape[:-1]
        grid = np.full((*leading_shape, *self.mask.shape), np.nan)
        grid[..., self.mask] = values
        return xr.DataArray(
            grid, coords=self.coords, dims=(*leading_dims, *self.dims), attrs=attrs
        )


def from_xarray(
    obj: tp.Union[xr.Dataset, xr.DataArray],
    target: str,
    inputs: tp.Sequence[str],
    *,
    dropna: bool = True,
) -> tuple[Dataset, GridSpec]:
    r"""Flatten labelled xarray data into a :class:`~gpjax.dataset.Dataset`.

    Every input is broadcast onto the target's grid, so an input on fewer dims
    (e.g. ``elevation(lat, lon)`` for a ``(time, lat, lon)`` target) repeats
    along the rest. Datetime inputs become float days since their earliest
    timestamp.

    Args:
        obj: The labelled data. A ``DataArray`` must be named, and that name is
            the target.
        target: Name of the data variable to model. Its dims define the grid.
        inputs: Coordinates and/or data variables to use as inputs, in the order
            of the columns of ``X``.
        dropna: Drop grid cells where the target or any input is NaN. When
            ``False``, such cells raise instead.

    Returns:
        The flattened ``Dataset`` and the ``GridSpec`` needed to map
        predictions back onto the grid.

    Raises:
        ValueError: If a name is missing, ``inputs`` is empty, repeats a name or
            includes the target, an input has a dim the target lacks, a
            ``DataArray`` is unnamed, or no cells survive NaN handling (or any
            NaN is present when ``dropna=False``).
        TypeError: If ``inputs`` is a single string, or the target or an input
            is not numeric (inputs may also be ``datetime64``).
    """
    if isinstance(inputs, str):
        raise TypeError(
            f"inputs must be a list of names, e.g. [{inputs!r}], not a string"
        )
    if not inputs:
        raise ValueError("inputs must name at least one coordinate or variable")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs {list(inputs)} contain duplicate names")
    if target in inputs:
        raise ValueError(f"the target {target!r} cannot also be an input")
    dataset = _as_dataset(obj)
    target_values, *variables = _resolve(dataset, [target, *inputs])
    dims = tuple(target_values.dims)
    time_origins = {
        variable.name: _earliest(variable.values)
        for variable in variables
        if np.issubdtype(variable.dtype, np.datetime64)
    }
    input_matrix = _input_matrix(variables, target_values, dims, time_origins)
    outputs = _as_float(target_values.values.reshape(-1, 1), f"target {target!r}")

    keep = np.isfinite(input_matrix).all(axis=1) & np.isfinite(outputs[:, 0])
    n_dropped = int(keep.size - keep.sum())
    if n_dropped and not dropna:
        raise ValueError(
            f"{n_dropped} cell(s) contain NaN in the target or an input; pass "
            "dropna=True to drop them"
        )
    if not keep.any():
        raise ValueError("no cells are left once NaN cells are dropped")

    data = Dataset(X=jnp.asarray(input_matrix[keep]), y=jnp.asarray(outputs[keep]))
    spec = GridSpec(
        target=target,
        target_attrs=dict(target_values.attrs),
        inputs=tuple(inputs),
        dims=dims,
        coords=target_values.coords,
        time_origins=time_origins,
        mask=keep.reshape(target_values.shape),
        n_dropped=n_dropped,
    )
    return data, spec


def _as_dataset(obj: tp.Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    r"""Promote a named ``DataArray`` to a one-variable ``Dataset``."""
    if isinstance(obj, xr.Dataset):
        return obj
    if obj.name is None:
        raise ValueError(
            "a DataArray must have a name to be used as the target; set one with "
            ".rename('name') or pass an xr.Dataset"
        )
    return obj.to_dataset()


def _resolve(dataset: xr.Dataset, names: tp.Sequence[str]) -> list[xr.DataArray]:
    r"""Look up coordinates or data variables by name, reporting every miss."""
    missing = [name for name in names if name not in dataset.variables]
    if missing:
        raise ValueError(
            f"{missing} not found; available coordinates and variables are "
            f"{sorted(map(str, dataset.variables))}"
        )
    return [dataset[name] for name in names]


def _grid_coords(
    dataset: xr.Dataset, dims: tuple[str, ...]
) -> dict[tp.Hashable, xr.DataArray]:
    r"""Every coordinate of ``dataset`` that lives on the grid's dims."""
    return {
        name: coord
        for name, coord in dataset.coords.items()
        if set(coord.dims) <= set(dims)
    }


def _earliest(timestamps: np.ndarray) -> np.datetime64:
    r"""The earliest non-NaT timestamp, or NaT if there is none."""
    valid = timestamps[~np.isnat(timestamps)]
    return valid.min() if valid.size else np.datetime64("NaT")


def _input_matrix(
    variables: list[xr.DataArray],
    grid: xr.DataArray,
    dims: tuple[str, ...],
    time_origins: dict[str, np.datetime64],
) -> np.ndarray:
    r"""Stack ``variables`` into an ``(N, D)`` float matrix over ``grid``.

    A datetime input must have a time origin, and an input with a time origin
    must still be a datetime: otherwise the same number would mean different
    things in training and prediction.
    """
    columns = []
    for variable in variables:
        name = variable.name
        column = _column(variable, grid, dims)
        is_datetime = np.issubdtype(column.dtype, np.datetime64)
        if is_datetime != (name in time_origins):
            then, now = ("a", "not a") if name in time_origins else ("not a", "a")
            raise TypeError(
                f"input {name!r} was {then} datetime when the spec was built but "
                f"is {now} datetime now; encode it the same way in both"
            )
        if is_datetime:
            days = (column - time_origins[name]) / np.timedelta64(1, "D")
            columns.append(np.where(np.isnat(column), np.nan, days))
        else:
            columns.append(_as_float(column, f"input {name!r}"))
    return np.stack(columns, axis=1)


def _column(
    variable: xr.DataArray, grid: xr.DataArray, dims: tuple[str, ...]
) -> np.ndarray:
    r"""Broadcast ``variable`` onto ``grid`` and flatten it in C order over ``dims``.

    A variable on fewer dims than the grid repeats along the missing ones. A
    variable on a dim the grid lacks has no cell to land in, so it is rejected.
    """
    extra_dims = [dim for dim in variable.dims if dim not in dims]
    if extra_dims:
        raise ValueError(
            f"input {variable.name!r} has dims {extra_dims} that the grid "
            f"{dims} does not have"
        )
    return xr.broadcast(grid, variable)[1].transpose(*dims).values.ravel()


def _as_float(values: np.ndarray, description: str) -> np.ndarray:
    r"""Cast numeric or boolean ``values`` to float; reject anything else."""
    if np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        return values.astype(float)
    raise TypeError(
        f"{description} has dtype {values.dtype}, which is not numeric; convert "
        "it before calling from_xarray"
    )


__all__ = [
    "GridSpec",
    "from_xarray",
]
