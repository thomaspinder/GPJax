# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
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
r"""Exponential-family machinery for natural-gradient variational inference.

Implements the coordinate maps and update of Salimbeni, Eleftheriadis and
Hensman (2018), *Natural Gradients in Practice: Non-Conjugate Variational
Inference in Gaussian Process Models* (arXiv:1803.09151).

A Gaussian $q(\mathbf u)=\mathcal N(\mathbf m,\mathbf S)$ has sufficient statistics
$[\mathbf u,\mathbf u\mathbf u^\top]$ and three coordinate systems:

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - coordinates
     - symbol
     - contents
   * - moment (stored)
     - $\boldsymbol\xi$
     - $(\mathbf m,\ \mathbf L)$ with $\mathbf S=\mathbf L\mathbf L^\top$
   * - natural
     - $\boldsymbol\theta$
     - $(\mathbf S^{-1}\mathbf m,\ -\tfrac12\mathbf S^{-1})$
   * - expectation
     - $\boldsymbol\eta$
     - $(\mathbf m,\ \mathbf S+\mathbf m\mathbf m^\top)$

The Fisher information is $\partial\boldsymbol\eta/\partial\boldsymbol\theta$;
thus $\tilde\nabla_{\boldsymbol\theta}\ell
=\partial\ell/\partial\boldsymbol\eta$ without forming a Fisher matrix. The update
is $\boldsymbol\theta\leftarrow\boldsymbol\theta
-\gamma\,\partial\ell/\partial\boldsymbol\eta$. For $\gamma\in[0,1]$ it is a convex
combination in natural coordinates; $\gamma=1$ reaches the conjugate optimum in one
step.

The dual t-SVGP update of Adam, Chang, Khan and Solin (2021),
*Dual Parameterization of Sparse Variational Gaussian Processes*
(arXiv:2111.03412), stores sites $\boldsymbol\lambda$ with
$\boldsymbol\eta=\boldsymbol\eta_0(\boldsymbol\theta)+\boldsymbol\lambda$.
Since $\nabla_{\boldsymbol\mu}\operatorname{KL}=\boldsymbol\lambda$, it need not
differentiate the KL or round-trip through coordinates:
$\boldsymbol\lambda\leftarrow(1-\rho)\boldsymbol\lambda
+\rho\,\nabla_{\boldsymbol\mu}\mathcal L_{\text{ell}}$.
This matches Salimbeni iterates at $\gamma=\rho$ when computed
$\boldsymbol\beta\ge0$ leaves ``beta_floor`` inert. GPJax's clipped probit link
can violate that condition in the far tails.
"""

import functools
import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jaxtyping import Float
import paramax

from gpjax.dataset import Dataset
from gpjax.likelihoods import AbstractLikelihood
from gpjax.objectives import Objective
from gpjax.parameters import (
    LowerTriangular,
    Real,
    val,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from gpjax.variational_families import (
    AbstractVariationalFamily,
    DualVariationalGaussian,
    VariationalGaussian,
    _symmetrise,
)

VF = tp.TypeVar("VF", bound=AbstractVariationalFamily)


def _lower_solve(
    factor: Float[Array, "M M"], rhs: Float[Array, "M K"]
) -> Float[Array, "M K"]:
    """Solve ``factor @ x = rhs`` for lower-triangular ``factor``.

    Args:
        factor: A lower-triangular matrix with a strictly positive diagonal.
        rhs: The right-hand side of the triangular system.

    Returns:
        The solution ``x``.
    """
    return jsp.linalg.solve_triangular(factor, rhs, lower=True)


def _upper_solve(
    factor: Float[Array, "M M"], rhs: Float[Array, "M K"]
) -> Float[Array, "M K"]:
    """Solve ``factor @ x = rhs`` for upper-triangular ``factor``.

    Args:
        factor: An upper-triangular matrix with a strictly positive diagonal.
        rhs: The right-hand side of the triangular system.

    Returns:
        The solution ``x``.
    """
    return jsp.linalg.solve_triangular(factor, rhs, lower=False)


def expectation_from_moments(
    variational_mean: Float[Array, "M 1"],
    variational_root_covariance: Float[Array, "M M"],
) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
    r"""Map $\boldsymbol\xi=(\mathbf m,\mathbf L)$ to the expectation parameters.

    Args:
        variational_mean: The variational mean $\mathbf m$, stored as an
            $M\times1$ column.
        variational_root_covariance: The lower-triangular root $\mathbf L$ with
            $\mathbf S=\mathbf L\mathbf L^\top$.

    Returns:
        The expectation parameters $(\boldsymbol\eta_1,\mathbf H_2)$.

    Notes:
        $\boldsymbol\eta_1=\mathbf m$ and
        $\mathbf H_2=\mathbf L\mathbf L^\top+\mathbf m\mathbf m^\top$.
        No jitter or solve is needed.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.natural_gradients import expectation_from_moments
        >>>
        >>> mean = jnp.array([[1.0], [2.0]])
        >>> root_covariance = jnp.eye(2)
        >>> expectation_vector, expectation_matrix = expectation_from_moments(
        ...     mean, root_covariance
        ... )
        >>> [round(entry, 3) for entry in expectation_matrix.ravel().tolist()]
        [2.0, 2.0, 2.0, 5.0]
    """
    expectation_matrix = (
        variational_root_covariance @ variational_root_covariance.T
        + variational_mean @ variational_mean.T
    )
    return variational_mean, expectation_matrix


def natural_from_moments(
    variational_mean: Float[Array, "M 1"],
    variational_root_covariance: Float[Array, "M M"],
) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
    r"""Map $\boldsymbol\xi=(\mathbf m,\mathbf L)$ to the natural parameters.

    Args:
        variational_mean: The variational mean $\mathbf m$, stored as an
            $M\times1$ column.
        variational_root_covariance: The lower-triangular root $\mathbf L$ with
            $\mathbf S=\mathbf L\mathbf L^\top$.

    Returns:
        The natural parameters $(\boldsymbol\theta_1,\boldsymbol\Theta_2)$.

    Notes:
        With $\mathbf P=(\mathbf L\mathbf L^\top)^{-1}$,
        $\boldsymbol\theta_1=\mathbf P\mathbf m$ and
        $\boldsymbol\Theta_2=-\tfrac12\mathbf P$. Triangular solves avoid forming
        $\mathbf S$; no jitter is needed for a root with positive diagonal.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.natural_gradients import natural_from_moments
        >>>
        >>> mean = jnp.array([[1.0], [2.0]])
        >>> root_covariance = jnp.eye(2)
        >>> natural_vector, natural_matrix = natural_from_moments(
        ...     mean, root_covariance
        ... )
        >>> [round(entry, 3) for entry in jnp.diag(natural_matrix).tolist()]
        [-0.5, -0.5]
    """
    num_inducing = variational_root_covariance.shape[0]
    root_inverse = _lower_solve(
        variational_root_covariance,
        jnp.eye(num_inducing, dtype=variational_root_covariance.dtype),
    )
    precision = root_inverse.T @ root_inverse
    return precision @ variational_mean, -0.5 * precision


def moments_from_expectation(
    expectation_vector: Float[Array, "M 1"],
    expectation_matrix: Float[Array, "M M"],
    map_jitter: ScalarFloat | int = 0.0,
) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
    r"""Map the expectation parameters back to $\boldsymbol\xi=(\mathbf m,\mathbf L)$.

    Args:
        expectation_vector: The first expectation parameter $\boldsymbol\eta_1$.
        expectation_matrix: The second expectation parameter $\mathbf H_2$.
        map_jitter: Diagonal jitter $\varepsilon$ before Cholesky (default ``0.0``).
            Unlike ``Prior.jitter``, a non-zero value biases $\mathbf S$ by
            $\approx\varepsilon\lVert\mathbf S\rVert^2$.

    Returns:
        The moment parameters $(\mathbf m,\mathbf L)$.

    Notes:
        $\mathbf S=\operatorname{sym}(\mathbf H_2-
        \boldsymbol\eta_1\boldsymbol\eta_1^\top)$ and
        $\mathbf L=\operatorname{chol}(\mathbf S+\varepsilon\mathbf I)$.
        When $\lVert\mathbf m\rVert^2\gg\lVert\mathbf S\rVert$, subtraction
        can cancel; prefer the whitened family. This map has no admissibility
        guard: the recovered root may be finite but inaccurate before it becomes
        ``NaN``.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.natural_gradients import moments_from_expectation
        >>>
        >>> expectation_vector = jnp.array([[1.0], [2.0]])
        >>> expectation_matrix = jnp.array([[2.0, 2.0], [2.0, 5.0]])
        >>> mean, root_covariance = moments_from_expectation(
        ...     expectation_vector, expectation_matrix
        ... )
        >>> [round(entry, 3) for entry in root_covariance.ravel().tolist()]
        [1.0, 0.0, 0.0, 1.0]
    """
    num_inducing = expectation_matrix.shape[0]
    covariance = _symmetrise(
        expectation_matrix - expectation_vector @ expectation_vector.T
    )
    identity = jnp.eye(num_inducing, dtype=covariance.dtype)
    root_covariance = jnp.linalg.cholesky(covariance + map_jitter * identity)
    return expectation_vector, root_covariance


def moments_from_natural(
    natural_vector: Float[Array, "M 1"],
    natural_matrix: Float[Array, "M M"],
    map_jitter: ScalarFloat | int = 0.0,
) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
    r"""Map the natural parameters back to $\boldsymbol\xi=(\mathbf m,\mathbf L)$.

    Args:
        natural_vector: The first natural parameter $\boldsymbol\theta_1$.
        natural_matrix: The second natural parameter $\boldsymbol\Theta_2$;
            its jittered precision must be positive definite for a finite result.
        map_jitter: Jitter $\varepsilon$ added to the diagonal before each Cholesky.
            Defaults to ``0.0``; see :func:`moments_from_expectation` for why it is
            not inherited from the family.

    Returns:
        The moment parameters $(\mathbf m,\mathbf L)$.

    Notes:
        Set $\mathbf P_\varepsilon=\operatorname{sym}(-2\boldsymbol\Theta_2)
        +\varepsilon\mathbf I$. Cholesky and triangular solves give
        $\mathbf m=\mathbf P_\varepsilon^{-1}\boldsymbol\theta_1$ and
        $\mathbf S=\mathbf P_\varepsilon^{-1}$; a second Cholesky of
        $\mathbf S+\varepsilon\mathbf I$ gives $\mathbf L$. Failed Cholesky
        returns ``NaN`` rather than raising, permitting ``jit``-compatible backoff.

    Example:
        >>> import jax.numpy as jnp
        >>> from gpjax.natural_gradients import moments_from_natural
        >>>
        >>> natural_vector = jnp.array([[1.0], [2.0]])
        >>> natural_matrix = -0.5 * jnp.eye(2)
        >>> mean, root_covariance = moments_from_natural(
        ...     natural_vector, natural_matrix
        ... )
        >>> [round(entry, 3) for entry in mean.ravel().tolist()]
        [1.0, 2.0]
    """
    num_inducing = natural_matrix.shape[0]
    precision = _symmetrise(-2.0 * natural_matrix)
    identity = jnp.eye(num_inducing, dtype=precision.dtype)
    root_precision = jnp.linalg.cholesky(precision + map_jitter * identity)
    root_precision_inverse = _lower_solve(root_precision, identity)
    covariance = _symmetrise(root_precision_inverse.T @ root_precision_inverse)
    variational_mean = _upper_solve(
        root_precision.T, _lower_solve(root_precision, natural_vector)
    )
    root_covariance = jnp.linalg.cholesky(covariance + map_jitter * identity)
    return variational_mean, root_covariance


@functools.singledispatch
def variational_coordinates(
    variational_family: VF,
) -> tp.Callable[[VF], tuple[tp.Any, ...]]:
    """Return an ``eqx.tree_at`` selector naming the exponential-family coordinates.

    Args:
        variational_family: The variational family whose coordinates are to be
            selected.

    Returns:
        A ``where`` function mapping a family to the tuple of nodes holding its
        exponential-family coordinates.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>> from gpjax.natural_gradients import variational_coordinates
        >>>
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(), kernel=gpx.kernels.RBF()
        ... )
        >>> model = prior * gpx.likelihoods.Gaussian()
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     model=model, inducing_inputs=jnp.linspace(0, 1, 2).reshape(-1, 1)
        ... )
        >>>
        >>> where = variational_coordinates(q)
        >>> [type(node).__name__ for node in where(q)]
        ['Real', 'LowerTriangular']
    """
    raise NotImplementedError(
        f"Natural gradients are not defined for {type(variational_family).__name__}."
    )


@variational_coordinates.register(VariationalGaussian)
def _variational_gaussian_coordinates(
    variational_family: VariationalGaussian,
) -> tp.Callable[[VariationalGaussian], tuple[tp.Any, ...]]:
    r"""Select $(\mathbf m,\mathbf L)$ on the Salimbeni-family Gaussians.

    Also covers ``WhitenedVariationalGaussian`` and
    ``GraphVariationalGaussian``, subclasses storing the same two coordinates;
    whitening changes the loss, not the coordinate maps.

    Args:
        variational_family: The family being partitioned. Unused; dispatch is on its
            type.

    Returns:
        A selector returning ``(variational_mean, variational_root_covariance)``.
    """
    del variational_family
    return lambda tree: (tree.variational_mean, tree.variational_root_covariance)


@variational_coordinates.register(DualVariationalGaussian)
def _dual_variational_coordinates(
    variational_family: DualVariationalGaussian,
) -> tp.Callable[[DualVariationalGaussian], tuple[tp.Any, ...]]:
    r"""Select the dual sites $(\boldsymbol\lambda_1,\boldsymbol\Lambda_2)$.

    The stored sites are an affine image of the natural parameters,
    $\boldsymbol\eta=\boldsymbol\eta_0(\boldsymbol\theta)
    +(\boldsymbol\lambda_1,-\tfrac12\boldsymbol\Lambda_2)$, so they *are* the
    exponential-family coordinates of this family.

    Args:
        variational_family: The family being partitioned. Unused; dispatch is on its
            type.

    Returns:
        A selector returning ``(dual_vector, dual_matrix)``.
    """
    del variational_family
    return lambda tree: (tree.dual_vector, tree.dual_matrix)


def partition_variational(variational_family: VF) -> tuple[VF, VF]:
    """Split a family into (variational-coordinate, hyperparameter) partitions.

    Args:
        variational_family: The variational family to split.

    Returns:
        The coordinate partition and the hyperparameter partition (including
        inducing inputs).

    Notes:
        A prefix filter spec supports different parameter-wrapper fields and
        ``paramax.non_trainable``; renamed coordinates raise ``AttributeError``.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import jax.tree_util as jtu
        >>> import gpjax as gpx
        >>> from gpjax.natural_gradients import partition_variational
        >>>
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(), kernel=gpx.kernels.RBF()
        ... )
        >>> model = prior * gpx.likelihoods.Gaussian()
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     model=model, inducing_inputs=jnp.linspace(0, 1, 2).reshape(-1, 1)
        ... )
        >>>
        >>> variational, hyper = partition_variational(q)
        >>> sorted(jtu.keystr(path) for path, _ in jtu.tree_flatten_with_path(
        ...     variational
        ... )[0])
        ['.variational_mean.value', '.variational_root_covariance._flat']
        >>> len(jtu.tree_leaves(hyper))
        5
    """
    where = variational_coordinates(variational_family)
    spec = jtu.tree_map(lambda _: False, variational_family)
    spec = eqx.tree_at(where, spec, replace=(True, True))
    return eqx.partition(variational_family, spec)


def _contains_non_trainable(node: tp.Any) -> bool:
    """Return whether ``node`` holds a ``paramax.NonTrainable`` anywhere inside it.

    Args:
        node: A subtree of a variational family.

    Returns:
        ``True`` if any part of ``node`` is frozen.

    Notes:
        ``paramax.non_trainable`` wraps leaves within parameter nodes; ``is_leaf``
        exposes those wrappers to the traversal.
    """
    is_frozen = lambda leaf: isinstance(leaf, paramax.NonTrainable)
    return any(map(is_frozen, jtu.tree_leaves(node, is_leaf=is_frozen)))


def _reject_frozen_coordinates(variational_family: VF) -> None:
    r"""Raise if any exponential-family coordinate is ``paramax.non_trainable``.

    Args:
        variational_family: The family whose coordinates are to be checked.

    Raises:
        ValueError: If a coordinate is wrapped in ``paramax.NonTrainable``.

    Notes:
        Partial natural-gradient steps are not meaningful. The static tree check
        runs at trace time; the selector finds nested coordinates and reports
        their full key paths.
    """
    where = variational_coordinates(variational_family)
    coordinates = where(variational_family)

    def is_coordinate(node: tp.Any) -> bool:
        return node is not None and any(node is selected for selected in coordinates)

    paths_and_nodes = jtu.tree_flatten_with_path(
        variational_family, is_leaf=is_coordinate
    )[0]
    frozen = [
        jtu.keystr(path).lstrip(".")
        for path, node in paths_and_nodes
        if is_coordinate(node) and _contains_non_trainable(node)
    ]
    if frozen:
        verb = "are" if len(frozen) > 1 else "is"
        raise ValueError(
            "Natural gradients require every exponential-family coordinate to be "
            f"trainable, but {', '.join(frozen)} {verb} wrapped in "
            "paramax.non_trainable. Drop the wrapper, or optimise this family with "
            "gpjax.fit instead of gpjax.fit_natgrads."
        )


def _first_valid_trial(
    natural_vector: Float[Array, "M 1"],
    natural_matrix: Float[Array, "M M"],
    gradient_vector: Float[Array, "M 1"],
    gradient_matrix: Float[Array, "M M"],
    natgrad_lr: ScalarFloat | int,
    map_jitter: ScalarFloat | int,
    backoff: ScalarFloat | int,
    max_backoff: int,
) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
    r"""Take the largest admissible step from $\{\gamma\beta^k\}_{k=0}^{K}$.

    Args:
        natural_vector: The current $\boldsymbol\theta_1$.
        natural_matrix: The current $\boldsymbol\Theta_2$.
        gradient_vector: $\partial\ell/\partial\boldsymbol\eta_1$.
        gradient_matrix: $\partial\ell/\partial\mathbf H_2$, already symmetrised.
        natgrad_lr: The requested step size $\gamma$.
        map_jitter: Jitter passed through to :func:`moments_from_natural`.
        backoff: The multiplicative shrink factor $\beta\in(0,1)$.
        max_backoff: The number $K$ of shrink attempts after the first.

    Returns:
        The moment parameters of the accepted trial.

    Notes:
        ``vmap`` probes $K+1$ trials for a finite precision Cholesky and mean;
        only the accepted trial completes the covariance map. ``argmax`` selects
        the first valid step; if none is valid it selects the first trial, allowing
        ``NaN`` to propagate instead of silently accepting an invalid step.

        Step sizes use $\boldsymbol\Theta_2$'s dtype: under ``jax_enable_x64``,
        an ``int64`` exponent could otherwise promote ``float32`` and break the
        ``lax.scan`` carry.
    """
    step_sizes = (natgrad_lr * backoff ** jnp.arange(max_backoff + 1)).astype(
        natural_matrix.dtype
    )
    identity = jnp.eye(natural_matrix.shape[0], dtype=natural_matrix.dtype)

    def is_admissible(step_size):
        trial_matrix = natural_matrix - step_size * gradient_matrix
        precision = _symmetrise(-2.0 * trial_matrix)
        root_precision = jnp.linalg.cholesky(precision + map_jitter * identity)
        trial_mean = _upper_solve(
            root_precision.T,
            _lower_solve(root_precision, natural_vector - step_size * gradient_vector),
        )
        return jnp.all(jnp.isfinite(root_precision)) & jnp.all(jnp.isfinite(trial_mean))

    accepted = jnp.argmax(jax.vmap(is_admissible)(step_sizes))
    accepted_step_size = step_sizes[accepted]
    return moments_from_natural(
        natural_vector - accepted_step_size * gradient_vector,
        natural_matrix - accepted_step_size * gradient_matrix,
        map_jitter,
    )


@functools.singledispatch
def natural_gradient_step(
    variational: VF,
    hyper: VF,
    data: Dataset,
    objective: Objective,
    natgrad_lr: ScalarFloat | int,
    *,
    map_jitter: ScalarFloat | int = 0.0,
    backoff: ScalarFloat | int = 0.5,
    max_backoff: int = 5,
    beta_floor: ScalarFloat | int = 1e-8,
) -> tuple[VF, ScalarFloat]:
    r"""One natural-gradient step on the variational coordinates.

    Args:
        variational: The variational partition returned by
            :func:`partition_variational`.
        hyper: The hyperparameter partition returned by
            :func:`partition_variational`.
        data: The (possibly mini-)batch at which the loss is evaluated.
        objective: A loss ``(family, data) -> scalar`` that is *minimised*, e.g.
            ``lambda q, d: -gpjax.objectives.elbo(q, d)``.
        natgrad_lr: The natural-gradient step size $\gamma$.
        map_jitter: Jitter used by the
            $\boldsymbol\theta\leftrightarrow\boldsymbol\xi$ maps.
        backoff: Multiplicative shrink factor applied when a step leaves the
            negative-definite cone.
        max_backoff: Number of shrink attempts after the first.
        beta_floor: Accepted for a uniform dispatch contract; ignored by
            Salimbeni-family updates.

    Returns:
        The updated variational partition and pre-update loss (as in ``fit()``).
        For the Salimbeni branch, the loss uses
        $\boldsymbol\xi(\boldsymbol\eta_t)$: ``map_jitter=0.0`` preserves the stored
        coordinates, while non-zero jitter biases it by $\mathcal O(\varepsilon)$.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> import gpjax as gpx
        >>> from gpjax.natural_gradients import (
        ...     natural_gradient_step,
        ...     partition_variational,
        ... )
        >>>
        >>> xtrain = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> D = gpx.Dataset(X=xtrain, y=jnp.sin(xtrain))
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(), kernel=gpx.kernels.RBF()
        ... )
        >>> model = prior * gpx.likelihoods.Gaussian()
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     model=model, inducing_inputs=jnp.linspace(0, 1, 3).reshape(-1, 1)
        ... )
        >>>
        >>> variational, hyper = partition_variational(q)
        >>> negative_elbo = lambda p, d: -gpx.objectives.elbo(p, d)
        >>> stepped, loss = natural_gradient_step(
        ...     variational, hyper, D, negative_elbo, jnp.asarray(1.0)
        ... )
        >>> updated = eqx.combine(stepped, hyper)
        >>> bool(loss > -gpx.objectives.elbo(updated, D))
        True
    """
    del hyper, data, objective, natgrad_lr, map_jitter, backoff, max_backoff, beta_floor
    raise NotImplementedError(
        f"Natural gradients are not defined for {type(variational).__name__}."
    )


@natural_gradient_step.register(VariationalGaussian)
def _variational_gaussian_step(
    variational: VariationalGaussian,
    hyper: VariationalGaussian,
    data: Dataset,
    objective: Objective,
    natgrad_lr: ScalarFloat | int,
    *,
    map_jitter: ScalarFloat | int = 0.0,
    backoff: ScalarFloat | int = 0.5,
    max_backoff: int = 5,
    beta_floor: ScalarFloat | int = 1e-8,
) -> tuple[VariationalGaussian, ScalarFloat]:
    r"""Salimbeni update (N) for the Gaussian variational families.

    Updates $\boldsymbol\theta\leftarrow\boldsymbol\theta
    -\gamma\,\partial\ell/\partial\boldsymbol\eta$, then stores moment parameters.
    Also handles whitened and graph Gaussians, whose coordinate maps are identical.

    Args:
        variational: The variational partition.
        hyper: The hyperparameter partition.
        data: The batch at which the loss is evaluated.
        objective: The loss being minimised.
        natgrad_lr: The step size $\gamma$.
        map_jitter: Jitter for the coordinate maps.
        backoff: Multiplicative shrink factor for the step-size backoff.
        max_backoff: Number of shrink attempts after the first.
        beta_floor: Unused.

    Returns:
        The updated variational partition and pre-update loss, evaluated at
        $\boldsymbol\xi(\boldsymbol\eta_t)$. Non-zero ``map_jitter`` shifts that
        loss by $\mathcal O(\varepsilon)$.
    """
    del beta_floor
    _reject_frozen_coordinates(variational)

    family = eqx.combine(variational, hyper)
    initial_mean = val(family.variational_mean)
    initial_root_covariance = val(family.variational_root_covariance)

    # Map theta_0 directly from L to avoid cancellation in a round trip via eta_0.
    initial_natural = natural_from_moments(initial_mean, initial_root_covariance)
    initial_expectation = expectation_from_moments(
        initial_mean, initial_root_covariance
    )

    def loss_of_expectation(expectation):
        # Map eta directly to moments; wrapping the root lets the existing
        # objective consume the family unchanged.
        trial_mean, trial_root_covariance = moments_from_expectation(
            *expectation, map_jitter
        )
        trial = eqx.tree_at(
            lambda tree: (tree.variational_mean, tree.variational_root_covariance),
            family,
            (Real(trial_mean), LowerTriangular(trial_root_covariance)),
        )
        return objective(trial, data)

    loss_value, gradient = jax.value_and_grad(loss_of_expectation)(initial_expectation)
    # H_2 is symmetric, so the gradient must be read in the trace pairing on Sym(M).
    # Skipping this is a silent factor-of-two error on the off-diagonals.
    gradient = (gradient[0], _symmetrise(gradient[1]))

    updated_mean, updated_root_covariance = _first_valid_trial(
        *initial_natural,
        *gradient,
        natgrad_lr,
        map_jitter,
        backoff,
        max_backoff,
    )
    variational = eqx.tree_at(
        lambda tree: (tree.variational_mean, tree.variational_root_covariance),
        variational,
        (Real(updated_mean), LowerTriangular(updated_root_covariance)),
    )
    return variational, loss_value


def _expected_log_likelihood_derivatives(
    likelihood: AbstractLikelihood,
    response: Float[Array, "B 1"],
    mean: Float[Array, " B"],
    variance: Float[Array, " B"],
) -> tuple[Float[Array, " B"], Float[Array, " B"]]:
    r"""Return Bonnet's $\alpha$ and Price's $\beta$ for a batch.

    Args:
        likelihood: The observation model.
        response: The observed responses $\mathbf y_{\mathcal B}$, shaped $(B, 1)$.
        mean: The marginal means $m_i$ of $q(f_i)$.
        variance: The marginal variances $v_i$ of $q(f_i)$.

    Returns:
        The vectors $\boldsymbol\alpha$ and $\boldsymbol\beta$.

    Notes:
        $\alpha_i=\partial_{m_i}\mathbb E_{\mathcal N(m_i,v_i)}[\log p(y_i\mid f_i)]$
        and
        $\beta_i=-2\,\partial_{v_i}\mathbb E_{\mathcal N(m_i,v_i)}[\log p(y_i\mid f_i)]$.
        One gradient of ``expected_log_likelihood(y, mean, variance)`` suffices
        for closed-form and quadrature likelihoods; no second derivative is needed.
    """

    def total_expectation(mean_, variance_):
        return jnp.sum(
            likelihood.expected_log_likelihood(
                response, mean_[:, None], variance_[:, None]
            )
        )

    alpha, variance_gradient = jax.grad(total_expectation, argnums=(0, 1))(
        mean, variance
    )
    return alpha, -2.0 * variance_gradient


@natural_gradient_step.register(DualVariationalGaussian)
def _dual_variational_gaussian_step(
    variational: DualVariationalGaussian,
    hyper: DualVariationalGaussian,
    data: Dataset,
    objective: Objective,
    natgrad_lr: ScalarFloat | int,
    *,
    map_jitter: ScalarFloat | int = 0.0,
    backoff: ScalarFloat | int = 0.5,
    max_backoff: int = 5,
    beta_floor: ScalarFloat | int = 1e-8,
) -> tuple[DualVariationalGaussian, ScalarFloat]:
    r"""t-SVGP tied-site update for the dual parameterisation.

    Performs the convex combination
    $$\boldsymbol\lambda_1\leftarrow(1-\rho)\boldsymbol\lambda_1
    +\rho\tfrac NB\mathbf A_{\mathcal B}\mathbf g_1,\qquad
    \boldsymbol\Lambda_2\leftarrow(1-\rho)\boldsymbol\Lambda_2
    +\rho\tfrac NB\mathbf A_{\mathcal B}
    \operatorname{diag}(\mathbf g_2)\mathbf A_{\mathcal B}^\top,$$
    with $\mathbf A_{\mathcal B}=\mathbf K_{zz}^{-1}\mathbf K_{zb}$,
    $\mathbf g_1=\boldsymbol\alpha+\boldsymbol\beta\odot
    (\mathbf m_{\mathcal B}-\mu(\mathbf X_{\mathcal B}))$ and
    $\mathbf g_2=\boldsymbol\beta$.

    This matches Salimbeni at $\gamma=\rho$ when computed $\boldsymbol\beta\ge0$
    leaves ``beta_floor`` inert; otherwise the branches can diverge. Since
    $\nabla_{\boldsymbol\mu}\operatorname{KL}=\boldsymbol\lambda$, the KL is not
    differentiated. For $\rho\in[0,1]$
    and non-negative $\boldsymbol\beta$, the affine update stays in the PSD cone
    and needs no step-size backoff. Current-site factorisations of $\mathbf K_{zz}$
    and $\mathbf R$ still occur; :meth:`~gpjax.variational_families.DualVariationalGaussian._working_matrices`
    factorises $\mathbf R$ in a safe basis.

    Args:
        variational: The variational partition, holding the two dual sites.
        hyper: The hyperparameter partition.
        data: The batch at which the sites' target is evaluated.
        objective: The minimised loss, evaluated at pre-update sites.
        natgrad_lr: The step size $\rho\in(0,1]$.
        map_jitter: Unused.
        backoff: Unused.
        max_backoff: Unused.
        beta_floor: Lower clip on $\boldsymbol\beta$ before the matrix-site update;
            ensures a PSD target even when computed expected negative curvature is
            negative. In GPJax's clipped-probit Bernoulli this can happen in the
            far tails; the clip is inert when computed $\boldsymbol\beta\ge0$.

    Returns:
        The updated variational partition and the pre-update loss.
    """
    del map_jitter, backoff, max_backoff
    _reject_frozen_coordinates(variational)

    family = eqx.combine(variational, hyper)

    # Report the pre-update loss, consistently with fit and the Salimbeni branch.
    loss_value = objective(family, data)

    # The site target needs only chol(K_zz); marginals and the objective use R.
    _, root_gram = family._gram_and_root()
    mean, variance = family.marginals(data.X)

    alpha, beta = _expected_log_likelihood_derivatives(
        family.model.likelihood, data.y, mean, variance
    )
    # Clip beta rather than factorising and projecting the matrix site.
    beta = jnp.maximum(beta, beta_floor)

    # Centred sites subtract the prior mean for non-zero mean functions.
    prior_mean = family.model.prior.mean_function(data.X).squeeze(-1)
    natural_gradient_vector = alpha + beta * (mean - prior_mean)

    cross_covariance = family.model.prior.kernel.cross_covariance(
        family._fmt_inducing_inputs(), data.X
    )
    design = jsp.linalg.cho_solve((root_gram, True), cross_covariance)

    # Scale minibatch sites by N/B; the unscaled paper formula would converge
    # to B/N of the full-data target.
    scale = data.full_size / data.n
    target_vector = (design @ natural_gradient_vector)[:, None]
    target_matrix = _symmetrise(design @ (beta[:, None] * design.T))

    rate = natgrad_lr
    stored_vector = val(family.dual_vector)
    stored_matrix = val(family.dual_matrix)

    # Preserve the stored dtype across steps despite K_zz jitter promotion.
    updated_vector = (
        (1.0 - rate) * stored_vector + rate * (scale * target_vector)
    ).astype(stored_vector.dtype)
    # Symmetrise to avoid backend-dependent Cholesky behavior from roundoff.
    updated_matrix = _symmetrise(
        (1.0 - rate) * stored_matrix + rate * (scale * target_matrix)
    ).astype(stored_matrix.dtype)

    variational = eqx.tree_at(
        lambda tree: (tree.dual_vector, tree.dual_matrix),
        variational,
        (Real(updated_vector), Real(updated_matrix)),
    )
    return variational, loss_value


__all__ = [
    "expectation_from_moments",
    "moments_from_expectation",
    "moments_from_natural",
    "natural_from_moments",
    "natural_gradient_step",
    "partition_variational",
    "variational_coordinates",
]
