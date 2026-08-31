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

This module implements the coordinate maps and the update rule of Salimbeni,
Eleftheriadis and Hensman (2018), *Natural Gradients in Practice: Non-Conjugate
Variational Inference in Gaussian Process Models* (arXiv:1803.09151).

A Gaussian $q(\mathbf u)=\mathcal N(\mathbf m,\mathbf S)$ over $M$ inducing outputs is
an exponential family with sufficient statistics
$\mathbf t(\mathbf u)=[\mathbf u,\ \mathbf u\mathbf u^\top]$. Three coordinate systems
are in play:

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

The Fisher information in the natural coordinates is exactly the Jacobian
$\partial\boldsymbol\eta/\partial\boldsymbol\theta$, so the natural gradient of a loss
$\ell$ is the *ordinary* gradient with respect to the expectation parameters,
$\tilde\nabla_{\boldsymbol\theta}\ell=\mathbf F^{-1}\partial\ell/\partial\boldsymbol\theta
=\partial\ell/\partial\boldsymbol\eta$. No Fisher matrix is ever formed. The step is

$$\boldsymbol\theta\leftarrow\boldsymbol\theta
-\gamma\,\partial\ell/\partial\boldsymbol\eta,$$

with $\gamma$ the natural-gradient step size. For $\gamma\in[0,1]$ this is a convex
combination in $\boldsymbol\theta$-space; for a conjugate model at $\gamma=1$ it lands
on the exact optimum in a single step.

The module also carries the dual (t-SVGP) update of Adam, Chang, Khan and Solin (2021),
*Dual Parameterization of Sparse Variational Gaussian Processes* (arXiv:2111.03412).
There the stored coordinates are the sites
$(\boldsymbol\lambda_1,\boldsymbol\Lambda_2)$ of
$\boldsymbol\eta=\boldsymbol\eta_0(\boldsymbol\theta)+\boldsymbol\lambda$, so the step
is affine in the stored parameters and needs no
$\boldsymbol\theta\leftrightarrow\boldsymbol\eta$ round trip. Since
$\nabla_{\boldsymbol\mu}\operatorname{KL}=\boldsymbol\lambda$ exactly, the KL is never
differentiated and the update reduces to the convex combination
$\boldsymbol\lambda\leftarrow(1-\rho)\boldsymbol\lambda
+\rho\,\nabla_{\boldsymbol\mu}\mathcal L_{\text{ell}}$ --
which is the Salimbeni step at $\gamma=\rho$, producing identical iterates -- provided
the computed $\boldsymbol\beta$ stays non-negative, so that ``beta_floor`` is inert.
That holds for a truly log-concave likelihood; GPJax's clipped probit link violates it
in the far tails, where the two branches then diverge.
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
        $\boldsymbol\eta_1=\mathbf m$,
        $\mathbf H_2=\mathbf L\mathbf L^\top+\mathbf m\mathbf m^\top$.
        No factorisation, no solve, no jitter; cost $\mathcal O(M^3)$ matmul.

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
        Two triangular solves:
        $\mathbf L^{-1}=\texttt{tril\_solve}(\mathbf L,\mathbf I)$,
        $\mathbf P=(\mathbf L^{-1})^\top\mathbf L^{-1}$,
        $\boldsymbol\theta_1=\mathbf P\mathbf m$,
        $\boldsymbol\Theta_2=-\tfrac12\mathbf P$.
        No jitter: $\mathbf L$ already has a strictly positive diagonal.

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
        map_jitter: Jitter $\varepsilon$ added to the diagonal before the Cholesky.
            Defaults to ``0.0``; this is deliberately *not* the model's
            ``Prior.jitter``, because a non-zero value biases $\mathbf S$ by
            $\approx\varepsilon\lVert\mathbf S\rVert^2$ rather than merely
            perturbing it.

    Returns:
        The moment parameters $(\mathbf m,\mathbf L)$.

    Notes:
        Jitter site #1:
        $\mathbf S=\operatorname{sym}(\mathbf H_2-\boldsymbol\eta_1\boldsymbol\eta_1^\top)$,
        $\mathbf L=\operatorname{chol}(\mathbf S+\varepsilon\mathbf I)$.
        The subtraction is a cancellation site when
        $\lVert\mathbf m\rVert^2\gg\lVert\mathbf S\rVert$; prefer the whitened
        family there. Unlike the natural route this map has no admissibility guard,
        so in the cancellation regime it degrades quietly: at
        $\lVert\mathbf m\rVert\sim10^4$ with $\mathbf S=10^{-8}\mathbf I$ the
        recovered $\mathbf L$ is finite but wrong by $\mathcal O(10^{-1})$ relative
        error before it eventually returns ``NaN``.

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
        natural_matrix: The second natural parameter $\boldsymbol\Theta_2$, required
            to be negative definite for the result to be finite.
        map_jitter: Jitter $\varepsilon$ added to the diagonal before each Cholesky.
            Defaults to ``0.0``; see :func:`moments_from_expectation` for why it is
            not inherited from the family.

    Returns:
        The moment parameters $(\mathbf m,\mathbf L)$.

    Notes:
        Route A (jitter site #2):
        $\mathbf P=\operatorname{sym}(-2\boldsymbol\Theta_2)$,
        $\mathbf L_P=\operatorname{chol}(\mathbf P+\varepsilon\mathbf I)$,
        $\mathbf X=\texttt{tril\_solve}(\mathbf L_P,\mathbf I)$,
        $\mathbf S=\operatorname{sym}(\mathbf X^\top\mathbf X)$,
        $\mathbf m=\texttt{triu\_solve}(\mathbf L_P^\top,
        \texttt{tril\_solve}(\mathbf L_P,\boldsymbol\theta_1))$,
        $\mathbf L=\operatorname{chol}(\mathbf S+\varepsilon\mathbf I)$.

        Route B (the reverse/anti-diagonal Cholesky) is *not* used: the two are
        numerically equivalent and route A needs no exchange matrix. Returns ``NaN``
        rather than raising when $\boldsymbol\Theta_2\not\prec0$ -- that is what
        makes the step-size backoff ``jit``-clean.

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

    This registration also covers ``WhitenedVariationalGaussian`` and
    ``GraphVariationalGaussian``, which subclass ``VariationalGaussian`` and store the
    same two fields. That is deliberate: the whitened $q(\mathbf v)$ belongs to the
    same exponential family, so the coordinate maps are identical. The graph family
    is usable end to end since ``variational_expectation`` switched from a
    per-point ``vmap`` to the conditioned posterior's diagonal path; the smoke
    test in ``tests/test_natural_gradients.py`` predates that and drives the
    graph family with ``prior_kl``.

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
        The variational partition (holding only the exponential-family coordinates)
        and the hyperparameter partition (holding everything else, including the
        inducing inputs, which Salimbeni et al. count as hyperparameters).

    Notes:
        Uses a **prefix** filter spec so the split is independent of the parameter
        wrapper's internal field names (``.value`` versus ``._flat``), composes with
        ``paramax.non_trainable``, and raises loudly (``AttributeError``) if a field
        is renamed.

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
        ``paramax.non_trainable`` wraps *leaves*, not whole nodes, so a frozen
        ``variational_mean`` is a ``Real`` whose ``value`` is a ``NonTrainable``. The
        ``is_leaf`` predicate stops the traversal at those wrappers so they are
        visible.
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
        A partial natural-gradient step on $\boldsymbol\theta$ is not meaningful.
        This is a Python-level ``isinstance`` check on a static tree node, executed
        at trace time, so it never branches on a traced value.

        The offending coordinates are located by re-walking the tree with the
        *selector* itself as the ``is_leaf`` predicate, rather than by matching the
        selected nodes against top-level dataclass fields. A future registration
        whose ``where`` picks a nested node --
        ``tree.signal_variational.variational_mean``, say -- is then still reported,
        with its full key path, instead of silently passing the guard.
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
        ``jnp.linalg.cholesky`` returns ``NaN`` rather than raising, so admissibility
        is a *value*: ``vmap`` over the $K+1$ trials, mask on ``jnp.isfinite`` and
        select with ``jnp.argmax``, which returns the first ``True`` and ``0`` when
        all are ``False`` so that ``NaN`` propagates rather than silently returning a
        wrong answer.

        Only the *probe* is replicated, not the whole
        $\boldsymbol\theta\to\boldsymbol\xi$ map. Admissibility of a trial is decided
        by $\operatorname{chol}(\mathbf P+\varepsilon\mathbf I)$ and the two
        triangular solves for $\mathbf m$; the $\mathcal O(M^3)$ inversion, the
        $\mathbf X^\top\mathbf X$ product and the second Cholesky of
        :func:`moments_from_natural` cannot turn an admissible
        $\boldsymbol\Theta_2$ inadmissible, so they run once, at the accepted step
        size. Replicating them instead measured 13% of total training wall clock at
        $M=200$, which is not the negligible cost the plan assumed.

        The trial ladder is cast to the dtype of $\boldsymbol\Theta_2$: under
        ``jax_enable_x64`` the exponent ``jnp.arange(K + 1)`` is ``int64``, so
        $\beta^{k}$ would otherwise be a non-weak ``float64`` that silently promotes
        a ``float32`` model and breaks the ``lax.scan`` carry.
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
        The updated **variational partition** and the loss evaluated at the
        *pre-update* coordinates, so that ``history[t]`` matches ``fit()``'s
        convention. The reported loss is evaluated at
        $\boldsymbol\xi(\boldsymbol\eta_t)$, so a non-zero ``map_jitter`` biases it by
        $\mathcal O(\varepsilon)$; at the default of ``0.0`` it is exactly
        $\ell(\boldsymbol\xi_t,\boldsymbol\phi_t)$.

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

    Performs $\boldsymbol\theta\leftarrow\boldsymbol\theta
    -\gamma\,\partial\ell/\partial\boldsymbol\eta$ and writes the resulting moment
    parameters back into the variational partition. Also covers
    ``WhitenedVariationalGaussian`` and ``GraphVariationalGaussian``: the whitened
    $q(\mathbf v)$ is a member of the same exponential family, and the whitening
    enters only through ``prior_kl``/``condition``, which the loss calls
    polymorphically.

    ``beta_floor`` is accepted for a uniform dispatch contract and ignored here.

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
        The updated variational partition and the pre-update loss. That loss is read
        off the differentiated closure, hence evaluated at
        $\boldsymbol\xi(\boldsymbol\eta_t)$ rather than at the stored $\mathbf L$: with
        ``map_jitter=0.0`` the two agree exactly, and a non-zero ``map_jitter`` shifts
        the reported value by $\mathcal O(\varepsilon)$.
    """
    del beta_floor
    _reject_frozen_coordinates(variational)

    family = eqx.combine(variational, hyper)
    initial_mean = val(family.variational_mean)
    initial_root_covariance = val(family.variational_root_covariance)

    # theta_0 comes from L directly, never by round-tripping through eta_0: the
    # detour costs an extra Cholesky and passes through a cancellation site.
    initial_natural = natural_from_moments(initial_mean, initial_root_covariance)
    initial_expectation = expectation_from_moments(
        initial_mean, initial_root_covariance
    )

    def loss_of_expectation(expectation):
        # The direct map xi(eta) is used rather than xi(theta(eta)): one Cholesky
        # instead of three, with identical gradients. The LowerTriangular round trip
        # is softplus o softplus_inv = identity and lets us call GPJax's own
        # objectives unmodified.
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
        Bonnet's and Price's theorems give
        $\alpha_i=\partial_{m_i}\mathbb E_{\mathcal N(m_i,v_i)}[\log p(y_i\mid f_i)]$
        and
        $\beta_i=-2\,\partial_{v_i}\mathbb E_{\mathcal N(m_i,v_i)}[\log p(y_i\mid f_i)]$,
        so one ``jax.grad`` of the likelihood's existing ``expected_log_likelihood``
        suffices -- **no second derivatives of the likelihood are needed**, and the
        routine works for closed-form and quadrature likelihoods alike. Note that
        GPJax's argument order is ``(y, mean, variance)``, with the response first.
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

    Because $\nabla_{\boldsymbol\mu}\operatorname{KL}=\boldsymbol\lambda$ exactly, this
    *is* the Salimbeni step at $\gamma=\rho$: started from the same $q$ the two
    branches produce identical iterates, and the KL is never differentiated. The
    identity holds provided the computed $\boldsymbol\beta$ stays non-negative, so that
    ``beta_floor`` never engages -- true for a genuinely log-concave likelihood, and
    violated by GPJax's clipped probit link in the far tails.

    ``map_jitter``, ``backoff`` and ``max_backoff`` are accepted for a uniform dispatch
    contract and ignored. The update is affine and, for $\rho\in[0,1]$ and
    $\boldsymbol\beta\ge0$, never leaves the positive semi-definite cone, so the
    Salimbeni branch's backoff -- which exists to rescue
    $\operatorname{chol}(\mathbf S)$ after an overshoot in $\boldsymbol\theta$ -- has
    nothing to guard here. The step does still factorise $\mathbf K_{zz}$ and, inside
    the objective and ``marginals``, $\mathbf R$; those are properties of the *current*
    sites rather than of the step, and
    :meth:`~gpjax.variational_families.DualVariationalGaussian._working_matrices`
    factorises $\mathbf R$ in a basis where it cannot fail.

    Args:
        variational: The variational partition, holding the two dual sites.
        hyper: The hyperparameter partition.
        data: The batch at which the sites' target is evaluated.
        objective: The loss being minimised. Evaluated once at the pre-update sites
            so that ``history[t]`` means the same thing in both dispatch branches.
        natgrad_lr: The step size $\rho\in(0,1]$.
        map_jitter: Unused.
        backoff: Unused.
        max_backoff: Unused.
        beta_floor: Lower clip applied to $\boldsymbol\beta$ before it enters
            $\boldsymbol\Lambda_2$. A no-op for likelihoods that are log-concave *as
            computed*; it keeps the update inside the PSD cone for those (Student-t,
            some heteroscedastic models) whose expected negative curvature can go
            negative. GPJax's Bernoulli is in the latter group in the far tails:
            ``inv_probit`` clips its output away from $0$ and $1$, which flattens
            $\log p$ and makes $\beta_i<0$ for a confidently mislabelled point, so
            the clip does engage there.

    Returns:
        The updated variational partition and the pre-update loss.
    """
    del map_jitter, backoff, max_backoff
    _reject_frozen_coordinates(variational)

    family = eqx.combine(variational, hyper)

    # One extra forward pass, taken deliberately: it makes `history[t]` the loss at the
    # pre-update parameters, exactly as in `fit` and in the Salimbeni branch. XLA
    # commonly common-subexpression-eliminates it against the `marginals` call below.
    loss_value = objective(family, data)

    # Only the Cholesky of K_zz is needed for the target, which is built from
    # A = K_zz^{-1} K_zb; `_gram_and_root` therefore stops short of factorising R.
    # `marginals` below does factorise it, and so does `objective` above; under `jit`
    # -- which is how `fit_natgrads` always runs -- XLA folds the repeats down to one
    # chol(K_zz) and one chol(R) for the whole step.
    _, root_gram = family._gram_and_root()
    mean, variance = family.marginals(data.X)

    alpha, beta = _expected_log_likelihood_derivatives(
        family.model.likelihood, data.y, mean, variance
    )
    # Clip beta, never Lambda_2: `jnp.maximum` is trace-safe, whereas jittering or
    # projecting Lambda_2 would need a factorisation it never otherwise requires.
    beta = jnp.maximum(beta, beta_floor)

    # Centred sites (Adam et al. section on non-zero mean functions). The reference
    # implementation shifts by `predict_f(Z)`, which already includes the mean
    # function, and is therefore wrong for any non-zero mean function.
    prior_mean = family.model.prior.mean_function(data.X).squeeze(-1)
    natural_gradient_vector = alpha + beta * (mean - prior_mean)

    cross_covariance = family.model.prior.kernel.cross_covariance(
        family._fmt_inducing_inputs(), data.X
    )
    design = jsp.linalg.cho_solve((root_gram, True), cross_covariance)

    # N / B. The paper prints the mini-batch update with no such factor; taken
    # literally the sites converge to B/N of their correct value. `get_batch` stamps
    # the full-dataset size onto each minibatch as `Dataset.n_total`, which
    # `Dataset.full_size` reads back (falling through to `n` for a whole dataset).
    scale = data.full_size / data.n
    target_vector = (design @ natural_gradient_vector)[:, None]
    target_matrix = _symmetrise(design @ (beta[:, None] * design.T))

    rate = natgrad_lr
    stored_vector = val(family.dual_vector)
    stored_matrix = val(family.dual_matrix)

    # `add_jitter` builds its identity at the default float type, so under
    # `jax_enable_x64` everything downstream of K_zz is float64 even for a float32
    # model. Cast back, or the `lax.scan` carry changes dtype between iterations.
    updated_vector = (
        (1.0 - rate) * stored_vector + rate * (scale * target_vector)
    ).astype(stored_vector.dtype)
    # Symmetrise after the update: it costs nothing and removes the O(eps) asymmetry
    # that would otherwise make `cholesky(R)` backend-nondeterministic.
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
