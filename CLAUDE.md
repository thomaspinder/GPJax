# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is GPJax?

GPJax is a Gaussian process library built on JAX. The API mirrors GP mathematics: you compose a `Prior` (kernel + mean function), multiply by a `Likelihood` to get a `Posterior`, then optimise an objective. All modules are Equinox `eqx.Module` subclasses, making them JAX-pytree-compatible for `jit`, `vmap`, and `grad`.

## Commands

All commands must be prefixed with `uv run`:

```bash
uv run poe test          # pytest with xdist (8 workers) and beartype enforcement
uv run poe lint          # ruff format check + ruff check --fix
uv run poe format        # ruff import sorting + formatting (mutates files)
uv run poe docstrings    # xdoctest on gpjax/
uv run poe all-tests     # lint + docstrings + test (CI gate)
uv run poe docs          # sphinx-build: executes + caches the example notebooks
uv run poe docs-ci       # smoke render, warnings-as-errors (the CI docs gate)
uv run poe docs-serve    # build, then serve docs/_build/html at localhost:8000
uv run poe docs-linkcheck # sphinx linkcheck, no notebook execution
uv run poe docs-clean    # remove built docs (keeps the notebook exec cache)
```

Run a single test file or test:
```bash
uv run pytest tests/test_kernels/test_stationary.py -v
uv run pytest tests/test_gps.py::test_conjugate_posterior -v
```

Install dev environment: `uv venv && uv sync`

## Architecture

### Core pipeline

```
Prior(kernel, mean_function)  *  Likelihood  -->  Posterior
         |                                            |
    prior(Xtest)                          posterior(Xtest, train_data)
         |                                            |
   GaussianDistribution                      GaussianDistribution
```

`Prior.__mul__(likelihood)` dispatches via `construct_posterior()` to return the correct posterior type:
- `Gaussian` likelihood -> `ConjugatePosterior` (closed-form inference)
- `Bernoulli`/`Poisson` -> `NonConjugatePosterior` (latent function values optimised)
- `HeteroscedasticGaussian` -> `HeteroscedasticPosterior` or `ChainedPosterior`

### Parameter system (`gpjax/parameters.py`)

Parameters are `paramax.AbstractUnwrappable` subclasses. Each class stores its value in an unconstrained internal field and implements `unwrap()` to apply the constraining bijection:

| Class | Bijection | Internal storage |
|---|---|---|
| `Real` | Identity | `value` (unchanged) |
| `PositiveReal` | Softplus | `_unconstrained` via `inv_softplus` |
| `NonNegativeReal` | Softplus | `_unconstrained` via `inv_softplus` |
| `SigmoidBounded` | Sigmoid scaled to `[low, high]` | `_unconstrained` via `logit` |
| `LowerTriangular` | Fill-triangular | `_flat` vector |

**Models are always held in wrapped form.** Wrapping happens once, at construction; nothing unwraps a model tree, including `fit`, which returns a wrapped model. The rule is:

> Call `val(param)` (from `gpjax.parameters`) wherever a parameter meets arithmetic. Pass models around untouched everywhere else.

`val` is idempotent and recursive: it returns the constrained value of a parameter (resolving nested wrappers such as `paramax.non_trainable`) and returns plain arrays or floats unchanged, so it never needs an `isinstance` guard. When writing a new kernel, mean function, likelihood, or objective, every parameter read needs a `val()`.

Forgetting `val()` is not silent — a parameter is a pytree node, not an array, so arithmetic raises `TypeError: unsupported operand type(s) ... 'PositiveReal'` immediately. The one quiet exception is `jax.tree_util.tree_map` over a model, which reaches the *unconstrained* leaf; that is what makes optimisation work, but it means tree-mapping arithmetic over a model is not the same as scaling its parameters.

Do not add `paramax.unwrap(model)` to a loss, objective, or prediction function — that is the pre-consolidation pattern, and it reintroduces an undocumented "must be unwrapped first" precondition on user-facing extension points. `paramax.unwrap` remains correct for generic tree introspection (see `gpjax/summary.py`). To freeze parameters, wrap them with `paramax.non_trainable(param)`.

### Kernel system (`gpjax/kernels/`)

`AbstractKernel` defines `__call__(x, y) -> scalar`, `gram(x) -> LinearOperator`, `cross_covariance(x, y) -> array`, and `diagonal(x) -> LinearOperator`. Kernels compose with `+` (SumKernel) and `*` (ProductKernel). Each kernel delegates matrix computation to a `compute_engine` (typically `DenseKernelComputation`).

Kernel categories: `stationary/` (RBF, Matern12/32/52, Periodic, etc.), `nonstationary/` (Linear, Polynomial, ArcCosine), `non_euclidean/` (GraphKernel), `multioutput/` (ICMKernel, LCMKernel), `approximations/` (RFF), `additive/` (OrthogonalAdditiveKernel).

### Linear algebra (`gpjax/linalg/`)

Built on [Lineax](https://docs.kidger.site/lineax/). Kernel `gram()` returns `lx.AbstractLinearOperator` (typically `lx.MatrixLinearOperator`). Use `.as_matrix()` to materialise. Custom operators: `BlockDiag`, `Kronecker`. Key utilities: `cholesky_factor()` (singledispatch, returns lower-triangular operator), `logdet()`, `add_jitter()`. Linear solves use `lx.linear_solve()`.

### Objectives (`gpjax/objectives.py`)

Functions `(model, Dataset) -> scalar`:
- `conjugate_mll` / `conjugate_loocv` -- for `ConjugatePosterior`
- `log_posterior_density` (alias `non_conjugate_mll`) -- for `NonConjugatePosterior`
- `elbo` / `collapsed_elbo` -- for variational families
- `dual_elbo` -- for `DualVariationalGaussian` (t-SVGP; same value as `elbo`, different hyperparameter gradient)
- `heteroscedastic_elbo` -- for heteroscedastic models

Optimise by negating: `nmll = lambda p, d: -conjugate_mll(p, d)`

### Fitting (`gpjax/fit.py`)

Four optimisers: `fit()` (Optax gradient descent with scan), `fit_scipy()` (SciPy L-BFGS-B), `fit_lbfgs()` (Optax L-BFGS with `while_loop`), `fit_natgrads()` (natural-gradient steps on a variational family, alternated with Optax steps on the hyperparameters). All handle the constrained/unconstrained bijection automatically, without converting the model: `eqx.partition`/`eqx.combine` with `eqx.is_array` manage trainable vs static parts, gradients are taken with respect to the internal unconstrained arrays, and the objective receives — and the optimiser returns — a wrapped model. The bijection is applied only at the `val()` call sites inside kernels, mean functions, and likelihoods.

### Variational inference (`gpjax/variational_families.py`)

`VariationalGaussian`, `WhitenedVariationalGaussian`, `DualVariationalGaussian`, `CollapsedVariationalGaussian`, `GraphVariationalGaussian`, `HeteroscedasticVariationalFamily`. All inherit from `AbstractVariationalFamily` and implement `predict()` + `prior_kl()`.

### NumPyro integration (`gpjax/numpyro_extras.py`)

Helpers for registering GPJax `Parameter` priors as NumPyro sample sites, enabling MCMC inference over hyperparameters.

### Dataset (`gpjax/dataset.py`)

`Dataset` is a `@dataclass(slots=True)` registered as a JAX pytree. Requires 2D arrays: `X` shape `(N, D)`, `y` shape `(N, Q)`. Warns if inputs are not float64.

## Testing notes

- `conftest.py` enables `jax_enable_x64` and installs the beartype import hook over `gpjax`
- pytest config treats warnings as errors (`filterwarnings = ["error", "ignore::DeprecationWarning"]`)
- Tests run with 8 xdist workers by default
- Hypothesis is configured with `deadline=None, max_examples=20`

## Code style

- Ruff with 88-char line limit, google docstring convention
- `F722` suppressed (jaxtyping string annotations like `"N D"`)
- Unicode math identifiers allowed in docstrings (`RUF002`/`RUF003` suppressed)
- Imports: `isort` via ruff with combined-as-imports and force-sort-within-sections
- Tests: `pytest` with `pytest-mock` for mocking and `hypothesis` for property-based testing. Prefer functions over classes for test organization.

## Examples

Stored in `docs/examples/` as `py:percent` format (jupytext), executed at docs
build time by MyST-NB (`nb_custom_formats` in `docs/conf.py`). Convert with:
```bash
jupytext --to notebook example.py   # .py -> .ipynb
jupytext --to py:percent example.ipynb  # .ipynb -> .py
```
