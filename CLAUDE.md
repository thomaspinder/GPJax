# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is GPJax?

GPJax is a Gaussian process library built on JAX. The API mirrors GP mathematics: you compose a `Prior` (kernel + mean function), multiply by a `Likelihood` to get a `Posterior`, then optimise an objective. All modules are Flax NNX `nnx.Module` subclasses, making them JAX-pytree-compatible for `jit`, `vmap`, and `grad`.

## Commands

All commands must be prefixed with `uv run`:

```bash
uv run poe test          # pytest with xdist (8 workers) and beartype enforcement
uv run poe lint          # ruff format check + ruff check --fix
uv run poe format        # ruff import sorting + formatting (mutates files)
uv run poe docstrings    # xdoctest on gpjax/
uv run poe all-tests     # lint + docstrings + test (CI gate)
uv run poe docs-build    # execute example notebooks then mkdocs build
uv run poe docs-serve    # local docs server at localhost:8000
```

Run a single test file or test:
```bash
uv run pytest tests/test_kernels/test_stationary.py -v
uv run pytest tests/test_gps.py::test_conjugate_posterior -v
```

Install dev environment: `uv venv && uv sync --extra dev`

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

Parameters are `nnx.Variable` subclasses with a `tag` string that selects the bijection for unconstrained optimisation:

| Class | Tag | Bijection |
|---|---|---|
| `Real` | `"real"` | Identity |
| `PositiveReal` | `"positive"` | Softplus |
| `NonNegativeReal` | `"non_negative"` | Softplus |
| `SigmoidBounded` | `"sigmoid"` | Sigmoid |
| `LowerTriangular` | `"lower_triangular"` | FillTriangular |

Access the raw value with `param[...]` (the `__getitem__` ellipsis convention from Flax NNX). The `transform()` function maps entire `nnx.State` trees between constrained and unconstrained spaces using `DEFAULT_BIJECTION`.

**Flax NNX Variable metadata**: `Variable` uses `__slots__`, so metadata must be set via `self.set_metadata(key=value)` and read via `self.get_metadata("key", default)`.

### Kernel system (`gpjax/kernels/`)

`AbstractKernel` defines `__call__(x, y) -> scalar`, `gram(x) -> LinearOperator`, `cross_covariance(x, y) -> array`, and `diagonal(x) -> LinearOperator`. Kernels compose with `+` (SumKernel) and `*` (ProductKernel). Each kernel delegates matrix computation to a `compute_engine` (typically `DenseKernelComputation`).

Kernel categories: `stationary/` (RBF, Matern12/32/52, Periodic, etc.), `nonstationary/` (Linear, Polynomial, ArcCosine), `non_euclidean/` (GraphKernel), `multioutput/` (ICMKernel, LCMKernel), `approximations/` (RFF), `additive/` (OrthogonalAdditiveKernel).

### Linear algebra (`gpjax/linalg/`)

Custom `LinearOperator` hierarchy: `Dense`, `Diagonal`, `Triangular`, `Identity`, `BlockDiag`, `Kronecker`. The `psd()` wrapper marks an operator as PSD. Key operations: `lower_cholesky()`, `solve()`, `logdet()`, `diag()`. These wrap JAX linalg but provide custom gradient-friendly implementations.

### Objectives (`gpjax/objectives.py`)

Functions `(model, Dataset) -> scalar`:
- `conjugate_mll` / `conjugate_loocv` -- for `ConjugatePosterior`
- `log_posterior_density` (alias `non_conjugate_mll`) -- for `NonConjugatePosterior`
- `elbo` / `collapsed_elbo` -- for variational families
- `heteroscedastic_elbo` -- for heteroscedastic models

Optimise by negating: `nmll = lambda p, d: -conjugate_mll(p, d)`

### Fitting (`gpjax/fit.py`)

Three optimisers: `fit()` (Optax gradient descent with scan), `fit_scipy()` (SciPy L-BFGS-B), `fit_lbfgs()` (Optax L-BFGS with `while_loop`). All handle the constrained/unconstrained bijection automatically via `nnx.split`/`nnx.merge`.

### Variational inference (`gpjax/variational_families.py`)

`VariationalGaussian`, `WhitenedVariationalGaussian`, `NaturalVariationalGaussian`, `ExpectationVariationalGaussian`, `CollapsedVariationalGaussian`, `GraphVariationalGaussian`, `HeteroscedasticVariationalFamily`. All inherit from `AbstractVariationalFamily` and implement `predict()` + `prior_kl()`.

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

- Ruff with 88-char line limit, numpy docstring convention
- `F722` suppressed (jaxtyping string annotations like `"N D"`)
- Unicode math identifiers allowed in docstrings (`RUF002`/`RUF003` suppressed)
- Imports: `isort` via ruff with combined-as-imports and force-sort-within-sections
- Tests: `pytest` with `pytest-mock` for mocking and `hypothesis` for property-based testing. Prefer functions over classes for test organization.

## Examples

Stored in `examples/` as `py:percent` format (jupytext). Convert with:
```bash
jupytext --to notebook example.py   # .py -> .ipynb
jupytext --to py:percent example.ipynb  # .ipynb -> .py
```
