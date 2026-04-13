# Changelog

All notable changes to GPJax are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.14.0rc1] — 2026-04-13

See [`docs/migration.md`](docs/migration.md) for full upgrade instructions.

### Changed (breaking)

- **Module backend**: `flax.nnx.Module` has been replaced with
  `equinox.Module`. Custom kernels, mean functions, likelihoods, and
  variational families that subclassed `nnx.Module` must now subclass
  `eqx.Module` and declare class-level type annotations for every field.
- **Parameter system**: `PositiveReal`, `NonNegativeReal`, `Real`,
  `SigmoidBounded`, and `LowerTriangular` now inherit from
  `paramax.AbstractUnwrappable`. Read a parameter value with
  `param.unwrap()`, or resolve an entire model tree with
  `paramax.unwrap(model)`. `param.value` no longer exists.
- **Fitting API**: `gpx.fit`, `gpx.fit_scipy`, and `gpx.fit_lbfgs` no
  longer accept `params_bijection=` or `trainable=`. Bijection handling
  is automatic; freeze parameters by wrapping them with
  `paramax.non_trainable` via `eqx.tree_at`.
- **`LowerTriangular`** now requires a strictly positive diagonal
  (parameterised by `softplus_lower_cholesky`). Passing a matrix with
  zero or negative diagonal entries produces `NaN` during construction.
- **Linear algebra**: `gpjax.linalg` has been rewritten on top of
  [Lineax](https://docs.kidger.site/lineax/). Kernel `gram()` returns a
  `lineax.AbstractLinearOperator` instead of a `cola.LinearOperator`.
  `PSD`, `psd`, `Dense`, `Diagonal`, `Identity`, `Triangular`,
  `LinearOperator`, `diag`, `solve`, and `lower_cholesky` are removed;
  see the migration guide for Lineax equivalents.
- **Bijectors**: the custom bijector stack has been removed.
  `Parameter` (the old generic base class), `DEFAULT_BIJECTION`,
  `transform(...)`, and `FillTriangularTransform` are gone. Custom
  parameter classes should use `numpyro.distributions.biject_to` with a
  `numpyro.distributions.constraints` object.
- **`register_parameters` removed**. With the Equinox backend, GPJax
  identifies parameter classes through `isinstance` checks on
  `AbstractUnwrappable`, so the decorator is no longer needed.

### Added

- `gpjax.linalg.cholesky_factor` — single-dispatch Cholesky factoriser
  that returns a lower-triangular Lineax operator.
- `gpjax.linalg.add_jitter` — helper for adding a jitter term to a
  covariance operator.
- [`docs/migration.md`](docs/migration.md) — upgrade guide from 0.13.x.

### Removed

- Flax is no longer a runtime dependency. The package `__description__`
  has been updated from "Gaussian processes in JAX and Flax" to
  "Gaussian processes in JAX".
- `cola-ml` is no longer a runtime dependency.

### Dependencies

- Added: `equinox>=0.11`, `paramax>=0.0.5`, `lineax`.
- Removed: `flax`, `cola-ml`.

[0.14.0rc1]: https://github.com/thomaspinder/GPJax/releases/tag/v0.14.0rc1
