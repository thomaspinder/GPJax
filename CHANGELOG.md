# Changelog

All notable changes to GPJax are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.17.0] — 2026-07-04

### Fixed

- **`Prior.sample_approx` / `ConjugatePosterior.sample_approx`**: split the PRNG
  key before drawing RFF frequencies, Fourier weights, and (posterior) noise.
  They previously shared one key, making the pathwise-sample cross-covariance
  biased (Wilson et al. 2020 §3 requires independence). Marginals were
  unaffected.
- **`kernels.approximations.RFF`**: multi-dimensional Matérn frequencies are now
  drawn from a `MultivariateStudentT` (shared inverse-χ² per row), matching the
  isotropic Matérn spectral density. Previously iid-per-dim `StudentT` draws
  approximated a *tensor-product* Matérn in `d > 1`. `d == 1` is unchanged.
- **`objectives.conjugate_loocv`**: now routes noise and targets through the
  likelihood protocol, so multi-output (`MultiOutputGaussian`) LOOCV is correct
  (leave-one-scalar-out on the flattened NP system). Also removed a redundant
  `jnp.linalg.inv` (audit #662).
- **`kernels.nonstationary.ArcCosine`**: `weight_variance` and `bias_variance`
  are now wrapped in `NonNegativeReal`. They were previously assigned raw —
  silently frozen (Python-float default, excluded by `eqx.partition`) or
  trainable-but-unconstrained (array value, driveable negative → NaN).
- **`models.create_oilmm_from_data`**: now performs the documented PCA
  eigen-initialisation of the mixing matrix (top-M eigenvectors/eigenvalues of
  the empirical output covariance) instead of returning the random default. Raises
  `ValueError` for `N < 2` (empirical covariance is undefined).

### Changed

- Pathwise samples from `sample_approx` with a fixed `key` now differ from
  `0.16.0` (the old draws were biased — see Fixed above). No restore path; the
  old outputs were incorrect.
- Multi-dimensional Matérn `RFF` Gram matrices now converge to the isotropic
  Matérn. `d > 1` outputs differ from `0.16.0`. No restore path.
- Multi-output `conjugate_loocv` values differ from `0.16.0`. No restore path.
- **`models.OILMMPosterior.predict(..., return_full_cov=False)`** now returns a
  `lineax.DiagonalLinearOperator` scale (was a densified `MatrixLinearOperator`).
  Values are unchanged.

#### Migration

- **ArcCosine**: if you relied on the accidental freeze of `weight_variance` /
  `bias_variance`, restore it explicitly with `paramax.non_trainable(...)`.
- **`sample_approx` / multi-dim Matérn RFF / multi-output LOOCV**: no restore
  path — the previous outputs were incorrect.
- **OILMM diagonal predict**: `predict(..., return_full_cov=False).scale` is now a
  `DiagonalLinearOperator`. Use `.as_matrix()` / `.diagonal` rather than assuming a
  dense matrix.

## [0.16.0] — 2026-06-28

### Added

- **`gpjax.summarise(model)`**: render any GPJax pytree (kernel, prior,
  posterior, likelihood, variational family, ...) as a flat `rich` table —
  one row per parameter — showing parameter path, class, constrained value,
  bijector, trainability, shape, and dtype. User-facing abstract bases gain
  `__rich__` and `_repr_mimebundle_` hooks for `rich.print(model)` and
  Jupyter auto-rendering; `repr()` is unchanged. `rich` is now a core
  runtime dependency. An optional `priors` mapping (parameter name → prior,
  e.g. a NumPyro distribution) populates the otherwise-empty Prior column.

## [0.15.0] — 2026-06-03

### Added

- **`gpjax.state_space` sub-package**: state-space (Markovian) Gaussian
  processes with a square-root Kalman filter (chunked checkpointed scan
  for `O(sqrt(N) · d²)` reverse-mode AD memory) and an RTS smoother.
  Public API:
  - `StateSpacePrior`, `StateSpaceConjugatePosterior`
  - `TruncatedPeriodic` (Solin & Särkkä 2014 truncated-Fourier kernel)
  - `to_sde` singledispatch dispatching kernels to closed-form SDEs
  - `state_space_mll` MLL objective
  - `fit`, `fit_scipy`, `fit_lbfgs` thin wrappers around the
    corresponding `gpjax.fit_*` optimisers
  - Supports `Matern12`, `Matern32`, `Matern52`, `TruncatedPeriodic`,
    and their `SumKernel` compositions on 1-D temporal data.
- **New poe task**: `all-tests-slow` runs long-running numerical and
  memory stress tests opt-in via the `slow` pytest marker.

### Changed (breaking)

- **`gpjax.likelihoods.Gaussian.predict` and
  `gpjax.likelihoods.HeteroscedasticGaussian.predict`** now return
  `gpjax.distributions.GaussianDistribution` (was
  `numpyro.distributions.MultivariateNormal`). For `Gaussian.predict`, the
  diagonal fast path preserves a `lineax.DiagonalLinearOperator` scale; the
  dense path wraps a `lineax.MatrixLinearOperator`.
  `HeteroscedasticGaussian.predict` wraps its (always-dense) covariance in a
  `lineax.MatrixLinearOperator`. Callers relying on
  `MultivariateNormal`-specific attributes (`scale_tril`,
  `precision_matrix`, etc.) must update; callers using `mean`,
  `variance`, `covariance_matrix` are unaffected.

## [0.14.0] — 2026-04-21

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

[0.16.0]: https://github.com/thomaspinder/GPJax/releases/tag/v0.16.0
[0.17.0]: https://github.com/thomaspinder/GPJax/releases/tag/v0.17.0
[0.15.0]: https://github.com/thomaspinder/GPJax/releases/tag/v0.15.0
[0.14.0]: https://github.com/thomaspinder/GPJax/releases/tag/v0.14.0
