# Glossary

The vocabulary GPJax's documentation assumes. Every term here is linkable from
anywhere in the docs with `` {term}`jitter` ``, so the examples can use the words
without stopping to redefine them.

<!-- `{.glossary}` is an `attrs_block` attribute (enabled in conf.py). It turns the
     plain markdown definition list below into a real Sphinx glossary, which is what
     makes `{term}` resolve. Writing it as a `{glossary}` directive would work too,
     but then the source stops being readable markdown. Keep entries alphabetical. -->

{.glossary}
ARD
: Automatic relevance determination. A kernel with one {term}`lengthscale` per
  input dimension rather than a single shared one. Dimensions that the data says
  are uninformative are driven to large lengthscales, effectively switching them
  off. In GPJax any stationary kernel becomes ARD by passing a vector
  `lengthscale` of length `n_dims`.

Cholesky factor
: The unique lower-triangular $\mathbf{L}$ with $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top$
  for a symmetric positive-definite $\mathbf{\Sigma}$. Computing it costs
  $\sim n^3/3$ flops — half an LU factorisation — after which each solve is only
  two triangular substitutions at $\mathcal{O}(n^2)$ apiece. That asymmetry, a
  factorisation paid once against solves that are cheap thereafter, is why
  essentially every GP computation routes through it. See
  [`cholesky_factor`](#gpjax.linalg.cholesky_factor) and
  [the sharp bits](sharp_bits.md#the-cholesky-drawback).

conjugate
: A prior and likelihood pair for which the posterior has the same form as the
  prior, so it is available in closed form. For Gaussian processes this means a
  Gaussian likelihood, giving a [`ConjugateModel`](#gpjax.gps.ConjugateModel)
  whose [`conjugate_mll`](#gpjax.objectives.conjugate_mll) needs no approximation.
  Anything else — Bernoulli, Poisson — is non-conjugate, and the latent function
  values must be approximated: by MAP or a Laplace approximation, by
  {term}`variational inference`, or by MCMC.

ELBO
: Evidence lower bound. A tractable lower bound on the marginal log-likelihood,
  maximised in place of it when the latter is unavailable. GPJax provides
  [`elbo`](#gpjax.objectives.elbo) for the uncollapsed (mini-batchable) bound and
  [`collapsed_elbo`](#gpjax.objectives.collapsed_elbo) for the collapsed bound,
  which solves the variational parameters analytically but requires a
  {term}`conjugate` likelihood and a full pass over the data.

Gram matrix
: The matrix $\mathbf{K}_{\boldsymbol{xx}}$ of kernel evaluations between every
  pair of inputs, $[\mathbf{K}_{\boldsymbol{xx}}]_{ij} = k(x_i, x_j)$. Symmetric
  and positive semi-definite by construction. Produced by a kernel's `gram()`
  method, which returns a Lineax
  [`AbstractLinearOperator`](inv:lineax#lineax.AbstractLinearOperator) rather than
  a dense array, so structure can be exploited.

inducing points
: A set of $m \ll n$ pseudo-inputs $\boldsymbol{z}$ that summarise the training
  data, reducing inference from $\mathcal{O}(n^3)$ to $\mathcal{O}(nm^2)$. They
  are ordinary model parameters and are optimised alongside the kernel
  hyperparameters. Also called pseudo-points. See
  [the sparse regression notebook](examples/collapsed_vi.py).

jitter
: A small constant — $10^{-6}$ by default — added to the diagonal of a
  {term}`Gram matrix` before factorisation. Kernel matrices are positive definite
  mathematically but can pick up tiny negative eigenvalues in floating point when
  inputs are close together, which makes the {term}`Cholesky factor` fail. Applied
  by [`add_jitter`](#gpjax.linalg.add_jitter).

lengthscale
: The kernel hyperparameter $\ell$ controlling how far apart two inputs must be
  before their function values decorrelate. Small lengthscales give wiggly
  functions, large ones give smooth functions. Strictly positive, so GPJax stores
  it as a [`PositiveReal`](#gpjax.parameters.PositiveReal).

marginal log-likelihood
: $\log p(\boldsymbol{y})$ with the latent function integrated out — the standard
  objective for learning GP hyperparameters. Available in closed form only in the
  {term}`conjugate` case, where it is
  [`conjugate_mll`](#gpjax.objectives.conjugate_mll). Optimised by *maximising*
  it, so GPJax's fitting routines are handed its negation.

natural parameters
: The parameterisation $(\mathbf{\Sigma}^{-1}\boldsymbol{\mu}, -\tfrac{1}{2}\mathbf{\Sigma}^{-1})$
  of a Gaussian, as opposed to the moment parameterisation $(\boldsymbol{\mu}, \mathbf{\Sigma})$.
  Their use is what makes natural gradients cheap: the natural gradient with
  respect to the natural parameters is exactly the ordinary gradient with respect
  to the *expectation* parameters, so following the information geometry of the
  distribution costs no Fisher-matrix inversion. Convergence is typically far
  faster than plain gradient descent on the moments. GPJax computes both
  coordinate systems on the fly inside
  [`fit_natgrads`](#gpjax.fit.fit_natgrads), which operates directly on
  [`VariationalGaussian`](#gpjax.variational_families.VariationalGaussian) and
  [`WhitenedVariationalGaussian`](#gpjax.variational_families.WhitenedVariationalGaussian).

variance
: The kernel hyperparameter $\sigma^2$ setting the marginal variance of the
  process — how far function values stray from the mean. Distinct from the
  *observation* noise variance carried by the likelihood, which is a common source
  of confusion when reading a fitted model's parameters.

variational inference
: Approximating an intractable posterior by finding the closest member of a
  tractable family, measured by KL divergence. Turns integration into
  optimisation, and is what makes non-{term}`conjugate` likelihoods and large
  datasets tractable. GPJax's families are listed under
  [variational families](reference/variational_families.md).

whitening
: A reparameterisation of the variational distribution in terms of
  $\boldsymbol{v}$ with $\boldsymbol{u} = \mathbf{L}\boldsymbol{v}$, where
  $\mathbf{L}$ is the {term}`Cholesky factor` of the prior covariance over the
  {term}`inducing points`. This decouples the variational parameters from the
  kernel hyperparameters and conditions the optimisation problem much better. See
  [`WhitenedVariationalGaussian`](#gpjax.variational_families.WhitenedVariationalGaussian).
