from typing import TypeVar

from flax import nnx
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Float
import numpyro.distributions as npd
import typing_extensions as tpe

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    ConjugatePosterior,
    NonConjugatePosterior,
)
from gpjax.likelihoods import (
    AbstractHeteroscedasticLikelihood,
)
from gpjax.linalg import (
    Dense,
    lower_cholesky,
    psd,
    solve,
)
from gpjax.linalg.utils import add_jitter
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from gpjax.variational_families import (
    AbstractVariationalFamily,
    HeteroscedasticVariationalFamily,
)

VF = TypeVar("VF", bound=AbstractVariationalFamily)
HVF = TypeVar("HVF", bound=HeteroscedasticVariationalFamily)


Objective = tpe.Callable[[nnx.Module, Dataset], ScalarFloat]


def conjugate_mll(posterior: ConjugatePosterior, data: Dataset) -> ScalarFloat:
    r"""Evaluate the marginal log-likelihood of the Gaussian process.

    Compute the marginal log-likelihood function of the Gaussian process.
    The returned function can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation
    given here enables exact estimation of the Gaussian process' latent
    function values.

    For a training dataset $\{x_n, y_n\}_{n=1}^N$, set of test inputs
    $\mathbf{x}^{\star}$ the corresponding latent function evaluations are given
    by $\mathbf{f}=f(\mathbf{x})$ and $\mathbf{f}^{\star}f(\mathbf{x}^{\star})$,
    the marginal log-likelihood is given by:

    ```math
    \begin{align}
        \log p(\mathbf{y}) & = \int p(\mathbf{y}\mid\mathbf{f})
        p(\mathbf{f}, \mathbf{f}^{\star})\mathrm{d}\mathbf{f}^{\star}\\
        & = 0.5\left(-\mathbf{y}^{\top}\left(k(\mathbf{x}, \mathbf{x}')
        + \sigma^2\mathbf{I}_N\right)^{-1}\mathbf{y} \right.\\
        & \quad\left. -\log\lvert k(\mathbf{x}, \mathbf{x}')
        + \sigma^2\mathbf{I}_N\rvert - n\log 2\pi \right).
    \end{align}
    ```

    Example:
        >>> import gpjax as gpx

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> gpx.objectives.conjugate_mll(posterior, D)

        Our goal is to maximise the marginal log-likelihood. Therefore, when optimising
        the model's parameters with respect to the parameters, we use the negative
        marginal log-likelihood. This can be realised through

        >>> nmll = lambda p, d: -gpx.objectives.conjugate_mll(p, d)

    Args:
        posterior (ConjugatePosterior): The posterior distribution for which
            we want to compute the marginal log-likelihood.
        data:: The training dataset used to compute the
            marginal log-likelihood.

    Returns
    -------
    ScalarFloat
        The marginal log-likelihood of the Gaussian process.
    """

    from gpjax.kernels.multioutput.base import MultiOutputKernel

    x, y = data.X, data.y
    kernel = posterior.prior.kernel
    mx = posterior.prior.mean_function(x)

    # Validation for multi-output models (user-facing error messages)
    if isinstance(kernel, MultiOutputKernel):
        if not data.multi_output:
            raise ValueError("MultiOutputKernel requires multi-output data.")
        if data.num_outputs != kernel.num_outputs:
            raise ValueError(
                f"Dataset has {data.num_outputs} outputs "
                f"but kernel expects {kernel.num_outputs}."
            )

    # Unified path — prepare_targets is identity for single-output,
    # output-major reshape for multi-output
    y_flat, mx_flat = posterior.likelihood.prepare_targets(y, mx)
    noise = posterior.likelihood.noise_vector(data.n)

    Kxx = kernel.gram(x)
    Kxx_dense = add_jitter(Kxx.to_dense(), posterior.prior.jitter)
    Sigma_dense = Kxx_dense + jnp.diag(noise)
    Sigma = psd(Dense(Sigma_dense))

    mll = GaussianDistribution(jnp.atleast_1d(mx_flat.squeeze()), Sigma)
    return mll.log_prob(jnp.atleast_1d(y_flat.squeeze())).squeeze()


def conjugate_loocv(posterior: ConjugatePosterior, data: Dataset) -> ScalarFloat:
    r"""Evaluate the leave-one-out log predictive probability of the Gaussian process following
    section 5.4.2 of Rasmussen et al. 2006 - Gaussian Processes for Machine Learning. This metric
    calculates the average performance of all models that can be obtained by training on all but one
    data point, and then predicting the left out data point.

    The returned metric can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation
    given here enables exact estimation of the Gaussian process' latent
    function values.

    For a given ``ConjugatePosterior`` object, the following code snippet shows
    how the leave-one-out log predicitive probability can be evaluated.

    Example:
        >>> import gpjax as gpx
        ...
        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)
        ...
        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function = meanf, kernel=kernel)
        >>> posterior = prior * likelihood
        ...
        >>> gpx.objectives.conjugate_loocv(posterior, D)

        Our goal is to maximise the leave-one-out log predictive probability. Therefore, when
        optimising the model's parameters with respect to the parameters, we use the negative
        leave-one-out log predictive probability. This can be realised through

        >>> nloocv = lambda p, d: -gpx.objectives.conjugate_loocv(p, d)

    Args:
        posterior (ConjugatePosterior): The posterior distribution for which
            we want to compute the marginal log-likelihood.
        data:: The training dataset used to compute the
            marginal log-likelihood.

    Returns
    -------
    ScalarFloat
        The marginal log-likelihood of the Gaussian process.
    """

    x, y = data.X, data.y

    # Observation noise o²
    obs_var = posterior.likelihood.obs_stddev[...] ** 2

    mx = posterior.prior.mean_function(x)  # [N, M]

    # Σ = (Kxx + Io²)
    Kxx = posterior.prior.kernel.gram(x)
    Sigma_dense = Kxx.to_dense() + jnp.eye(Kxx.shape[0]) * (
        obs_var + posterior.prior.jitter
    )
    Sigma = psd(Dense(Sigma_dense))  # [N, N]

    Sigma_inv_y = solve(Sigma, y - mx)  # [N, 1]
    Sigma_inv = jnp.linalg.inv(Sigma.to_dense())
    Sigma_inv_diag = jnp.diag(Sigma_inv)[:, None]  # [N, 1]

    loocv_means = mx + (y - mx) - Sigma_inv_y / Sigma_inv_diag
    loocv_stds = jnp.sqrt(1.0 / Sigma_inv_diag)

    loocv_posterior = npd.Normal(loc=loocv_means, scale=loocv_stds)
    return jnp.sum(loocv_posterior.log_prob(y))


def log_posterior_density(
    posterior: NonConjugatePosterior, data: Dataset
) -> ScalarFloat:
    r"""The log-posterior density of a non-conjugate Gaussian process. This is
    sometimes referred to as the marginal log-likelihood.

    Evaluate the log-posterior density of a Gaussian process.

    Compute the marginal log-likelihood, or log-posterior density of the Gaussian
    process. The returned function can then be used for gradient based optimisation
    of the model's parameters or for model comparison. The implementation given
    here is general and will work for any likelihood support by GPJax.

    Unlike the marginal_log_likelihood function of the `ConjugatePosterior` object,
    the marginal_log_likelihood function of the `NonConjugatePosterior` object does
    not provide an exact marginal log-likelihood function. Instead, the
    `NonConjugatePosterior` object represents the posterior distributions as a
    function of the model's hyperparameters and the latent function. Markov chain
    Monte Carlo, variational inference, or Laplace approximations can then be used
    to sample from, or optimise an approximation to, the posterior distribution.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> gpx.objectives.log_posterior_density(posterior, D)

    Args:
        posterior (NonConjugatePosterior): The posterior distribution for which
            we want to compute the marginal log-likelihood.
        data: The training dataset used to compute the
            marginal log-likelihood.

    Returns
    -------
    ScalarFloat
        The log-posterior density of the Gaussian process.
    """

    x, y = data.X, data.y

    # Gram matrix
    Kxx = posterior.prior.kernel.gram(x)
    Kxx_dense = add_jitter(Kxx.to_dense(), posterior.prior.jitter)
    Kxx = psd(Dense(Kxx_dense))
    Lx = lower_cholesky(Kxx)

    # Compute the prior mean function
    mx = posterior.prior.mean_function(x)

    # Whitened function values, wx, corresponding to the inputs, x
    wx = posterior.latent[...]

    # f(x) = mx  +  Lx wx
    fx = mx + Lx @ wx

    # p(y | f(x), θ), where θ are the model hyperparameters
    likelihood = posterior.likelihood.link_function(fx)

    # Whitened latent function values prior, p(wx | θ) = N(0, I)
    latent_prior = npd.Normal(loc=0.0, scale=1.0)
    return likelihood.log_prob(y).sum() + latent_prior.log_prob(wx).sum()


non_conjugate_mll = log_posterior_density


def elbo(variational_family: VF, data: Dataset) -> ScalarFloat:
    r"""Compute the evidence lower bound of a variational approximation.

    Compute the evidence lower bound under this model. In short, this requires
    evaluating the expectation of the model's log-likelihood under the variational
    approximation. To this, we sum the KL divergence from the variational posterior
    to the prior. When batching occurs, the result is scaled by the batch size
    relative to the full dataset size.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     posterior=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.elbo(q, D)

    Args:
        variational_family: The variational
            approximation for whose parameters we should maximise the ELBO with
            respect to.
        data: The training data for which we should maximise the
            ELBO with respect to.

    Returns
    -------
    ScalarFloat
        The evidence lower bound of the variational approximation.
    """
    # KL[q(f(·)) || p(f(·))]
    kl = variational_family.prior_kl()

    # ∫[log(p(y|f(·))) q(f(·))] df(·)
    var_exp = variational_expectation(variational_family, data)

    # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
    return (
        jnp.sum(var_exp)
        * variational_family.posterior.likelihood.num_datapoints
        / data.n
        - kl
    )


def variational_expectation(
    variational_family: VF,
    data: Dataset,
) -> Float[Array, " N"]:
    r"""Compute the variational expectation.

    Compute the expectation of our model's log-likelihood under our variational
    distribution. Batching can be done here to speed up computation.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Bernoulli(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.VariationalGaussian(
        ...     posterior=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.variational_expectation(q, D)

    Args:
        variational_family: The variational family that we
            are using to approximate the posterior.
        data: The batch for which the expectation should be computed for.

    Returns
    -------
    Array
        The expectation of the model's log-likelihood under our variational
        distribution.
    """
    # Unpack training batch
    x, y = data.X, data.y

    # Variational distribution q(f(·)) = N(f(·); μ(·), Σ(·, ·))
    q = variational_family

    # TODO: This needs cleaning up! We are squeezing then broadcasting `mean` and `variance`, which is not ideal.

    # Compute variational mean, μ(x), and variance, diag(Σ(x, x)), at the training
    # inputs, x
    def q_moments(x):
        qx = q(x)
        return qx.mean.squeeze(), qx.covariance().squeeze()

    mean, variance = vmap(q_moments)(x[:, None])

    # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
    expectation = q.posterior.likelihood.expected_log_likelihood(
        y, mean[:, None], variance[:, None]
    )
    return expectation


# TODO: Replace code within CollapsedELBO to using (low rank structure of) LinOps and the GaussianDistribution object to be as succinct as e.g., the `ConjugateMLL`.


def collapsed_elbo(variational_family: VF, data: Dataset) -> ScalarFloat:
    r"""Compute a single step of the collapsed evidence lower bound.

    Compute the evidence lower bound under this model. In short, this requires
    evaluating the expectation of the model's log-likelihood under the variational
    approximation. To this, we sum the KL divergence from the variational posterior
    to the prior. When batching occurs, the result is scaled by the batch size
    relative to the full dataset size.

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp

        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)

        >>> meanf = gpx.mean_functions.Constant()
        >>> kernel = gpx.kernels.RBF()
        >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        >>> prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        >>> posterior = prior * likelihood

        >>> z = jnp.linspace(0, 1, 10).reshape(-1, 1)
        >>> q = gpx.variational_families.CollapsedVariationalGaussian(
        ...     posterior=posterior, inducing_inputs=z
        ... )

        >>> gpx.objectives.collapsed_elbo(q, D)

    Args:
        variational_family: The variational
            approximation for whose parameters we should maximise the ELBO with
            respect to.
        data: The training data for which we should maximise the
            ELBO with respect to.

    Returns
    -------
    ScalarFloat
        The evidence lower bound of the variational approximation.
    """
    # Unpack training data
    x, y, n = data.X, data.y, data.n

    # Unpack mean function and kernel
    mean_function = variational_family.posterior.prior.mean_function
    kernel = variational_family.posterior.prior.kernel

    m = variational_family.num_inducing

    noise = variational_family.posterior.likelihood.obs_stddev[...] ** 2
    z = variational_family.inducing_inputs[...]
    Kzz = kernel.gram(z)
    Kzz_dense = add_jitter(Kzz.to_dense(), variational_family.jitter)
    Kzz = psd(Dense(Kzz_dense))
    Kzx = kernel.cross_covariance(z, x)
    Kxx_diag = vmap(kernel, in_axes=(0, 0))(x, x)
    μx = mean_function(x)

    Lz = lower_cholesky(Kzz)

    # Notation and derivation:
    #
    # Let Q = KxzKzz⁻¹Kzx, we must compute the log normal pdf:
    #
    #   log N(y; μx, o²I + Q) = -nπ - n/2 log|o²I + Q|
    #   - 1/2 (y - μx)ᵀ (o²I + Q)⁻¹ (y - μx).
    #
    # The log determinant |o²I + Q| is computed via applying the matrix determinant
    #   lemma
    #
    #   |o²I + Q| = log|o²I| + log|I + Lz⁻¹ Kzx (o²I)⁻¹ Kxz Lz⁻¹| = log(o²) +  log|B|,
    #
    #   with B = I + AAᵀ and A = Lz⁻¹ Kzx / o.
    #
    # Similarly we apply matrix inversion lemma to invert o²I + Q
    #
    #   (o²I + Q)⁻¹ = (Io²)⁻¹ - (Io²)⁻¹ Kxz Lz⁻ᵀ (I + Lz⁻¹ Kzx (Io²)⁻¹ Kxz Lz⁻ᵀ )⁻¹ Lz⁻¹ Kzx (Io²)⁻¹
    #               = (Io²)⁻¹ - (Io²)⁻¹ oAᵀ (I + oA (Io²)⁻¹ oAᵀ)⁻¹ oA (Io²)⁻¹
    #               = I/o² - Aᵀ B⁻¹ A/o²,
    #
    # giving the quadratic term as
    #
    #   (y - μx)ᵀ (o²I + Q)⁻¹ (y - μx) = [(y - μx)ᵀ(y - µx)  - (y - μx)ᵀ Aᵀ B⁻¹ A (y - μx)]/o²,
    #
    #   with A and B defined as above.

    A = solve(Lz, Kzx) / jnp.sqrt(noise)

    # AAᵀ
    AAT = jnp.matmul(A, A.T)

    # B = I + AAᵀ
    B = jnp.eye(m) + AAT

    # LLᵀ = I + AAᵀ
    L = jnp.linalg.cholesky(B)

    # log|B| = 2 trace(log|L|) = 2 Σᵢ log Lᵢᵢ  [since |B| = |LLᵀ| = |L|²  => log|B| = 2 log|L|, and |L| = Πᵢ Lᵢᵢ]
    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))

    diff = y - μx

    # L⁻¹ A (y - μx)
    L_inv_A_diff = jsp.linalg.solve_triangular(L, jnp.matmul(A, diff), lower=True)

    # (y - μx)ᵀ (Io² + Q)⁻¹ (y - μx)
    quad = (jnp.sum(diff**2) - jnp.sum(L_inv_A_diff**2)) / noise

    # 2 * log N(y; μx, Io² + Q)
    two_log_prob = -n * jnp.log(2.0 * jnp.pi * noise) - log_det_B - quad

    # 1/o² tr(Kxx - Q) [Trace law tr(AB) = tr(BA) => tr(KxzKzz⁻¹Kzx) = tr(KxzLz⁻ᵀLz⁻¹Kzx) = tr(Lz⁻¹Kzx KxzLz⁻ᵀ) = trace(o²AAᵀ)]
    two_trace = jnp.sum(Kxx_diag) / noise - jnp.trace(AAT)

    # log N(y; μx, Io² + KxzKzz⁻¹Kzx) - 1/2o² tr(Kxx - KxzKzz⁻¹Kzx)
    return (two_log_prob - two_trace).squeeze() / 2.0


def heteroscedastic_elbo_conjugate(
    variational_family: HVF, data: Dataset
) -> ScalarFloat:
    r"""Tight bound from Lázaro-Gredilla & Titsias (2011) for heteroscedastic Gaussian likelihoods."""
    likelihood = variational_family.posterior.likelihood
    mean_f, var_f, mean_g, var_g = variational_family.predict(data.X)

    expected_ll, _ = likelihood.expected_log_likelihood(
        data.y,
        mean_f,
        var_f,
        mean_g=mean_g,
        variance_g=var_g,
        return_parts=True,
    )

    scale = likelihood.num_datapoints / data.n
    return scale * jnp.sum(expected_ll) - variational_family.prior_kl()


def heteroscedastic_elbo_chained(variational_family: HVF, data: Dataset) -> ScalarFloat:
    r"""Generic chained bound for heteroscedastic likelihoods."""
    likelihood: AbstractHeteroscedasticLikelihood = (
        variational_family.posterior.likelihood
    )
    mean_f, var_f, mean_g, var_g = variational_family.predict(data.X)
    noise_stats = likelihood.noise_statistics(mean_g, var_g)

    expected_ll = likelihood.expected_log_likelihood(
        data.y,
        mean_f,
        var_f,
        mean_g=mean_g,
        variance_g=var_g,
        noise_stats=noise_stats,
    )

    scale = likelihood.num_datapoints / data.n
    return scale * jnp.sum(expected_ll) - variational_family.prior_kl()


def heteroscedastic_elbo(variational_family: HVF, data: Dataset) -> ScalarFloat:
    likelihood = variational_family.posterior.likelihood
    if likelihood.supports_tight_bound():
        return heteroscedastic_elbo_conjugate(variational_family, data)
    return heteroscedastic_elbo_chained(variational_family, data)
