"""Closed-form optimal $q$ for a conjugate SVGP.

Reference implementation of the fixed point that a $\\gamma=1$ natural-gradient step
lands on for a Gaussian likelihood (Salimbeni et al. 2018, as transcribed in
``plans/natgrads-specs/spec-salimbeni.md`` section 8.2). Written once and shared by
``tests/test_natural_gradients.py`` and ``tests/test_fit.py`` so that a sign or
transpose correction cannot be applied to one transcription and missed by the other.
"""

from gpjax.linalg import add_jitter
from gpjax.parameters import val
from gpjax.variational_families import WhitenedVariationalGaussian
import jax.numpy as jnp
import jax.scipy as jsp


def conjugate_optimum(variational_family, data, scale=None):
    r"""Return the optimal $(\mathbf m^\star,\mathbf S^\star,\boldsymbol\Lambda)$.

    Unwhitened: $\Lambda = K_{zz}^{-1} + s\sigma^{-2}A^\top A$ and
    $b = s\sigma^{-2}A^\top\tilde y + K_{zz}^{-1}\mu_z$.
    Whitened: $\Lambda_w = I + s\sigma^{-2}A_w^\top A_w$ and
    $b_w = s\sigma^{-2}A_w^\top(y-\mu_x)$.

    Parameters
    ----------
    variational_family
        The family whose hyperparameters define the optimum. Its variational
        coordinates are ignored -- the optimum does not depend on them.
    data
        The batch the optimum is computed on.
    scale
        The mini-batch correction $s$ that ``elbo`` applies. Defaults to
        ``full_size / data.n`` with ``full_size = data.n_total if data.n_total is
        not None else data.n``, which is what ``elbo`` uses; pass ``1.0`` to
        compare against a bound written without the correction.

    Returns
    -------
    tuple
        ``(optimal_mean, optimal_covariance, precision)``.
    """
    inducing_inputs = val(variational_family.inducing_inputs)
    kernel = variational_family.model.prior.kernel
    mean_function = variational_family.model.prior.mean_function

    gram = add_jitter(
        kernel.gram(inducing_inputs).as_matrix(),
        variational_family.model.prior.jitter,
    )
    cross = kernel.cross_covariance(data.X, inducing_inputs)
    inducing_mean = mean_function(inducing_inputs)
    input_mean = mean_function(data.X)
    noise_variance = val(variational_family.model.likelihood.obs_stddev) ** 2
    if scale is None:
        full_size = data.n_total if data.n_total is not None else data.n
        scale = full_size / data.n
    num_inducing = gram.shape[0]

    if isinstance(variational_family, WhitenedVariationalGaussian):
        root_gram = jnp.linalg.cholesky(gram)
        design = jsp.linalg.solve_triangular(root_gram, cross.T, lower=True).T
        precision = jnp.eye(num_inducing) + scale * design.T @ design / noise_variance
        shift = scale * design.T @ (data.y - input_mean) / noise_variance
    else:
        gram_inverse = jnp.linalg.inv(gram)
        design = cross @ gram_inverse
        residual = data.y - input_mean + design @ inducing_mean
        precision = gram_inverse + scale * design.T @ design / noise_variance
        shift = (
            scale * design.T @ residual / noise_variance + gram_inverse @ inducing_mean
        )

    covariance = jnp.linalg.inv(precision)
    return covariance @ shift, covariance, precision
