import tensorflow_probability.substrates.jax.bijectors as tfb
from ml_collections import ConfigDict


def get_defaults():
    config = ConfigDict()
    # Covariance matrix stabilising jitter
    config.jitter = 1e-6
    # Parameter transformations
    config.transformations = transformations = ConfigDict()
    transformations.positive_transform = tfb.Softplus
    transformations.identity_transform = tfb.Identity

    transformations.lengthscale = "positive_transform"
    transformations.variance = "positive_transform"
    transformations.obs_noise = "positive_transform"
    transformations.latent = "identity_transform"
    transformations.basis_fns = "identity_transform"
    transformations.inducing_inputs = "identity_transform"
    return config
