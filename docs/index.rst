Welcome to GPJax's documentation!
=================================

GPJax is a didactic Gaussian process that supports GPU acceleration and just-in-time compilation. The intention of GPJax is to provide a framework for researchers to rapidly prototype and develop new Gaussian process methods. We seek to provide this by providing an API that seeks to best represent the underlying mathematics of Gaussian processes.

Gaussian process 'Hello World' example
-----------------------------------------------------

To gain an intuition for the exposed API provided by GPJax, a simple example of that derives the Gaussian process posterior for regression can be represented by::

    import jax.numpy as jnp
    import jax.random as jr
    import gpjax
    key = jr.PRNGKey(123)

    X = jnp.linspace(-2., 2., 100)
    y = jnp.sin(x) + jr.normal(key, shape=X.shape)*0.05

    kernel = gpjax.kernels.RBF()
    f_prior = gpjax.gps.Prior(kernel)

    likelihood = gpjax.likelihoods.Gaussian()

    posterior = f_prior * likelihood



.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    installation
    nbs/intro

.. toctree::
    :maxdepth: 1
    :caption: Examples

    nbs/regression
    nbs/classification
    nbs/technical_details
