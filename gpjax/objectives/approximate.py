import jax.numpy as jnp
from ..gps import ConjugatePosterior
from multipledispatch import dispatch
from typing import Callable
from ..types import SparseDataset, Array
from ..utils import concat_dictionaries, I, chol_log_det
from ..kernels import gram, cross_covariance
from jax.scipy.linalg import solve_triangular, cholesky
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


@dispatch(ConjugatePosterior, Callable)
def elbo(gp: ConjugatePosterior, transform: Callable, jitter: float=1e-6, to_minimise: bool=True):
    """
    The bound given below follows equation (4) in https://www.jmlr.org/papers/volume18/16-603/16-603.pdf
    Where possible, variable names will mimic the notation in this paper, and equation numbers will be referenced
    where possible.
    """
    def bound(params: dict, training: SparseDataset,static_params: dict = None) -> Array:
        # Extract data matrices
        X, y = training.X, training.y
        # TODO: Perhaps we can just use the Dataset class here and remove that chunk
        Z = params['inducing_inputs']

        # Define necessary constants
        N = training.n
        pi = jnp.pi
        params = transform(params)
        if static_params:
            params = concat_dictionaries(params, transform(static_params))

        # Compute the necessary kernel matrices
        kernel = gp.prior.kernel
        Kff = gram(kernel, X, params) + I(N)*jitter # N x N matrix
        Kuu = gram(kernel, Z, params) + I(Z.shape[0])*jitter # M x M matrix
        Kuf = cross_covariance(kernel, Z, X, params) # M x N matrix
        Luu = cholesky(Kuu, lower=True)
        Qff =

        # Set about computing the ELBO term by term
        constant = N*jnp.log(2*pi)/2

        Qff = jnp.matmul(jnp.matmul(jnp.transpose(Kuf), jnp.linalg.inv(Kuu)), Kuf)
        G = I(N) * params['obs_noise']
        Kffhat = Qff + G

        complexity = 0.5*chol_log_det(Kffhat)
        data_fit = 0.5 * jnp.matmul(y.dot(y), jnp.linalg.inv(Kffhat))
        trace_term = 0.5*jnp.clip(jnp.sum(jnp.diag(Kff) - jnp.diag(Qff))/params['obs_noise'], a_min=0.)

        minimising_constant = jnp.array(1.0) if to_minimise else jnp.array(-1.0)
        return minimising_constant*(constant+complexity+data_fit+trace_term)
    return bound
