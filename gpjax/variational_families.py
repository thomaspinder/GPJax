# Copyright 2022 The GPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
from dataclasses import dataclass

import beartype.typing as tp
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import (
    Float,
    Int,
)
import lineax as lx

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.gps import (
    HeteroscedasticModel,
    JointModel,
    Prior,
)
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractHeteroscedasticLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.linalg import cholesky_factor
from gpjax.linalg.utils import add_jitter
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import (
    LowerTriangular,
    Real,
    _val,
)
from gpjax.summary import _SummaryMixin
from gpjax.typing import (
    Array,
    ScalarFloat,
)

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", Gaussian, NonGaussian)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)
HL = tp.TypeVar("HL", bound=AbstractHeteroscedasticLikelihood)
P = tp.TypeVar("P", bound=Prior)
PP = tp.TypeVar("PP", bound=JointModel)
HP = tp.TypeVar("HP", bound=HeteroscedasticModel)


def _psd(matrix):
    """Wrap a dense matrix as a PSD lineax operator."""
    return lx.MatrixLinearOperator(matrix)


def _tri_solve(L, B):
    """Solve L x = B where L is lower triangular. Works for matrix B."""
    return jsp.linalg.solve_triangular(L, B, lower=True)


class AbstractVariationalFamily(_SummaryMixin, eqx.Module, tp.Generic[L]):
    r"""
    Abstract base class used to represent families of distributions that can be
    used within variational inference.
    """

    posterior: JointModel

    def __init__(self, posterior: JointModel):
        self.posterior = posterior

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Evaluate the variational family's density.

        For a given set of parameters, compute the latent function's prediction
        under the variational approximation.

        Args:
            *args (Any): Arguments of the variational family's `predict` method.
            **kwargs (Any): Keyword arguments of the variational family's `predict`
                method.

        Returns:
            GaussianDistribution: The output of the variational family's `predict` method.
        """
        return self.predict(*args, **kwargs)

    @abc.abstractmethod
    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> GaussianDistribution:
        r"""Predict the GP's output given the input.

        Args:
            *args (Any): Arguments of the variational family's ``predict``
                method.
            **kwargs (Any): Keyword arguments of the variational family's
                ``predict`` method.

        Returns:
            GaussianDistribution: The output of the variational family's ``predict`` method.
        """
        raise NotImplementedError


class AbstractVariationalGaussian(AbstractVariationalFamily[L]):
    r"""The variational Gaussian family of probability distributions."""

    inducing_inputs: tp.Any
    jitter: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        posterior: JointModel,
        inducing_inputs: tp.Union[
            Int[Array, "N D"],
            Float[Array, "N D"],
            Real,
        ],
        jitter: ScalarFloat = 1e-6,
    ):
        if not isinstance(inducing_inputs, Real):
            inducing_inputs = Real(inducing_inputs)

        self.inducing_inputs = inducing_inputs
        self.jitter = jitter

        super().__init__(posterior)

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return _val(self.inducing_inputs).shape[0]


class VariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$, where
    $u = f(z)$ are the function values at the inducing inputs $z$
    and the distribution over the inducing inputs is
    $q(u) = \mathcal{N}(\mu, S)$.  We parameterise this over
    $\mu$ and $sqrt$ with $S = sqrt sqrt^{\top}$.
    """

    variational_mean: tp.Any
    variational_root_covariance: tp.Any

    def __init__(
        self,
        posterior: JointModel,
        inducing_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]],
        variational_mean: tp.Union[Float[Array, "N 1"], None] = None,
        variational_root_covariance: tp.Union[Float[Array, "N N"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if variational_mean is None:
            variational_mean = jnp.zeros((self.num_inducing, 1))

        if variational_root_covariance is None:
            variational_root_covariance = jnp.eye(self.num_inducing)

        self.variational_mean = Real(variational_mean)
        self.variational_root_covariance = LowerTriangular(variational_root_covariance)

    def _fmt_Kzt_Ktt(self, Kzt, Ktt):
        return Kzt, Ktt

    def _fmt_inducing_inputs(self):
        return _val(self.inducing_inputs)

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the prior KL divergence.

        Compute the KL-divergence between our variational approximation and the
        Gaussian process prior.

        For this variational family, we have

        .. math::

            \begin{aligned}
            \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
            & = \operatorname{KL}[ \mathcal{N}(\mu, S) \mid\mid N(\mu z, \mathbf{K}_{zz}) ],
            \end{aligned}

        where $u = f(z)$ and $z$ are the inducing inputs.

        With $S = LL^{\top}$ for the stored triangular root $L$ and
        $\mathbf{K}_{zz} = L_z L_z^{\top}$, this evaluates in closed form as

        .. math::

            \tfrac{1}{2}\left(
                \lVert L_z^{-1}(\mu_z - \mu)\rVert^2
                + \lVert L_z^{-1} L\rVert_F^2
                - m
                + 2\sum_i \log [L_z]_{ii}
                - 2\sum_i \log \lvert L_{ii}\rvert
            \right),

        so the Cholesky factor of $\mathbf{K}_{zz}$ is the only factorisation
        required; $S$ is never formed and never re-factorised.

        Returns:
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        variational_mean = _val(self.variational_mean)
        variational_sqrt = _val(self.variational_root_covariance)
        inducing_inputs = self._fmt_inducing_inputs()
        num_inducing = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        inducing_mean = mean_function(inducing_inputs)
        Kzz = kernel.gram(inducing_inputs)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)

        # Lz Lz^T = Kzz. The single unavoidable factorisation.
        Lz = jnp.linalg.cholesky(Kzz_dense)

        # (muz - mu)^T Kzz^{-1} (muz - mu) = ||Lz^{-1} (muz - mu)||^2
        mahalanobis = jnp.sum(
            jnp.square(_tri_solve(Lz, inducing_mean - variational_mean))
        )

        # tr[Kzz^{-1} S] = ||Lz^{-1} sqrt||_F^2  [recall S = sqrt sqrt^T]
        trace = jnp.sum(jnp.square(_tri_solve(Lz, variational_sqrt)))

        # log|Kzz| and log|S|. The absolute value keeps log|S| = 2 sum log|sqrt_ii|
        # valid for any square root, though `LowerTriangular` already guarantees a
        # positive diagonal.
        log_det_prior = 2.0 * jnp.sum(jnp.log(jnp.diag(Lz)))
        log_det_variational = 2.0 * jnp.sum(
            jnp.log(jnp.abs(jnp.diag(variational_sqrt)))
        )

        return 0.5 * (
            mahalanobis - num_inducing + log_det_prior - log_det_variational + trace
        )

    def predict(
        self, test_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]]
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral $q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u$, which
        can be computed in closed form as:

        .. math::

            \mathcal{N}\left(f(t); \mu t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu - \mu z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt}\right).

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        # Unpack variational parameters
        variational_mean = _val(self.variational_mean)
        variational_sqrt = _val(self.variational_root_covariance)
        inducing_inputs = self._fmt_inducing_inputs()

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(inducing_inputs)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)
        Lz = jnp.linalg.cholesky(Kzz_dense)
        inducing_mean = mean_function(inducing_inputs)

        # Unpack test inputs
        test_points = test_inputs

        Ktt = kernel.gram(test_points).as_matrix()
        Kzt = kernel.cross_covariance(inducing_inputs, test_points)
        test_mean = mean_function(test_points)

        Kzt, Ktt = self._fmt_Kzt_Ktt(Kzt, Ktt)

        # Lz^{-1} Kzt
        Lz_inv_Kzt = _tri_solve(Lz, Kzt)

        # Kzz^{-1} Kzt
        Kzz_inv_Kzt = jsp.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)

        # Ktz Kzz^{-1} sqrt
        Ktz_Kzz_inv_sqrt = jnp.matmul(Kzz_inv_Kzt.T, variational_sqrt)

        # mut + Ktz Kzz^{-1} (mu - muz)
        mean = test_mean + jnp.matmul(Kzz_inv_Kzt.T, variational_mean - inducing_mean)

        # Ktt - Ktz Kzz^{-1} Kzt + Ktz Kzz^{-1} S Kzz^{-1} Kzt  [recall S = sqrt sqrt^T]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )

        covariance = add_jitter(covariance, self.jitter)
        covariance_op = lx.MatrixLinearOperator(covariance)

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance_op
        )


class GraphVariationalGaussian(VariationalGaussian[L]):
    r"""A variational Gaussian defined over graph-structured inducing inputs.

    This subclass adapts the :class:`VariationalGaussian` family to the
    case where the inducing inputs are discrete graph node indices rather
    than continuous spatial coordinates.

    The main differences are:
      * Inducing inputs are integer node IDs.
      * Kernel matrices are ensured to be dense and 2D.
    """

    def __init__(
        self,
        posterior: JointModel,
        inducing_inputs: Int[Array, "N D"],
        variational_mean: tp.Union[Float[Array, "N 1"], None] = None,
        variational_root_covariance: tp.Union[Float[Array, "N N"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(
            posterior,
            inducing_inputs,
            variational_mean,
            variational_root_covariance,
            jitter,
        )
        self.inducing_inputs = _val(self.inducing_inputs).astype(jnp.int64)

    def _fmt_Kzt_Ktt(self, Kzt, Ktt):
        Ktt = Ktt.as_matrix() if hasattr(Ktt, "as_matrix") else Ktt
        Kzt = Kzt.as_matrix() if hasattr(Kzt, "as_matrix") else Kzt
        Ktt = jnp.atleast_2d(Ktt)
        Kzt = (
            jnp.transpose(jnp.atleast_2d(Kzt)) if Kzt.ndim < 2 else jnp.atleast_2d(Kzt)
        )
        return Kzt, Ktt

    def _fmt_inducing_inputs(self):
        return self.inducing_inputs

    @property
    def num_inducing(self) -> int:
        """The number of inducing inputs."""
        return _val(self.inducing_inputs).shape[0]


class WhitenedVariationalGaussian(VariationalGaussian[L]):
    r"""The whitened variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$,
    where $u = f(z)$
    are the function values at the inducing inputs $z$ and the distribution over
    the inducing inputs is $q(u) = \mathcal{N}(Lz \mu + mz, Lz S Lz^{\top})$. We parameterise this
    over $\mu$ and $sqrt$ with $S = sqrt sqrt^{\top}$.
    """

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the KL-divergence between our variational approximation and
        the Gaussian process prior.

        For this variational family, we have

        .. math::

            \begin{aligned}
            \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)]\\
                & = \operatorname{KL}[N(\mu  , S)\mid\mid N(0, I)].
            \end{aligned}

        Against a standard normal prior the divergence has a closed form that
        needs no matrix factorisation at all. Writing $S = LL^{\top}$ for the
        stored triangular root $L$, and using
        $\operatorname{tr}[S] = \lVert L\rVert_F^2$ and
        $\log\lvert S\rvert = 2\sum_i \log\lvert L_{ii}\rvert$,

        .. math::

            \operatorname{KL}[\mathcal{N}(\mu, S)\mid\mid\mathcal{N}(0, I)] =
            \tfrac{1}{2}\left(
                \lVert\mu\rVert^2 + \lVert L\rVert_F^2 - m
                - 2\sum_i \log\lvert L_{ii}\rvert
            \right),

        where $m$ is the number of inducing points.

        Returns:
            ScalarFloat: The KL-divergence between our variational
                approximation and the GP prior.
        """
        # Unpack variational parameters
        mu = _val(self.variational_mean)
        sqrt = _val(self.variational_root_covariance)

        # mu^T I^{-1} mu, tr[S] = ||sqrt||_F^2 and log|S| = 2 sum log|sqrt_ii|.
        # The absolute value keeps the log-determinant valid for any square root,
        # though `LowerTriangular` already guarantees a positive diagonal.
        mahalanobis = jnp.sum(jnp.square(mu))
        trace = jnp.sum(jnp.square(sqrt))
        log_det_variational = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(sqrt))))

        return 0.5 * (mahalanobis + trace - self.num_inducing - log_det_variational)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        This is the integral q(f(t)) = \int p(f(t)\midu) q(u) du, which can be computed in
        closed form as

        .. math::

            \mathcal{N}\left(f(t); \mu t  +  \mathbf{K}_{tz} \mathbf{L}z^{\top} \mu  ,  \mathbf{K}_{tt}  -  \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt}  +  \mathbf{K}_{tz} \mathbf{L}z^{\top} S \mathbf{L}z^{-1} \mathbf{K}_{zt} \right).

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        # Unpack variational parameters
        mu = _val(self.variational_mean)
        sqrt = _val(self.variational_root_covariance)
        z = _val(self.inducing_inputs)

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzz = kernel.gram(z)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)
        Lz = jnp.linalg.cholesky(Kzz_dense)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t).as_matrix()
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz^{-1} Kzt
        Lz_inv_Kzt = _tri_solve(Lz, Kzt)

        # Ktz Lz^{-T} sqrt
        Ktz_Lz_invT_sqrt = jnp.matmul(Lz_inv_Kzt.T, sqrt)

        # mut + Ktz Lz^{-T} mu
        mean = mut + jnp.matmul(Lz_inv_Kzt.T, mu)

        # Ktt - Ktz Kzz^{-1} Kzt + Ktz Lz^{-T} S Lz^{-1} Kzt  [recall S = sqrt sqrt^T]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Lz_invT_sqrt, Ktz_Lz_invT_sqrt.T)
        )
        covariance = add_jitter(covariance, self.jitter)
        covariance_op = lx.MatrixLinearOperator(covariance)

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance_op
        )


class NaturalVariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The natural variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$,
    where $u = f(z)$ are
    the function values at the inducing inputs $z$ and the distribution over the
    inducing inputs is $q(u) = N(\mu, S)$. Expressing the variational distribution, in
    the form of the exponential family, $q(u) = exp(\theta^{\top} T(u) - a(\theta))$, gives rise to the
    natural parameterisation $\theta  = (\theta_{1}, \theta_{2}) = (S^{-1}\mu, -S^{-1}/2)$, to perform model inference,
    where $T(u) = [u, uu^{\top}]$ are the sufficient statistics.
    """

    natural_vector: tp.Any
    natural_matrix: tp.Any

    def __init__(
        self,
        posterior: JointModel,
        inducing_inputs: Float[Array, "N D"],
        natural_vector: tp.Union[Float[Array, "M 1"], None] = None,
        natural_matrix: tp.Union[Float[Array, "M M"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if natural_vector is None:
            natural_vector = jnp.zeros((self.num_inducing, 1))

        if natural_matrix is None:
            natural_matrix = -0.5 * jnp.eye(self.num_inducing)

        self.natural_vector = Real(natural_vector)
        self.natural_matrix = Real(natural_matrix)

    def prior_kl(self) -> ScalarFloat:
        r"""Compute the KL-divergence between our current variational approximation
        and the Gaussian process prior.

        For this variational family, we have

        .. math::

            \begin{aligned}
            \operatorname{KL}[q(f(\cdot))\mid\mid p(\cdot)] & = \operatorname{KL}[q(u)\mid\mid p(u)] \\
                & = \operatorname{KL}[N(\mu, S)\mid\mid N(mz, \mathbf{K}_{zz})],
            \end{aligned}

        with $\mu$ and $S$ computed from the natural parameterisation $\theta  = (S^{-1}\mu  , -S^{-1}/2)$.

        Returns:
            ScalarFloat: The KL-divergence between our variational approximation and
                the GP prior.
        """
        # Unpack variational parameters
        natural_vector = _val(self.natural_vector)
        natural_matrix = _val(self.natural_matrix)
        z = _val(self.inducing_inputs)
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # S^{-1} = -2 theta_2
        S_inv = -2 * natural_matrix
        S_inv = add_jitter(S_inv, self.jitter)

        # Compute L^{-1}, where LL^T = S, via a trick found in the NumPyro source code
        sqrt_inv = jnp.swapaxes(
            jnp.linalg.cholesky(S_inv[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
        )

        # L = (L^{-1})^{-1}I
        sqrt = jsp.linalg.solve_triangular(sqrt_inv, jnp.eye(m), lower=True)

        # S = LL^T:
        S = lx.MatrixLinearOperator(sqrt @ sqrt.T)

        # mu = S theta_1
        mu = S.as_matrix() @ natural_vector

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)
        Kzz_op = _psd(Kzz_dense)

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz_op)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs $t$.

        This is the integral $q(f(t)) = \int p(f(t)\mid u) q(u) \mathrm{d}u$, which
        can be computed in closed form as

        .. math::

            \mathcal{N}\left(f(t); \mu  t + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} (\mu   - \mu  z),  \mathbf{K}_{tt} - \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} + \mathbf{K}_{tz} \mathbf{K}_{zz}^{-1} S \mathbf{K}_{zz}^{-1} \mathbf{K}_{zt} \right),

        with $\mu$ and $S$ computed from the natural parameterisation
        $\theta = (S^{-1}\mu  , -S^{-1}/2)$.

        Returns:
            GaussianDistribution: A function that accepts a set of test points and will
                return the predictive distribution at those points.
        """
        # Unpack variational parameters
        natural_vector = _val(self.natural_vector)
        natural_matrix = _val(self.natural_matrix)
        z = _val(self.inducing_inputs)
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # S^{-1} = -2 theta_2
        S_inv = -2 * natural_matrix
        S_inv = add_jitter(S_inv, self.jitter)

        # Compute L^{-1}, where LL^T = S
        sqrt_inv = jnp.swapaxes(
            jnp.linalg.cholesky(S_inv[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
        )

        # L = (L^{-1})^{-1}I
        sqrt = jsp.linalg.solve_triangular(sqrt_inv, jnp.eye(m), lower=True)

        # S = LL^T:
        S = jnp.matmul(sqrt, sqrt.T)

        # mu = S theta_1
        mu = jnp.matmul(S, natural_vector)

        Kzz = kernel.gram(z)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)
        Lz = jnp.linalg.cholesky(Kzz_dense)
        muz = mean_function(z)

        Ktt = kernel.gram(test_inputs).as_matrix()
        Kzt = kernel.cross_covariance(z, test_inputs)
        mut = mean_function(test_inputs)

        # Lz^{-1} Kzt
        Lz_inv_Kzt = _tri_solve(Lz, Kzt)

        # Kzz^{-1} Kzt
        Kzz_inv_Kzt = jsp.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)

        # Ktz Kzz^{-1} L
        Ktz_Kzz_inv_L = jnp.matmul(Kzz_inv_Kzt.T, sqrt)

        # mut + Ktz Kzz^{-1} (mu - muz)
        mean = mut + jnp.matmul(Kzz_inv_Kzt.T, mu - muz)

        # Ktt - Ktz Kzz^{-1} Kzt + Ktz Kzz^{-1} S Kzz^{-1} Kzt  [recall S = LL^T]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_L, Ktz_Kzz_inv_L.T)
        )
        covariance = add_jitter(covariance, self.jitter)
        covariance_op = lx.MatrixLinearOperator(covariance)

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance_op
        )


class ExpectationVariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The natural variational Gaussian family of probability distributions.

    The variational family is $q(f(\cdot)) = \int p(f(\cdot)\mid u) q(u) \mathrm{d}u$, where $u = f(z)$ are the
    function values at the inducing inputs $z$ and the distribution over the inducing
    inputs is $q(u) = \mathcal{N}(\mu, S)$. Expressing the variational distribution, in the form of
    the exponential family, $q(u) = exp(\theta^{\top} T(u) - a(\theta))$, gives rise to the natural
    parameterisation $\theta  = (\theta_{1}, \theta_{2}) = (S^{-1}\mu  , -S^{-1}/2)$ and sufficient statistics
    $T(u) = [u, uu^{\top}]$. The expectation parameters are given by $\nu = \int T(u) q(u) \mathrm{d}u$.
    This gives a parameterisation, $\nu = (\nu_{1}, \nu_{2}) = (\mu  , S + uu^{\top})$ to perform model
    inference over.
    """

    expectation_vector: tp.Any
    expectation_matrix: tp.Any

    def __init__(
        self,
        posterior: JointModel,
        inducing_inputs: Float[Array, "N D"],
        expectation_vector: tp.Union[Float[Array, "M 1"], None] = None,
        expectation_matrix: tp.Union[Float[Array, "M M"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if expectation_vector is None:
            expectation_vector = jnp.zeros((self.num_inducing, 1))

        if expectation_matrix is None:
            expectation_matrix = jnp.eye(self.num_inducing)

        self.expectation_vector = Real(expectation_vector)
        self.expectation_matrix = Real(expectation_matrix)

    def prior_kl(self) -> ScalarFloat:
        r"""Evaluate the prior KL-divergence.

        Compute the KL-divergence between our current variational approximation and
        the Gaussian process prior.

        For this variational family, we have

        .. math::

            \begin{aligned}
            \operatorname{KL}(q(f(\cdot))\mid\mid p(\cdot)) & = \operatorname{KL}(q(u)\mid\mid p(u)) \\
                & =\operatorname{KL}(\mathcal{N}(\mu, S)\mid\mid \mathcal{N}(m_z, K_{zz})),
            \end{aligned}

        where $\mu$ and $S$ are the expectation parameters of the variational
        distribution and $m_z$ and $K_{zz}$ are the mean and covariance of the prior
        distribution.

        Returns:
            ScalarFloat: The KL-divergence between our variational approximation and
                the GP prior.
        """
        # Unpack variational parameters
        expectation_vector = _val(self.expectation_vector)
        expectation_matrix = _val(self.expectation_matrix)
        z = _val(self.inducing_inputs)

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # mu = eta_1
        mu = expectation_vector

        # S = eta_2 - eta_1 eta_1^T
        S = expectation_matrix - jnp.outer(mu, mu)
        S_dense = add_jitter(S, self.jitter)
        S_op = _psd(S_dense)

        muz = mean_function(z)
        Kzz = kernel.gram(z)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)
        Kzz_op = _psd(Kzz_dense)

        qu = GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()), scale=S_op)
        pu = GaussianDistribution(loc=jnp.atleast_1d(muz.squeeze()), scale=Kzz_op)

        return qu.kl_divergence(pu)

    def predict(self, test_inputs: Float[Array, "N D"]) -> GaussianDistribution:
        r"""Evaluate the predictive distribution.

        Compute the predictive distribution of the GP at the test inputs $t$.

        This is the integral $q(f(t)) = \int p(f(t)\mid u)q(u)\mathrm{d}u$, which can
        be computed in closed form as  which can be computed in closed form as

        .. math::

            \mathcal{N}(f(t); \mu_t + \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}(\mu - \mu_z), \mathbf{K}_{tt} - \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}\mathbf{K}_{zt} + \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}\mathbf{S} \mathbf{K}_{zz}^{-1}\mathbf{K}_{zt})

        with $\mu$ and $S$ computed from the expectation parameterisation
        $\eta = (\mu, S + uu^\top)$.

        Returns:
            GaussianDistribution: The predictive distribution of the GP at the
                test inputs $t$.
        """
        # Unpack variational parameters
        expectation_vector = _val(self.expectation_vector)
        expectation_matrix = _val(self.expectation_matrix)
        z = _val(self.inducing_inputs)

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # mu = eta_1
        mu = expectation_vector

        # S = eta_2 - eta_1 eta_1^T
        S = expectation_matrix - jnp.matmul(mu, mu.T)
        S = add_jitter(S, self.jitter)
        S_op = _psd(S)

        # S = sqrt sqrt^T
        sqrt = cholesky_factor(S_op)
        sqrt_matrix = sqrt.as_matrix()

        Kzz = kernel.gram(z)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)
        Lz = jnp.linalg.cholesky(Kzz_dense)
        muz = mean_function(z)

        # Unpack test inputs
        t = test_inputs

        Ktt = kernel.gram(t).as_matrix()
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz^{-1} Kzt
        Lz_inv_Kzt = _tri_solve(Lz, Kzt)

        # Kzz^{-1} Kzt
        Kzz_inv_Kzt = jsp.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)

        # Ktz Kzz^{-1} sqrt
        Ktz_Kzz_inv_sqrt = Kzz_inv_Kzt.T @ sqrt_matrix

        # mut + Ktz Kzz^{-1} (mu - muz)
        mean = mut + jnp.matmul(Kzz_inv_Kzt.T, mu - muz)

        # Ktt - Ktz Kzz^{-1} Kzt + Ktz Kzz^{-1} S Kzz^{-1} Kzt  [recall S = sqrt sqrt^T]
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.T)
        )
        covariance = add_jitter(covariance, self.jitter)
        covariance_op = lx.MatrixLinearOperator(covariance)

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance_op
        )


class CollapsedVariationalGaussian(AbstractVariationalGaussian[GL]):
    r"""Collapsed variational Gaussian.

    Collapsed variational Gaussian family of probability distributions.
    The key reference is Titsias, (2009) - Variational Learning of Inducing Variables
    in Sparse Gaussian Processes.

    The bound is *collapsed*: the variational parameters are solved for
    analytically, which requires a Gaussian likelihood and a pass over the full
    dataset. Lift either restriction and you need the uncollapsed bound of
    :class:`VariationalGaussian` instead.

    .. seealso::

        :doc:`/examples/collapsed_vi` works through the sparse regression setting
        this family is designed for.
    """

    def __init__(
        self,
        posterior: JointModel,
        inducing_inputs: Float[Array, "N D"],
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if not isinstance(posterior.likelihood, Gaussian):
            raise TypeError("Likelihood must be Gaussian.")

    def predict(
        self, test_inputs: Float[Array, "N D"], train_data: Dataset
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs $t$ at which to make
                predictions.
            train_data (Dataset): The training data that was used to fit the GP.

        Returns:
            GaussianDistribution: The predictive distribution of the collapsed
                variational Gaussian process at the test inputs $t$.
        """
        # Unpack test inputs
        t = test_inputs

        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack variational parameters
        noise_var = _val(self.posterior.likelihood.obs_stddev) ** 2
        z = _val(self.inducing_inputs)
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        Kzx = kernel.cross_covariance(z, x)
        Kzz = kernel.gram(z)
        Kzz_dense = add_jitter(Kzz.as_matrix(), self.jitter)

        # Lz Lz^T = Kzz
        Lz = jnp.linalg.cholesky(Kzz_dense)

        # Lz^{-1} Kzx
        Lz_inv_Kzx = _tri_solve(Lz, Kzx)

        # A = Lz^{-1} Kzt / o
        A = Lz_inv_Kzx / _val(self.posterior.likelihood.obs_stddev)

        # AA^T
        AAT = jnp.matmul(A, A.T)

        # LL^T = I + AA^T
        L = jnp.linalg.cholesky(jnp.eye(m) + AAT)

        mux = mean_function(x)
        diff = y - mux

        # Lz^{-1} Kzx (y - mux)
        Lz_inv_Kzx_diff = jsp.linalg.cho_solve((L, True), jnp.matmul(Lz_inv_Kzx, diff))

        # Kzz^{-1} Kzx (y - mux)
        Kzz_inv_Kzx_diff = jsp.linalg.solve_triangular(
            Lz.T, Lz_inv_Kzx_diff, lower=False
        )

        Ktt = kernel.gram(t).as_matrix()
        Kzt = kernel.cross_covariance(z, t)
        mut = mean_function(t)

        # Lz^{-1} Kzt
        Lz_inv_Kzt = _tri_solve(Lz, Kzt)

        # L^{-1} Lz^{-1} Kzt
        L_inv_Lz_inv_Kzt = jsp.linalg.solve_triangular(L, Lz_inv_Kzt, lower=True)

        # mut + 1/o^2 Ktz Kzz^{-1} Kzx (y - mux)
        mean = mut + jnp.matmul(Kzt.T / noise_var, Kzz_inv_Kzx_diff)

        # Ktt - Ktz Kzz^{-1} Kzt + Ktz Lz^{-1} (I + AA^T)^{-1} Lz^{-1} Kzt
        covariance = (
            Ktt
            - jnp.matmul(Lz_inv_Kzt.T, Lz_inv_Kzt)
            + jnp.matmul(L_inv_Lz_inv_Kzt.T, L_inv_Lz_inv_Kzt)
        )
        covariance = add_jitter(covariance, self.jitter)
        covariance_op = lx.MatrixLinearOperator(covariance)

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance_op
        )


@dataclass(slots=True)
class VariationalGaussianInit:
    """Initialization parameters for a variational Gaussian distribution."""

    inducing_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]]
    variational_mean: tp.Union[Float[Array, "N 1"], None] = None
    variational_root_covariance: tp.Union[Float[Array, "N N"], None] = None


class HeteroscedasticPrediction(tp.NamedTuple):
    """Mean and variance of the signal and noise latent processes."""

    mean_f: Float[Array, "N 1"]
    variance_f: Float[Array, "N 1"]
    mean_g: Float[Array, "N 1"]
    variance_g: Float[Array, "N 1"]


class HeteroscedasticVariationalFamily(AbstractVariationalFamily[HL]):
    r"""Variational family for two independent latent processes f and g."""

    signal_variational: tp.Any
    noise_variational: tp.Any
    jitter: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        posterior: HP,
        inducing_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]] = None,
        inducing_inputs_g: tp.Union[
            Int[Array, "M D"], Float[Array, "M D"], None
        ] = None,
        variational_mean_f: tp.Union[Float[Array, "N 1"], None] = None,
        variational_root_covariance_f: tp.Union[Float[Array, "N N"], None] = None,
        variational_mean_g: tp.Union[Float[Array, "M 1"], None] = None,
        variational_root_covariance_g: tp.Union[Float[Array, "M M"], None] = None,
        jitter: ScalarFloat = 1e-6,
        signal_init: tp.Optional[VariationalGaussianInit] = None,
        noise_init: tp.Optional[VariationalGaussianInit] = None,
    ):
        self.jitter = jitter

        if signal_init is not None:
            self.signal_variational = VariationalGaussian(
                posterior=posterior,
                inducing_inputs=signal_init.inducing_inputs,
                variational_mean=signal_init.variational_mean,
                variational_root_covariance=signal_init.variational_root_covariance,
                jitter=jitter,
            )
        elif inducing_inputs is not None:
            self.signal_variational = VariationalGaussian(
                posterior=posterior,
                inducing_inputs=inducing_inputs,
                variational_mean=variational_mean_f,
                variational_root_covariance=variational_root_covariance_f,
                jitter=jitter,
            )
        else:
            raise ValueError("Either signal_init or inducing_inputs must be provided.")

        if noise_init is not None:
            self.noise_variational = VariationalGaussian(
                posterior=posterior.noise_model,
                inducing_inputs=noise_init.inducing_inputs,
                variational_mean=noise_init.variational_mean,
                variational_root_covariance=noise_init.variational_root_covariance,
                jitter=jitter,
            )
        else:
            noise_inducing = (
                inducing_inputs if inducing_inputs_g is None else inducing_inputs_g
            )
            if noise_inducing is None and signal_init is not None:
                noise_inducing = signal_init.inducing_inputs

            if noise_inducing is None:
                raise ValueError(
                    "Could not determine inducing inputs for noise process."
                )

            self.noise_variational = VariationalGaussian(
                posterior=posterior.noise_model,
                inducing_inputs=noise_inducing,
                variational_mean=variational_mean_g,
                variational_root_covariance=variational_root_covariance_g,
                jitter=jitter,
            )
        super().__init__(posterior)

    def prior_kl(self) -> ScalarFloat:
        return self.signal_variational.prior_kl() + self.noise_variational.prior_kl()

    def predict(
        self, test_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]]
    ) -> HeteroscedasticPrediction:
        dist_f = self.signal_variational.predict(test_inputs)
        dist_g = self.noise_variational.predict(test_inputs)

        mean_f = dist_f.mean[:, None] if dist_f.mean.ndim == 1 else dist_f.mean
        var_f = (
            dist_f.variance[:, None] if dist_f.variance.ndim == 1 else dist_f.variance
        )
        mean_g = dist_g.mean[:, None] if dist_g.mean.ndim == 1 else dist_g.mean
        var_g = (
            dist_g.variance[:, None] if dist_g.variance.ndim == 1 else dist_g.variance
        )

        return HeteroscedasticPrediction(
            mean_f=mean_f,
            variance_f=var_f,
            mean_g=mean_g,
            variance_g=var_g,
        )

    def predict_latents(
        self, test_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]]
    ) -> tuple[GaussianDistribution, GaussianDistribution]:
        return (
            self.signal_variational.predict(test_inputs),
            self.noise_variational.predict(test_inputs),
        )


__all__ = [
    "AbstractVariationalFamily",
    "AbstractVariationalGaussian",
    "CollapsedVariationalGaussian",
    "ExpectationVariationalGaussian",
    "GraphVariationalGaussian",
    "HeteroscedasticPrediction",
    "HeteroscedasticVariationalFamily",
    "NaturalVariationalGaussian",
    "VariationalGaussian",
    "VariationalGaussianInit",
    "WhitenedVariationalGaussian",
]
