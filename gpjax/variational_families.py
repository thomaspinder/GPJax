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


def _tri_solve(L, B):
    """Solve L x = B where L is lower triangular. Works for matrix B."""
    return jsp.linalg.solve_triangular(L, B, lower=True)


def _symmetrise(matrix):
    """Return ``(matrix + matrix.T) / 2``."""
    return 0.5 * (matrix + matrix.T)


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


class DualVariationalGaussian(AbstractVariationalGaussian[L]):
    r"""The dual (site) parameterisation of a sparse variational Gaussian process.

    Following the t-SVGP parameterisation of Adam, Chang, Khan and Solin (2021),
    [arXiv:2111.03412](https://arxiv.org/abs/2111.03412), the variational distribution
    is stored as an unnormalised Gaussian *site* on the inducing outputs rather than as
    moments:
    ```math
    t(u) = \exp\left(\lambda_1^{\top}\tilde u
        - \tfrac{1}{2}\tilde u^{\top}\Lambda_2\tilde u\right),
    \qquad q(u) \propto p_{\theta}(u)\,t(u),
    ```
    with $\tilde u = u - \mu_z$ the *centred* inducing outputs, giving
    ```math
    S = \left(\mathbf{K}_{zz}^{-1} + \Lambda_2\right)^{-1},
    \qquad \tilde m = S\lambda_1, \qquad m = \mu_z + \tilde m .
    ```
    ``dual_vector`` is $\lambda_1\in\mathbb{R}^{M\times1}$ and ``dual_matrix`` is
    $\Lambda_2\in\mathbb{R}^{M\times M}$, stored in the **precision** convention (PSD,
    no $-\tfrac{1}{2}$ factor). Both default to zero, so that $q(u) = p(u)$ and the
    prior KL vanishes at initialisation.

    Neither field is wrapped in a constraining bijection: positive semi-definiteness
    of $\Lambda_2$ comes from the convex-combination structure of the natural-gradient
    update, and a bijection here would destroy that affine step.

    Everything routes through the working matrix
    $\mathbf{R} = \mathbf{K}_{zz} + \mathbf{K}_{zz}\Lambda_2\mathbf{K}_{zz}
    = \mathbf{K}_{zz}\mathbf{S}^{-1}\mathbf{K}_{zz}$, so no matrix is ever explicitly
    inverted and $\Lambda_2$ is never factorised.

    Example:
    ```pycon
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import gpjax as gpx
        >>>
        >>> prior = gpx.gps.Prior(
        ...     mean_function=gpx.mean_functions.Constant(), kernel=gpx.kernels.RBF()
        ... )
        >>> posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=10)
        >>> q = gpx.variational_families.DualVariationalGaussian(
        ...     posterior=posterior, inducing_inputs=jnp.linspace(0, 1, 4).reshape(-1, 1)
        ... )
        >>> bool(abs(q.prior_kl()) < 1e-10)
        True
    ```
    """

    dual_vector: tp.Any
    dual_matrix: tp.Any

    def __init__(
        self,
        posterior: AbstractPosterior[P, L],
        inducing_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]],
        dual_vector: tp.Union[Float[Array, "N 1"], None] = None,
        dual_matrix: tp.Union[Float[Array, "N N"], None] = None,
        jitter: ScalarFloat = 1e-6,
    ):
        super().__init__(posterior, inducing_inputs, jitter)

        if dual_vector is None:
            dual_vector = jnp.zeros((self.num_inducing, 1))

        if dual_matrix is None:
            dual_matrix = jnp.zeros((self.num_inducing, self.num_inducing))

        self.dual_vector = Real(dual_vector)
        self.dual_matrix = Real(dual_matrix)

    def _fmt_Kzt_Ktt(self, Kzt, Ktt):
        return Kzt, Ktt

    def _fmt_inducing_inputs(self):
        return _val(self.inducing_inputs)

    def _working_matrices(
        self,
    ) -> tuple[Float[Array, "M M"], Float[Array, "M M"], Float[Array, "M M"]]:
        r"""Return $(\mathbf{K}_{zz},\ \mathbf{L}_K,\ \mathbf{L}_R)$.

        With $\mathbf{R} = \operatorname{sym}(\mathbf{K}_{zz}
        + \mathbf{K}_{zz}\Lambda_2\mathbf{K}_{zz})
        = \mathbf{K}_{zz}\mathbf{S}^{-1}\mathbf{K}_{zz}
        \succeq \mathbf{K}_{zz} \succ 0$ whenever $\Lambda_2\succeq0$, the Cholesky of
        $\mathbf{R}$ never fails -- even when $\Lambda_2$ is rank deficient, which it
        is at initialisation ($\Lambda_2=0$) and whenever the batch is smaller than the
        number of inducing points. Exactly **two** Cholesky factorisations per call.

        Returns
        -------
        tuple[Float[Array, "M M"], Float[Array, "M M"], Float[Array, "M M"]]
            The jittered gram matrix and the lower-triangular Cholesky factors of
            $\mathbf{K}_{zz}$ and $\mathbf{R}$.
        """
        inducing_inputs = self._fmt_inducing_inputs()
        kernel = self.posterior.prior.kernel

        Kzz = add_jitter(kernel.gram(inducing_inputs).as_matrix(), self.jitter)
        Lk = jnp.linalg.cholesky(Kzz)

        dual_matrix = _val(self.dual_matrix)
        R = _symmetrise(Kzz + Kzz @ dual_matrix @ Kzz)
        Lr = jnp.linalg.cholesky(R)
        return Kzz, Lk, Lr

    def moments(self) -> tuple[Float[Array, "M 1"], Float[Array, "M M"]]:
        r"""Return the implied moments $(\mathbf{m},\mathbf{S})$ of $q(u)$.

        ```math
        \mathbf{S} = \mathbf{K}_{zz}\mathbf{R}^{-1}\mathbf{K}_{zz},
        \qquad
        \mathbf{m} = \mu_z
            + \mathbf{K}_{zz}\mathbf{R}^{-1}\left(\mathbf{K}_{zz}\lambda_1\right).
        ```

        For reporting and for interoperating with :class:`VariationalGaussian`; the
        training path never needs it. Nothing here is cached on the module: caching
        would silently turn :func:`~gpjax.objectives.dual_elbo` back into
        :func:`~gpjax.objectives.elbo` under differentiation.

        Returns
        -------
        tuple[Float[Array, "M 1"], Float[Array, "M M"]]
            The mean and covariance of $q(u)$.
        """
        Kzz, _, Lr = self._working_matrices()
        dual_vector = _val(self.dual_vector)
        inducing_mean = self.posterior.prior.mean_function(self._fmt_inducing_inputs())

        covariance = _symmetrise(Kzz @ jsp.linalg.cho_solve((Lr, True), Kzz))
        centred_mean = Kzz @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)
        return inducing_mean + centred_mean, covariance

    def marginals(
        self, inputs: Float[Array, "P D"]
    ) -> tuple[Float[Array, " P"], Float[Array, " P"]]:
        r"""Batched marginal mean and variance of $q(f(\cdot))$ at ``inputs``.

        ```math
        \mu_{\star} = \mu(X_{\star})
            + \mathbf{K}_{\star z}\mathbf{R}^{-1}\mathbf{K}_{zz}\lambda_1,
        \qquad
        \sigma^2_{\star} = \operatorname{diag}(\mathbf{K}_{\star\star})
            - \lVert\mathbf{L}_K^{-1}\mathbf{K}_{z\star}\rVert^2_{\mathrm{col}}
            + \lVert\mathbf{L}_R^{-1}\mathbf{K}_{z\star}\rVert^2_{\mathrm{col}}
            + \varepsilon .
        ```

        Costs $\mathcal{O}(M^3 + PM^2)$, against the $\mathcal{O}(PM^3)$ of ``vmap``-ing
        :meth:`predict` over single points, which would rebuild $\mathbf{R}$ and its
        Cholesky once per datum.

        Returns
        -------
        tuple[Float[Array, " P"], Float[Array, " P"]]
            The marginal mean and variance at each input.

        Notes
        -----
        The trailing ``+ self.jitter`` on the variance is load-bearing, not a numerical
        nicety. :meth:`VariationalGaussian.predict` runs ``add_jitter`` on its output
        covariance, so the per-point marginals that :func:`~gpjax.objectives.elbo` sees
        are inflated by exactly ``self.jitter``. Dropping it here makes
        :func:`~gpjax.objectives.dual_elbo` disagree with
        :func:`~gpjax.objectives.elbo` at matched moments by
        $N\varepsilon/(2\sigma^2)$ -- a discrepancy that reads like a KL bug.
        """
        Kzz, Lk, Lr = self._working_matrices()
        dual_vector = _val(self.dual_vector)
        kernel = self.posterior.prior.kernel
        mean_function = self.posterior.prior.mean_function
        inducing_inputs = self._fmt_inducing_inputs()

        Kzs = kernel.cross_covariance(inducing_inputs, inputs)
        Kss_diagonal = lx.diagonal(kernel.diagonal(inputs))

        mean = mean_function(inputs).squeeze(-1) + (
            Kzs.T @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)
        ).squeeze(-1)

        prior_projection = _tri_solve(Lk, Kzs)
        site_projection = _tri_solve(Lr, Kzs)
        variance = (
            Kss_diagonal
            - jnp.sum(jnp.square(prior_projection), axis=0)
            + jnp.sum(jnp.square(site_projection), axis=0)
            + self.jitter
        )
        return mean, variance

    def prior_kl(self) -> ScalarFloat:
        r"""Compute $\operatorname{KL}[q(u)\mid\mid p(u)]$ from the stored sites.

        ```math
        \operatorname{KL} = \tfrac{1}{2}\left(
            \operatorname{tr}\left(\mathbf{R}^{-1}\mathbf{K}_{zz}\right) - M
            + \tilde m^{\top}\mathbf{K}_{zz}^{-1}\tilde m
            + \log\lvert\mathbf{R}\rvert - \log\lvert\mathbf{K}_{zz}\rvert
        \right),
        ```
        obtained from the standard Gaussian KL by substituting
        $\mathbf{S} = \mathbf{K}_{zz}\mathbf{R}^{-1}\mathbf{K}_{zz}$, which gives
        $\mathbf{K}_{zz}^{-1}\mathbf{S} = \mathbf{R}^{-1}\mathbf{K}_{zz}$ and
        $\log\lvert\mathbf{S}\rvert
        = 2\log\lvert\mathbf{K}_{zz}\rvert - \log\lvert\mathbf{R}\rvert$. The two
        log-determinants are read off the Cholesky diagonals; no matrix is inverted and
        $\mathbf{S}$ is never formed.

        Returns
        -------
        ScalarFloat
            The KL divergence between the variational approximation and the GP prior.
        """
        Kzz, Lk, Lr = self._working_matrices()
        dual_vector = _val(self.dual_vector)

        trace = jnp.trace(jsp.linalg.cho_solve((Lr, True), Kzz))

        # The sites act on the centred process, so the Mahalanobis term is built from
        # the centred mean and the zero-mean prior N(0, Kzz).
        centred_mean = Kzz @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)
        mahalanobis = jnp.sum(jnp.square(_tri_solve(Lk, centred_mean)))

        log_det_ratio = 2.0 * (
            jnp.sum(jnp.log(jnp.diag(Lr))) - jnp.sum(jnp.log(jnp.diag(Lk)))
        )
        return 0.5 * (trace - self.num_inducing + mahalanobis + log_det_ratio)

    def predict(
        self, test_inputs: tp.Union[Int[Array, "N D"], Float[Array, "N D"]]
    ) -> GaussianDistribution:
        r"""Compute the predictive distribution of the GP at the test inputs t.

        ```math
            \mathcal{N}\left(f(t);\ \mu_t
            + \mathbf{K}_{tz}\mathbf{R}^{-1}\mathbf{K}_{zz}\lambda_1,\
            \mathbf{K}_{tt} - \mathbf{K}_{tz}\mathbf{K}_{zz}^{-1}\mathbf{K}_{zt}
            + \mathbf{K}_{tz}\mathbf{R}^{-1}\mathbf{K}_{zt}\right).
        ```

        Because $\mathbf{R}\succeq\mathbf{K}_{zz}$ implies
        $\mathbf{R}^{-1}\preceq\mathbf{K}_{zz}^{-1}$, the predictive covariance can
        never exceed the prior covariance -- that is structural here, not something the
        jitter has to enforce.

        Args:
            test_inputs (Float[Array, "N D"]): The test inputs at which we wish to
                make a prediction.

        Returns:
            GaussianDistribution: The predictive distribution of the low-rank GP at
                the test inputs.
        """
        Kzz, Lk, Lr = self._working_matrices()
        dual_vector = _val(self.dual_vector)
        kernel = self.posterior.prior.kernel
        mean_function = self.posterior.prior.mean_function
        inducing_inputs = self._fmt_inducing_inputs()

        t = test_inputs
        Ktt = kernel.gram(t).as_matrix()
        Kzt = kernel.cross_covariance(inducing_inputs, t)
        mut = mean_function(t)

        Kzt, Ktt = self._fmt_Kzt_Ktt(Kzt, Ktt)

        # Lk^{-1} Kzt and Lr^{-1} Kzt
        prior_projection = _tri_solve(Lk, Kzt)
        site_projection = _tri_solve(Lr, Kzt)

        mean = mut + Kzt.T @ jsp.linalg.cho_solve((Lr, True), Kzz @ dual_vector)

        covariance = (
            Ktt
            - jnp.matmul(prior_projection.T, prior_projection)
            + jnp.matmul(site_projection.T, site_projection)
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
    "DualVariationalGaussian",
    "GraphVariationalGaussian",
    "HeteroscedasticPrediction",
    "HeteroscedasticVariationalFamily",
    "VariationalGaussian",
    "VariationalGaussianInit",
    "WhitenedVariationalGaussian",
]
