"""Orthogonal Instantaneous Linear Mixing Model (OILMM) for multi-output GPs.

OILMM achieves O(n³m) complexity instead of O(n³m³) by constraining the mixing
matrix to have orthogonal columns, which causes the projected noise to be
diagonal and enables inference to decompose into m independent single-output
GP problems.

Reference:
    Bruinsma et al. (2020). "Scalable Exact Inference in Multi-Output Gaussian
    Processes." ICML.
"""

from __future__ import annotations

import copy
import typing as tp

from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from gpjax.distributions import GaussianDistribution
from gpjax.linalg import Dense
from gpjax.linalg.utils import psd
from gpjax.parameters import NonNegativeReal, PositiveReal, Real
from gpjax.typing import ScalarFloat

if tp.TYPE_CHECKING:
    from gpjax.dataset import Dataset
    from gpjax.kernels.base import AbstractKernel


class OrthogonalMixingMatrix(nnx.Module):
    """Mixing matrix H = U S^(1/2) with orthogonal columns.

    Parameterizes an orthogonal mixing matrix for OILMM where:
    - U ∈ ℝ^(p×m) has orthonormal columns (U^T U = I_m)
    - S > 0 is a diagonal scaling matrix (m × m)
    - H = U S^(1/2) is the mixing matrix
    - T = S^(-1/2) U^T is the projection matrix

    The orthogonality of U ensures that the projected noise is diagonal:
        Σ_T = T Σ T^T = σ²S^(-1) + D
    where σ² is observation noise and D is latent noise.

    Attributes:
        num_outputs: Number of output dimensions (p)
        num_latent_gps: Number of latent GP functions (m)
        U_latent: Unconstrained matrix for SVD orthogonalization
        S: Positive diagonal scaling
        obs_noise_variance: Homogeneous observation noise (σ²)
        latent_noise_variance: Per-latent heterogeneous noise (D), non-negative
    """

    def __init__(
        self,
        num_outputs: int,
        num_latent_gps: int,
        key: Array,
    ):
        """Initialize orthogonal mixing matrix.

        Args:
            num_outputs: Number of output dimensions (p)
            num_latent_gps: Number of latent GPs (m), must satisfy m ≤ p
            key: JAX PRNG key for initialization
        """
        if num_latent_gps > num_outputs:
            raise ValueError(
                f"num_latent_gps ({num_latent_gps}) must be ≤ "
                f"num_outputs ({num_outputs})"
            )

        self.num_outputs = num_outputs
        self.num_latent_gps = num_latent_gps

        # Unconstrained latent representation (small init for stability)
        self.U_latent = Real(jr.normal(key, (num_outputs, num_latent_gps)) * 0.1)

        # Scaling diagonal (init to 1)
        self.S = PositiveReal(jnp.ones(num_latent_gps))

        # Noise parameters
        # obs_noise_variance is strictly positive (σ² > 0)
        self.obs_noise_variance = PositiveReal(jnp.array(1.0))
        # latent_noise_variance (D) can be zero — use NonNegativeReal
        self.latent_noise_variance = NonNegativeReal(jnp.zeros(num_latent_gps))

    @property
    def U(self) -> Float[Array, "P M"]:
        """Orthonormal columns via SVD.

        Uses SVD to project U_latent onto the Stiefel manifold (orthonormal columns).
        This ensures U^T U = I_m exactly.
        """
        U_svd, _, Vt_svd = jnp.linalg.svd(self.U_latent[...], full_matrices=False)
        return U_svd @ Vt_svd

    @property
    def sqrt_S(self) -> Float[Array, " M"]:
        """Square root of S diagonal: S^(1/2)."""
        return jnp.sqrt(self.S[...])

    @property
    def inv_sqrt_S(self) -> Float[Array, " M"]:
        """Inverse square root of S diagonal: S^(-1/2)."""
        return 1.0 / jnp.sqrt(self.S[...])

    @property
    def H(self) -> Float[Array, "P M"]:
        """Mixing matrix H = U S^(1/2).

        Maps from latent space (m dimensions) to output space (p dimensions).
        Each column is an orthogonal basis vector scaled by sqrt(S_i).
        """
        return self.U * self.sqrt_S[None, :]

    @property
    def T(self) -> Float[Array, "M P"]:
        """Projection matrix T = S^(-1/2) U^T.

        Projects from output space (p dimensions) to latent space (m dimensions).
        This is the left pseudo-inverse of H: T @ H = I_m.
        """
        return self.inv_sqrt_S[:, None] * self.U.T

    @property
    def H_squared(self) -> Float[Array, "P M"]:
        """Element-wise H² for fast diagonal variance reconstruction.

        When computing marginal variances, we need H² @ latent_vars:
            var_p = sum_m H²_pm * var_m
        This property caches H² to avoid recomputation.
        """
        return self.H**2

    @property
    def projected_noise_variance(self) -> Float[Array, " M"]:
        """Diagonal projected noise: Σ_T = σ²S^(-1) + D.

        This is the noise variance for each independent latent GP after projection.
        The orthogonality of U ensures this is diagonal, which is what makes
        OILMM tractable.

        Returns:
            Array of shape [M] with noise variance for each latent GP.
        """
        return (
            self.obs_noise_variance[...] * self.inv_sqrt_S**2
            + self.latent_noise_variance[...]
        )


class OILMMModel(nnx.Module):
    """Orthogonal Instantaneous Linear Mixing Model.

    OILMM decomposes multi-output GP inference into M independent single-output
    GP problems by using an orthogonal mixing matrix. This achieves O(n³m)
    complexity instead of O(n³m³).

    The generative model is:
        x_i ~ GP(0, K(t,t'))          for i=1..M (latent GPs)
        f(t) = H x(t)                  (mixing)
        y | f ~ N(f(t), Σ)             (noise: Σ = σ²I + HDH^T)

    The orthogonality constraint (U^T U = I) ensures the projected noise is diagonal:
        Σ_T = T Σ T^T = σ²S^(-1) + D
    enabling independent inference for each latent GP.

    Attributes:
        num_outputs: Number of output dimensions (p)
        num_latent_gps: Number of latent GPs (m)
        mixing_matrix: OrthogonalMixingMatrix containing H, T, noise params
        latent_priors: Tuple of M independent Prior objects
    """

    def __init__(
        self,
        num_outputs: int,
        num_latent_gps: int,
        kernel: AbstractKernel | list[AbstractKernel],
        key: Array,
        mean_function: tp.Any = None,
    ):
        """Initialize OILMM model.

        Args:
            num_outputs: Number of output dimensions (p)
            num_latent_gps: Number of latent GPs (m), must satisfy m ≤ p
            kernel: Kernel for latent GPs. If a single kernel, it is deep-copied
                M times so each latent GP has independent hyperparameters. If a
                list of M kernels, each is used directly.
            key: JAX PRNG key
            mean_function: Mean function for latent GPs (default: Zero)
        """
        from gpjax.gps import Prior
        from gpjax.mean_functions import Zero

        self.num_outputs = num_outputs
        self.num_latent_gps = num_latent_gps

        # Orthogonal mixing matrix
        key, subkey = jr.split(key)
        self.mixing_matrix = OrthogonalMixingMatrix(
            num_outputs=num_outputs,
            num_latent_gps=num_latent_gps,
            key=subkey,
        )

        # Mean function (shared across latents)
        if mean_function is None:
            mean_function = Zero()

        # Build per-latent kernel list
        if isinstance(kernel, list):
            if len(kernel) != num_latent_gps:
                raise ValueError(
                    f"Expected {num_latent_gps} kernels, got {len(kernel)}"
                )
            kernels = kernel
        else:
            kernels = [copy.deepcopy(kernel) for _ in range(num_latent_gps)]

        self.latent_priors = nnx.List(
            [Prior(kernel=k, mean_function=mean_function) for k in kernels]
        )

    def _project_observations(
        self, dataset: Dataset
    ) -> tuple[Float[Array, "N D"], Float[Array, "M N"]]:
        """Project observations to latent space: y_latent = T @ y.

        This is the first phase of OILMM inference. The projection is cheap (O(nmp))
        and transforms the multi-output problem into M single-output problems.

        Args:
            dataset: Training data with X [N, D] and y [N, P]

        Returns:
            Tuple of (X, y_projected) where:
                - X: Input locations [N, D] (unchanged)
                - y_projected: Projected observations [M, N]
        """
        T = self.mixing_matrix.T  # [M, P]
        y_projected = T @ dataset.y.T  # [M, P] @ [P, N] = [M, N]
        return dataset.X, y_projected

    def condition_on_observations(self, dataset: Dataset) -> OILMMPosterior:
        """Condition on observations to create posterior.

        This implements the core OILMM inference algorithm:
        1. Project observations: y_latent = T @ y
        2. Condition M independent GPs on projected data
        3. Return OILMMPosterior wrapping the M posteriors

        Args:
            dataset: Training data with X [N, D] and y [N, P]

        Returns:
            OILMMPosterior containing M independent posteriors
        """
        from gpjax.dataset import Dataset
        from gpjax.likelihoods import Gaussian

        # Phase 1: Project observations (O(nmp))
        X, y_projected = self._project_observations(dataset)  # [N, D], [M, N]

        # Phase 2: Get projected noise variances
        projected_noise_vars = self.mixing_matrix.projected_noise_variance  # [M]

        # Phase 3: Condition each latent GP independently.
        # NOTE: We use a Python loop rather than jax.vmap because each
        # latent Prior/Posterior is an nnx.Module with independent state.
        latent_posteriors = []
        latent_datasets = []
        for i in range(self.num_latent_gps):
            # Create dataset for this latent GP
            latent_dataset = Dataset(X=X, y=y_projected[i][:, None])  # [N, 1]
            latent_datasets.append(latent_dataset)

            # Create likelihood with projected noise
            likelihood = Gaussian(
                num_datapoints=dataset.n,
                obs_stddev=jnp.sqrt(projected_noise_vars[i]),
            )

            # Standard GPJax conditioning: Prior * Likelihood -> ConjugatePosterior
            latent_posteriors.append(self.latent_priors[i] * likelihood)

        return OILMMPosterior(
            latent_posteriors=tuple(latent_posteriors),
            latent_datasets=tuple(latent_datasets),
            mixing_matrix=self.mixing_matrix,
        )


class OILMMPosterior:
    """Posterior distribution for OILMM.

    Wraps M independent ConjugatePosterior objects and provides a unified
    predict() interface that reconstructs predictions in output space.

    This is a plain class (not nnx.Module) because it holds Dataset objects
    which are not JAX pytree nodes. The latent posteriors and mixing matrix
    are still nnx.Modules and participate in JAX transformations when accessed.

    Attributes:
        latent_posteriors: Tuple of M independent ConjugatePosterior objects
        latent_datasets: Tuple of M projected training Datasets (one per latent GP)
        mixing_matrix: OrthogonalMixingMatrix for reconstruction
        num_latent_gps: Number of latent GPs (m)
    """

    def __init__(
        self,
        latent_posteriors: tuple,
        latent_datasets: tuple,
        mixing_matrix: OrthogonalMixingMatrix,
    ):
        """Initialize OILMM posterior.

        Args:
            latent_posteriors: Tuple of M ConjugatePosterior objects
            latent_datasets: Tuple of M Dataset objects (projected training data)
            mixing_matrix: OrthogonalMixingMatrix containing H, T
        """
        self.latent_posteriors = latent_posteriors
        self.latent_datasets = latent_datasets
        self.mixing_matrix = mixing_matrix
        self.num_latent_gps = len(latent_posteriors)

    def predict(
        self,
        test_inputs: Float[Array, "N D"],
        return_full_cov: bool = True,
    ) -> GaussianDistribution:
        """Predict at test locations.

        Reconstructs predictions in output space from M independent latent posteriors:
        1. Predict each latent GP independently
        2. Reconstruct mean: f_mean = H @ latent_means
        3. Reconstruct covariance: Σ_f = (H ⊗ I) Σ_x (H ⊗ I)^T

        Args:
            test_inputs: Test input locations [N, D]
            return_full_cov: If True, return full [NP, NP] covariance.
                           If False, return diagonal covariance matrix.

        Returns:
            GaussianDistribution with:
                - loc: [NP] flattened output-major
                - scale: Dense [NP, NP] covariance (full or diagonal)
        """
        N = test_inputs.shape[0]
        H = self.mixing_matrix.H  # [P, M]
        H_squared = self.mixing_matrix.H_squared  # [P, M]

        # Phase 1: Predict each latent GP independently.
        # NOTE: Python loop — cannot vmap over nnx.Module instances.
        latent_preds = [
            post.predict(test_inputs, ds)
            for post, ds in zip(
                self.latent_posteriors, self.latent_datasets, strict=True
            )
        ]
        latent_means = jnp.array([pred.mean for pred in latent_preds])  # [M, N]
        latent_covs = [pred.covariance() for pred in latent_preds]  # M × [N, N]

        # Phase 2: Reconstruct mean
        f_mean = jnp.einsum("pm,mn->pn", H, latent_means)  # [P, N]
        f_mean_flat = f_mean.T.ravel()  # [N*P] output-major

        # Phase 3: Reconstruct covariance
        # Use plain Python if/else (not jax.lax.cond) because return_full_cov
        # is a Python bool that should not be traced by JAX.
        if return_full_cov:
            # Full covariance via block structure:
            # Cov[p1,p2] = Σ_m H[p1,m] H[p2,m] Σ_latent_m
            latent_covs_stacked = jnp.stack(latent_covs)  # [M, N, N]
            f_cov_blocks = jnp.einsum(
                "pm,qm,mij->pqij", H, H, latent_covs_stacked
            )  # [P, P, N, N]
            P = self.mixing_matrix.num_outputs
            # Reorder to [N, P, N, P] so flattening matches f_mean.T.ravel().
            f_cov = f_cov_blocks.transpose(2, 0, 3, 1).reshape(N * P, N * P)  # [NP, NP]
        else:
            # Diagonal-only covariance for efficiency
            latent_vars = jnp.array([jnp.diag(cov) for cov in latent_covs])  # [M, N]
            f_vars = jnp.einsum("pm,mn->pn", H_squared, latent_vars)  # [P, N]
            f_vars_flat = f_vars.T.ravel()  # [N*P]
            f_cov = jnp.diag(f_vars_flat)

        return GaussianDistribution(
            loc=jnp.atleast_1d(f_mean_flat.squeeze()),
            scale=psd(Dense(f_cov)),
        )


def oilmm_mll(model: OILMMModel, data: Dataset) -> ScalarFloat:
    """Log marginal likelihood for the OILMM.

    Implements Prop. 9 from Bruinsma et al. (2020):

        log p(Y) = correction_terms + Σᵢ log N((TY)ᵢ | 0, Kᵢ + noise_i Iₙ)

    The correction terms prevent the projection from collapsing and account
    for data in the (p - m) dimensions orthogonal to the mixing matrix.

    Args:
        model: OILMMModel with parameters to evaluate.
        data: Training data with X [N, D] and y [N, P].

    Returns:
        Scalar log marginal likelihood.
    """
    from gpjax.linalg.utils import add_jitter

    n = data.n
    p = model.num_outputs
    m = model.num_latent_gps
    mix = model.mixing_matrix

    U = mix.U  # [P, M]
    S = mix.S[...]  # [M]
    sigma2 = mix.obs_noise_variance[...]  # scalar

    # --- Correction term 1: -(n/2) log|S| ---
    # |S| = prod(S_i), so log|S| = sum(log(S_i))
    term_log_S = -0.5 * n * jnp.sum(jnp.log(S))

    # --- Correction term 2: -n(p-m)/2 log(2πσ²) ---
    term_noise = -0.5 * n * (p - m) * jnp.log(2.0 * jnp.pi * sigma2)

    # --- Correction term 3: -(1/(2σ²)) ||(I_p - UU^T)Y||_F² ---
    # Residual = Y - U(U^T Y), computed without forming the P×P projector.
    Y = data.y  # [N, P]
    UtY = U.T @ Y.T  # [M, N]
    projected = U @ UtY  # [P, N]
    residual = Y.T - projected  # [P, N]
    frob_sq = jnp.sum(residual**2)
    term_residual = -0.5 * frob_sq / sigma2

    correction = term_log_S + term_noise + term_residual

    # --- Latent GP log-likelihoods computed directly ---
    # We compute each latent GP's MLL inline to avoid constructing Gaussian
    # likelihood objects, which would trigger parameter validation checks
    # that are incompatible with JAX's JIT tracing.
    X, y_projected = model._project_observations(data)  # [N, D], [M, N]
    projected_noise_vars = mix.projected_noise_variance  # [M]

    latent_lls = []
    for i in range(m):
        yi = y_projected[i]  # [N]
        prior_i = model.latent_priors[i]
        mx = prior_i.mean_function(X).squeeze()  # [N]
        Kxx = prior_i.kernel.gram(X).to_dense()  # [N, N]
        Kxx = add_jitter(Kxx, prior_i.jitter)
        Sigma = Kxx + projected_noise_vars[i] * jnp.eye(n)
        dist = GaussianDistribution(jnp.atleast_1d(mx), psd(Dense(Sigma)))
        latent_lls.append(dist.log_prob(jnp.atleast_1d(yi)))

    return correction + jnp.sum(jnp.array(latent_lls))


# Convenience constructors


def create_oilmm(
    num_outputs: int,
    num_latent_gps: int,
    key: Array,
    kernel: AbstractKernel | list[AbstractKernel] | None = None,
    mean_function: tp.Any = None,
) -> OILMMModel:
    """Create OILMM model with shared kernel across latents.

    Args:
        num_outputs: Number of output dimensions (p)
        num_latent_gps: Number of latent GPs (m)
        key: JAX PRNG key
        kernel: Kernel for latent GPs (default: RBF)
        mean_function: Mean function for latent GPs (default: Zero)

    Returns:
        Initialized OILMMModel

    Example:
        >>> import gpjax as gpx
        >>> import jax.random as jr
        >>> model = gpx.models.create_oilmm(
        ...     num_outputs=5,
        ...     num_latent_gps=2,
        ...     key=jr.key(42),
        ...     kernel=gpx.kernels.Matern52()
        ... )
    """
    from gpjax.kernels.stationary import RBF

    if kernel is None:
        kernel = RBF()

    return OILMMModel(
        num_outputs=num_outputs,
        num_latent_gps=num_latent_gps,
        kernel=kernel,
        key=key,
        mean_function=mean_function,
    )


def create_oilmm_with_kernels(
    latent_kernels: list[AbstractKernel],
    num_outputs: int,
    key: Array,
    mean_function: tp.Any = None,
) -> OILMMModel:
    """Create OILMM with custom kernel per latent GP.

    Args:
        latent_kernels: List of M kernels, one per latent GP
        num_outputs: Number of output dimensions (p)
        key: JAX PRNG key
        mean_function: Mean function (shared, default: Zero)

    Returns:
        OILMMModel with heterogeneous latent kernels

    Example:
        >>> import gpjax as gpx
        >>> import jax.random as jr
        >>> model = gpx.models.create_oilmm_with_kernels(
        ...     latent_kernels=[gpx.kernels.RBF(), gpx.kernels.Matern52()],
        ...     num_outputs=6,
        ...     key=jr.key(42)
        ... )
    """
    import warnings

    warnings.warn(
        "create_oilmm_with_kernels is deprecated. Pass a list of kernels "
        "directly to OILMMModel or create_oilmm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return OILMMModel(
        num_outputs=num_outputs,
        num_latent_gps=len(latent_kernels),
        kernel=latent_kernels,
        key=key,
        mean_function=mean_function,
    )


def create_oilmm_from_data(
    dataset: Dataset,
    num_latent_gps: int,
    key: Array,
    kernel: AbstractKernel = None,
    mean_function: tp.Any = None,
) -> OILMMModel:
    """Create OILMM with data-informed initialization of mixing matrix.

    Initializes U to the top M eigenvectors and S to the top M eigenvalues of
    the empirical covariance matrix of the outputs. Near-zero eigenvalues are
    clamped to 1e-6 for numerical stability. This can provide better
    initialization than random, especially when outputs have clear correlation
    structure.

    Args:
        dataset: Training data with y [N, P]
        num_latent_gps: Number of latent GPs (m)
        key: JAX PRNG key
        kernel: Kernel for latent GPs (default: RBF)
        mean_function: Mean function (default: Zero)

    Returns:
        OILMMModel with U initialized to top M eigenvectors and S to
        top M eigenvalues

    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> X = jnp.linspace(0, 1, 50).reshape(-1, 1)
        >>> y = jnp.column_stack([jnp.sin(X), jnp.cos(X)])
        >>> data = gpx.Dataset(X=X, y=y)
        >>> model = gpx.models.create_oilmm_from_data(
        ...     dataset=data,
        ...     num_latent_gps=2,
        ...     key=jr.key(42)
        ... )
    """
    from gpjax.kernels.stationary import RBF

    num_outputs = dataset.y.shape[1]

    if kernel is None:
        kernel = RBF()

    # Create base model
    model = OILMMModel(
        num_outputs=num_outputs,
        num_latent_gps=num_latent_gps,
        kernel=kernel,
        key=key,
        mean_function=mean_function,
    )

    # Compute empirical covariance of outputs
    y_centered = dataset.y - dataset.y.mean(axis=0, keepdims=True)
    emp_cov = y_centered.T @ y_centered / dataset.n  # [P, P]

    # Get top M eigenvectors
    eigvals, eigvecs = jnp.linalg.eigh(emp_cov)
    # Sort descending
    idx = jnp.argsort(eigvals)[::-1]
    top_m_eigvecs = eigvecs[:, idx[:num_latent_gps]]  # [P, M]
    top_m_eigvals = jnp.maximum(eigvals[idx[:num_latent_gps]], 1e-6)

    # Initialize U_latent such that U will be close to these eigenvectors
    # Since U = U_svd @ V^T from SVD(U_latent), we can just set U_latent = eigvecs
    model.mixing_matrix.U_latent[...] = top_m_eigvecs
    model.mixing_matrix.S[...] = top_m_eigvals

    return model
