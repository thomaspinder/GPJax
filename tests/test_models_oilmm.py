"""Tests for OILMM (Orthogonal Instantaneous Linear Mixing Model)."""

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)


class TestOrthogonalMixingMatrix:
    """Tests for OrthogonalMixingMatrix parameter class."""

    def test_initialization(self):
        """Test basic initialization."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(42)
        mix = OrthogonalMixingMatrix(num_outputs=5, num_latent_gps=2, key=key)

        assert mix.num_outputs == 5
        assert mix.num_latent_gps == 2
        assert mix.U_latent[...].shape == (5, 2)
        assert mix.S[...].shape == (2,)
        assert mix.obs_noise_variance[...].shape == ()
        assert mix.latent_noise_variance[...].shape == (2,)

    def test_U_orthonormality(self):
        """Test that U has orthonormal columns via SVD."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(123)
        mix = OrthogonalMixingMatrix(num_outputs=10, num_latent_gps=3, key=key)

        U = mix.U
        UTU = U.T @ U
        expected = jnp.eye(3)

        assert jnp.allclose(UTU, expected, atol=1e-6)

    def test_H_matrix_shape_and_composition(self):
        """Test H = U S^(1/2) has correct shape and composition."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(456)
        p, m = 7, 3
        mix = OrthogonalMixingMatrix(num_outputs=p, num_latent_gps=m, key=key)

        H = mix.H
        assert H.shape == (p, m)

        # Check H = U * sqrt(S) (broadcasting)
        U = mix.U
        sqrt_S = jnp.sqrt(mix.S[...])
        expected_H = U * sqrt_S[None, :]
        assert jnp.allclose(H, expected_H, atol=1e-10)

    def test_T_matrix_is_pseudo_inverse(self):
        """Test T = S^(-1/2) U^T is the left pseudo-inverse of H."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(789)
        mix = OrthogonalMixingMatrix(num_outputs=8, num_latent_gps=4, key=key)

        T = mix.T
        H = mix.H

        # T H should equal I_m (identity in latent space)
        TH = T @ H
        expected = jnp.eye(mix.num_latent_gps)
        assert jnp.allclose(TH, expected, atol=1e-6)

    def test_projected_noise_variance_diagonal(self):
        """Test projected noise is diagonal: σ²S^(-1) + D."""
        from gpjax.models.oilmm import OrthogonalMixingMatrix

        key = jax.random.PRNGKey(101)
        mix = OrthogonalMixingMatrix(num_outputs=6, num_latent_gps=2, key=key)

        # Set specific noise values for testing
        mix.obs_noise_variance[...] = jnp.array(0.5)
        mix.latent_noise_variance[...] = jnp.array([0.1, 0.2])
        mix.S[...] = jnp.array([2.0, 4.0])

        proj_noise = mix.projected_noise_variance

        # Expected: σ²/S + D = 0.5/[2.0, 4.0] + [0.1, 0.2]
        expected = jnp.array([0.5 / 2.0 + 0.1, 0.5 / 4.0 + 0.2])
        assert jnp.allclose(proj_noise, expected, atol=1e-10)


class TestOILMMModel:
    """Tests for OILMMModel."""

    def test_initialization(self):
        """Test OILMMModel initialization."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        assert model.num_outputs == 3
        assert model.num_latent_gps == 2
        assert len(model.latent_priors) == 2

    def test_project_observations(self):
        """Test projection: y_latent = T @ y."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(123)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=5,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Create test data
        N, P = 10, 5
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.sin(2 * jnp.pi * X) + jr.normal(key, (N, P)) * 0.1
        dataset = gpx.Dataset(X=X, y=y)

        # Project
        X_out, y_projected = model._project_observations(dataset)

        # Check shapes
        assert X_out.shape == (N, 1)
        assert y_projected.shape == (2, N)  # [M, N]

        # Verify projection: T @ y^T = y_latent
        T = model.mixing_matrix.T  # [M, P]
        expected = T @ y.T  # [M, P] @ [P, N] = [M, N]
        assert jnp.allclose(y_projected, expected, atol=1e-10)

    def test_condition_on_observations(self):
        """Test conditioning creates M independent posteriors."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, OILMMPosterior

        key = jax.random.PRNGKey(456)
        kernel = gpx.kernels.Matern52()

        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Create training data
        N, P = 20, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        # Condition
        posterior = model.condition_on_observations(dataset)

        # Check type
        assert isinstance(posterior, OILMMPosterior)
        assert posterior.num_latent_gps == 2
        assert len(posterior.latent_posteriors) == 2

    def test_conditioned_posteriors_use_correct_noise(self):
        """Test that each latent posterior gets correct projected noise."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(789)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=6,
            num_latent_gps=3,
            kernel=kernel,
            key=key,
        )

        # Set known noise values
        model.mixing_matrix.obs_noise_variance[...] = jnp.array(0.5)
        model.mixing_matrix.latent_noise_variance[...] = jnp.array([0.1, 0.2, 0.3])
        model.mixing_matrix.S[...] = jnp.array([1.0, 2.0, 4.0])

        # Create data and condition
        N = 15
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, 6))
        dataset = gpx.Dataset(X=X, y=y)

        posterior = model.condition_on_observations(dataset)

        # Verify each posterior has correct noise.
        # Gaussian likelihood wraps obs_stddev in NonNegativeReal, so
        # we access [...] to get the raw array, then square to get variance.
        expected_noise_vars = model.mixing_matrix.projected_noise_variance
        for i in range(3):
            lik = posterior.latent_posteriors[i].likelihood
            # lik.obs_stddev is a NonNegativeReal — get raw value
            obs_var = lik.obs_stddev[...] ** 2
            expected = expected_noise_vars[i]
            assert jnp.allclose(obs_var, expected, atol=1e-6), (
                f"Latent GP {i}: expected noise var {expected}, got {obs_var}"
            )


class TestOILMMPosterior:
    """Tests for OILMMPosterior prediction."""

    def test_predict_mean_shape(self):
        """Test prediction mean has correct shape [NP]."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        kernel = gpx.kernels.RBF()

        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Create and condition on training data
        N_train, P = 20, 3
        X_train = jnp.linspace(0, 1, N_train).reshape(-1, 1)
        y_train = jr.normal(key, (N_train, P))
        train_data = gpx.Dataset(X=X_train, y=y_train)

        posterior = model.condition_on_observations(train_data)

        # Predict at test points
        N_test = 10
        X_test = jnp.linspace(0.1, 0.9, N_test).reshape(-1, 1)
        pred = posterior.predict(X_test)

        # Check mean shape (flattened output-major: [NP])
        assert pred.mean.shape == (N_test * P,)
        assert jnp.all(jnp.isfinite(pred.mean))

    def test_predict_mean_reconstruction(self):
        """Test mean is correctly reconstructed: f_mean = H @ latent_means."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(123)
        kernel = gpx.kernels.Matern52()

        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=kernel,
            key=key,
        )

        # Condition
        N, P = 15, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)
        posterior = model.condition_on_observations(dataset)

        # Predict
        X_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
        pred = posterior.predict(X_test)

        # Manually compute expected mean using each latent posterior + its dataset
        latent_means = jnp.array(
            [
                post.predict(X_test, ds).mean
                for post, ds in zip(
                    posterior.latent_posteriors,
                    posterior.latent_datasets,
                    strict=True,
                )
            ]
        )  # [M, N_test]
        H = model.mixing_matrix.H  # [P, M]
        expected_mean = jnp.einsum("pm,mn->pn", H, latent_means)  # [P, N_test]
        expected_mean_flat = expected_mean.T.ravel()  # [N_test * P]

        assert jnp.allclose(pred.mean, expected_mean_flat, atol=1e-6)

    def test_predict_full_covariance_shape(self):
        """Test full covariance has shape [NP, NP]."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(456)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 12, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        posterior = model.condition_on_observations(gpx.Dataset(X=X, y=y))

        X_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
        pred = posterior.predict(X_test)

        expected_shape = (5 * P, 5 * P)
        assert pred.covariance().shape == expected_shape
        assert jnp.all(jnp.isfinite(pred.covariance()))

    def test_predict_covariance_is_psd(self):
        """Test covariance is positive semi-definite."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(789)
        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=gpx.kernels.Matern32(),
            key=key,
        )

        N, P = 10, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        posterior = model.condition_on_observations(gpx.Dataset(X=X, y=y))

        X_test = jnp.linspace(0.1, 0.9, 8).reshape(-1, 1)
        pred = posterior.predict(X_test)
        cov = pred.covariance()

        # Check PSD via eigenvalues
        eigvals = jnp.linalg.eigvalsh(cov)
        assert jnp.all(eigvals >= -1e-6), "Covariance not PSD"


class TestOILMMConstructors:
    """Tests for OILMM convenience constructors."""

    def test_create_oilmm(self):
        """Test basic create_oilmm constructor."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm

        key = jax.random.PRNGKey(42)
        model = create_oilmm(
            num_outputs=5,
            num_latent_gps=2,
            key=key,
            kernel=gpx.kernels.RBF(),
        )

        assert model.num_outputs == 5
        assert model.num_latent_gps == 2

    def test_create_oilmm_with_kernels(self):
        """Test constructor with custom kernels per latent."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_with_kernels

        key = jax.random.PRNGKey(123)
        kernels = [gpx.kernels.RBF(), gpx.kernels.Matern52()]

        model = create_oilmm_with_kernels(
            latent_kernels=kernels,
            num_outputs=6,
            key=key,
        )

        assert model.num_outputs == 6
        assert model.num_latent_gps == 2
        # Verify each prior has the correct kernel
        assert isinstance(model.latent_priors[0].kernel, gpx.kernels.RBF)
        assert isinstance(model.latent_priors[1].kernel, gpx.kernels.Matern52)

    def test_create_oilmm_from_data(self):
        """Test data-informed initialization."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_from_data

        key = jax.random.PRNGKey(456)

        # Create dataset with structure
        N = 50
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        # Create correlated outputs
        base = jnp.sin(2 * jnp.pi * X.squeeze())
        y = jnp.column_stack(
            [
                base,
                base + jr.normal(key, (N,)) * 0.1,
                -base,
                jr.normal(key, (N,)),
            ]
        )
        dataset = gpx.Dataset(X=X, y=y)

        model = create_oilmm_from_data(
            dataset=dataset,
            num_latent_gps=2,
            key=key,
            kernel=gpx.kernels.RBF(),
        )

        assert model.num_outputs == 4
        assert model.num_latent_gps == 2

        # U should be initialized to top-2 eigenvectors
        U = model.mixing_matrix.U
        assert U.shape == (4, 2)
        assert jnp.allclose(U.T @ U, jnp.eye(2), atol=1e-6)


class TestOILMMIntegration:
    """End-to-end integration tests for OILMM workflow."""

    def test_full_workflow_create_condition_predict(self):
        """Test complete OILMM workflow: create, condition, predict."""
        import gpjax as gpx

        key = jax.random.PRNGKey(42)

        # 1. Create model
        model = gpx.models.create_oilmm(
            num_outputs=3,
            num_latent_gps=2,
            key=key,
            kernel=gpx.kernels.RBF(),
        )

        # 2. Generate synthetic training data
        N_train, P = 30, 3
        X_train = jnp.linspace(0, 1, N_train).reshape(-1, 1)
        # Create correlated outputs
        base1 = jnp.sin(2 * jnp.pi * X_train.squeeze())
        base2 = jnp.cos(2 * jnp.pi * X_train.squeeze())
        y_train = jnp.column_stack(
            [
                base1 + 0.5 * base2,
                base2,
                -base1 + base2,
            ]
        )
        y_train += jr.normal(key, y_train.shape) * 0.05  # Add noise
        train_data = gpx.Dataset(X=X_train, y=y_train)

        # 3. Condition on observations
        posterior = model.condition_on_observations(train_data)

        # Verify posterior type
        assert isinstance(posterior, gpx.models.OILMMPosterior)
        assert posterior.num_latent_gps == 2

        # 4. Predict at test points (train data stored internally)
        N_test = 10
        X_test = jnp.linspace(0.1, 0.9, N_test).reshape(-1, 1)
        pred = posterior.predict(X_test)

        # 5. Verify prediction properties
        assert pred.mean.shape == (N_test * P,)
        assert pred.covariance().shape == (N_test * P, N_test * P)
        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))

        # Verify covariance is PSD
        eigvals = jnp.linalg.eigvalsh(pred.covariance())
        assert jnp.all(eigvals >= -1e-6)

    def test_oilmm_vs_independent_gps_sanity(self):
        """Sanity check: OILMM with m=p should behave reasonably."""
        import gpjax as gpx

        key = jax.random.PRNGKey(123)
        P = 2

        # OILMM with m=p (maximal latent GPs)
        model = gpx.models.create_oilmm(
            num_outputs=P,
            num_latent_gps=P,
            key=key,
            kernel=gpx.kernels.Matern52(),
        )

        # Simple data
        N = 20
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        # Condition and predict
        posterior = model.condition_on_observations(dataset)
        pred = posterior.predict(X[:10])

        # Basic sanity: predictions should be finite and reasonable scale
        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.std(pred.mean) < 10.0  # Not exploding

    def test_jit_compatibility(self):
        """Test that OILMM prediction internals are JIT-compatible.

        GaussianDistribution is not a registered JAX pytree, so we JIT the
        computation and extract raw arrays inside the traced function.
        """
        import gpjax as gpx

        key = jax.random.PRNGKey(456)
        model = gpx.models.create_oilmm(
            num_outputs=4,
            num_latent_gps=2,
            key=key,
        )

        # Create data and condition
        N = 15
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, 4))
        dataset = gpx.Dataset(X=X, y=y)
        posterior = model.condition_on_observations(dataset)

        # JIT a function that returns raw arrays (not GaussianDistribution)
        @jax.jit
        def predict_arrays(X_test):
            pred = posterior.predict(X_test)
            return pred.loc, pred.covariance()

        X_test = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)

        # Run JIT version
        mean_jit, cov_jit = predict_arrays(X_test)

        # Run non-JIT version
        pred_normal = posterior.predict(X_test)

        # Results should match
        assert jnp.allclose(mean_jit, pred_normal.mean, atol=1e-10)
        assert jnp.allclose(cov_jit, pred_normal.covariance(), atol=1e-10)


class TestOILMMMLL:
    """Tests for oilmm_mll log marginal likelihood."""

    def test_returns_finite_scalar(self):
        """Test MLL returns a finite scalar."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(42)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 20, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        mll = oilmm_mll(model, data)
        assert mll.shape == ()
        assert jnp.isfinite(mll)

    def test_correction_terms_nonzero(self):
        """Correction terms make MLL differ from naive latent sum."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll
        from gpjax.objectives import conjugate_mll

        key = jax.random.PRNGKey(123)
        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 15, 4
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        full_mll = oilmm_mll(model, data)

        # Compute just the sum of latent MLLs
        posterior = model.condition_on_observations(data)
        naive_sum = sum(
            conjugate_mll(post, ds)
            for post, ds in zip(
                posterior.latent_posteriors,
                posterior.latent_datasets,
                strict=True,
            )
        )

        # They should differ due to correction terms
        assert not jnp.allclose(full_mll, naive_sum, atol=1e-6)

    def test_gradient_flows(self):
        """Gradients flow through all OILMM parameters."""
        from flax import nnx
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(456)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 10, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        graphdef, state = nnx.split(model)

        def loss_fn(state):
            m = nnx.merge(graphdef, state)
            return -oilmm_mll(m, data)

        grads = jax.grad(loss_fn)(state)

        # Check no NaN gradients in any leaf
        flat_grads = jax.tree.leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g)), f"NaN gradient found: {g}"

    def test_small_scale_brute_force(self):
        """For tiny problem, verify against brute-force MOGP MLL."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(789)
        n, p, m = 5, 3, 2

        model = OILMMModel(
            num_outputs=p,
            num_latent_gps=m,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        # Fix parameters for deterministic comparison
        model.mixing_matrix.obs_noise_variance[...] = jnp.array(0.1)
        model.mixing_matrix.latent_noise_variance[...] = jnp.zeros(m)
        model.mixing_matrix.S[...] = jnp.array([2.0, 1.5])

        X = jnp.linspace(0, 1, n).reshape(-1, 1)
        y = jr.normal(key, (n, p))
        data = gpx.Dataset(X=X, y=y)

        oilmm_val = oilmm_mll(model, data)

        # Brute-force: compute full NP×NP covariance
        H = model.mixing_matrix.H  # [P, M]
        sigma2 = model.mixing_matrix.obs_noise_variance[...]

        # Compute each latent kernel matrix
        latent_Ks = []
        for i in range(m):
            prior = model.latent_priors[i]
            K = prior.kernel.cross_covariance(X, X)  # [N, N]
            latent_Ks.append(K)

        latent_K_block = jax.scipy.linalg.block_diag(*latent_Ks)  # [MN, MN]
        H_kron_I = jnp.kron(H, jnp.eye(n))  # [PN, MN]
        full_cov = H_kron_I @ latent_K_block @ H_kron_I.T + sigma2 * jnp.eye(n * p)

        # Evaluate log N(vec(Y) | 0, full_cov)
        y_vec = y.T.ravel()  # [NP] output-major — but we need same ordering
        # The OILMM uses output-major flattening: y[0,:], y[1,:], ...
        # which is y.ravel() in row-major (N,P) -> alternating outputs
        # Actually vec(Y) for [N,P] Y with H being [P,M] needs Y^T flattened
        # Let's use the column-major convention: stack by output
        y_vec = y.T.ravel()  # [P*N]: all obs for output 0, then output 1, etc.

        # log N(y | 0, C) = -0.5 * (y^T C^{-1} y + log|C| + NP log(2π))
        L = jnp.linalg.cholesky(full_cov)
        alpha = jax.scipy.linalg.solve_triangular(L, y_vec, lower=True)
        brute_mll = (
            -0.5 * jnp.dot(alpha, alpha)
            - jnp.sum(jnp.log(jnp.diag(L)))
            - 0.5 * n * p * jnp.log(2.0 * jnp.pi)
        )

        assert jnp.allclose(oilmm_val, brute_mll, atol=1e-4), (
            f"OILMM MLL {oilmm_val} != brute-force {brute_mll}"
        )

    def test_deterministic(self):
        """oilmm_mll produces identical results on repeated calls."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel, oilmm_mll

        key = jax.random.PRNGKey(101)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 10, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        data = gpx.Dataset(X=X, y=y)

        val1 = oilmm_mll(model, data)
        val2 = oilmm_mll(model, data)

        assert jnp.allclose(val1, val2, atol=1e-10)


class TestKernelIndependence:
    """Tests for independent kernel instances per latent GP."""

    def test_single_kernel_creates_independent_copies(self):
        """Single kernel is deep-copied so latents have independent params."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        kernel = gpx.kernels.RBF()
        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=3,
            kernel=kernel,
            key=key,
        )

        # Modify latent_priors[0].kernel.lengthscale
        original_ls_1 = model.latent_priors[1].kernel.lengthscale[...].copy()
        model.latent_priors[0].kernel.lengthscale[...] = jnp.array(99.0)

        # latent_priors[1] should be unchanged
        assert jnp.allclose(
            model.latent_priors[1].kernel.lengthscale[...], original_ls_1
        )
        assert not jnp.allclose(
            model.latent_priors[0].kernel.lengthscale[...],
            model.latent_priors[1].kernel.lengthscale[...],
        )

    def test_list_of_kernels_used_directly(self):
        """List of kernels are used as-is."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(123)
        kernels = [gpx.kernels.RBF(), gpx.kernels.Matern52()]

        model = OILMMModel(
            num_outputs=4,
            num_latent_gps=2,
            kernel=kernels,
            key=key,
        )

        assert isinstance(model.latent_priors[0].kernel, gpx.kernels.RBF)
        assert isinstance(model.latent_priors[1].kernel, gpx.kernels.Matern52)

    def test_wrong_kernel_list_length_raises(self):
        """Passing wrong number of kernels raises ValueError."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel
        import pytest

        key = jax.random.PRNGKey(456)
        kernels = [gpx.kernels.RBF(), gpx.kernels.Matern52()]

        with pytest.raises(ValueError, match="Expected 3 kernels, got 2"):
            OILMMModel(
                num_outputs=4,
                num_latent_gps=3,  # Mismatch: 3 latents but 2 kernels
                kernel=kernels,
                key=key,
            )


class TestSInitialization:
    """Tests for eigenvalue initialization of S."""

    def test_create_from_data_initializes_S_from_eigenvalues(self):
        """S is initialized to top-m eigenvalues of empirical covariance."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_from_data

        key = jax.random.PRNGKey(42)

        N = 50
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        base = jnp.sin(2 * jnp.pi * X.squeeze())
        y = jnp.column_stack(
            [
                base * 3.0,
                base * 2.0 + jr.normal(key, (N,)) * 0.1,
                -base,
                jr.normal(key, (N,)) * 0.5,
            ]
        )
        dataset = gpx.Dataset(X=X, y=y)

        model = create_oilmm_from_data(dataset=dataset, num_latent_gps=2, key=key)

        # Manually compute expected eigenvalues
        y_centered = y - y.mean(axis=0, keepdims=True)
        emp_cov = y_centered.T @ y_centered / N
        eigvals = jnp.linalg.eigvalsh(emp_cov)
        idx = jnp.argsort(eigvals)[::-1]
        expected_S = jnp.maximum(eigvals[idx[:2]], 1e-6)

        assert jnp.allclose(model.mixing_matrix.S[...], expected_S, atol=1e-6)

    def test_create_from_data_clamps_small_eigenvalues(self):
        """Near-zero eigenvalues are clamped to 1e-6."""
        import gpjax as gpx
        from gpjax.models.oilmm import create_oilmm_from_data

        key = jax.random.PRNGKey(123)

        N = 50
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        # One output is near-constant (eigenvalue ≈ 0)
        y = jnp.column_stack(
            [
                jnp.sin(X.squeeze()),
                jnp.ones(N) * 5.0,  # constant -> zero variance
                jnp.cos(X.squeeze()),
            ]
        )
        dataset = gpx.Dataset(X=X, y=y)

        model = create_oilmm_from_data(dataset=dataset, num_latent_gps=3, key=key)

        assert jnp.all(model.mixing_matrix.S[...] >= 1e-6)


class TestCovarianceEquivalence:
    """Test that einsum covariance matches old Kronecker computation."""

    def test_full_and_diagonal_covariance_paths_are_consistent(self):
        """Full-covariance and diagonal-covariance paths should share indexing."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(7)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        X_train = jnp.linspace(0, 1, 10).reshape(-1, 1)
        y_train = jr.normal(key, (10, 3))
        dataset = gpx.Dataset(X=X_train, y=y_train)
        posterior = model.condition_on_observations(dataset)

        X_test = jnp.linspace(0.1, 0.9, 6).reshape(-1, 1)
        pred_full = posterior.predict(X_test, return_full_cov=True)
        pred_diag = posterior.predict(X_test, return_full_cov=False)

        full_diag = jnp.diag(pred_full.covariance())
        diag_only = jnp.diag(pred_diag.covariance())

        assert jnp.allclose(pred_full.mean, pred_diag.mean, atol=1e-8)
        assert jnp.allclose(full_diag, diag_only, atol=1e-6)

    def test_einsum_cov_matches_kronecker_reference(self):
        """New einsum covariance matches old Kronecker computation."""
        import gpjax as gpx
        from gpjax.models.oilmm import OILMMModel

        key = jax.random.PRNGKey(42)
        model = OILMMModel(
            num_outputs=3,
            num_latent_gps=2,
            kernel=gpx.kernels.RBF(),
            key=key,
        )

        N, P = 8, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jr.normal(key, (N, P))
        dataset = gpx.Dataset(X=X, y=y)

        posterior = model.condition_on_observations(dataset)

        # Get einsum result from current implementation
        X_test = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)
        pred = posterior.predict(X_test, return_full_cov=True)
        einsum_cov = pred.covariance()

        # Compute reference via old Kronecker method
        N_test = X_test.shape[0]
        H = model.mixing_matrix.H

        latent_preds = [
            post.predict(X_test, ds)
            for post, ds in zip(
                posterior.latent_posteriors,
                posterior.latent_datasets,
                strict=True,
            )
        ]
        latent_covs = [pred.covariance() for pred in latent_preds]

        latent_cov_block = jax.scipy.linalg.block_diag(*latent_covs)
        H_kron_I = jnp.kron(H, jnp.eye(N_test))
        kron_cov = H_kron_I @ latent_cov_block @ H_kron_I.T

        # Reindex from output-major ([p0 all t], [p1 all t], ...) to
        # N-major ([t0 all p], [t1 all p], ...) to match predict() output.
        n_major_idx = jnp.array(
            [p * N_test + n for n in range(N_test) for p in range(P)]
        )
        kron_cov_n_major = kron_cov[n_major_idx][:, n_major_idx]

        assert jnp.allclose(einsum_cov, kron_cov_n_major, atol=1e-6)
