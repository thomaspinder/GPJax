"""End-to-end integration tests for multi-output GP (ICM and LCM)."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestICMEndToEnd:
    def test_fit_and_predict(self):
        """Full ICM workflow: build, fit, predict."""
        import gpjax as gpx

        key = jax.random.PRNGKey(42)
        N, P = 30, 2

        # Generate training data
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack(
            [jnp.sin(2 * jnp.pi * X.squeeze()), jnp.cos(2 * jnp.pi * X.squeeze())]
        )
        D = gpx.Dataset(X=X, y=y)

        # Build model
        coreg = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=key)
        kernel = gpx.kernels.ICMKernel(
            base_kernel=gpx.kernels.RBF(),
            coregionalization_matrix=coreg,
        )
        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        posterior = prior * lik

        # Verify MLL is computable and finite
        mll = gpx.objectives.conjugate_mll(posterior, D)
        assert jnp.isfinite(mll)

        # Predict
        M = 10
        Xtest = jnp.linspace(0, 1, M).reshape(-1, 1)
        pred = posterior.predict(Xtest, D)

        # Shape checks — mean is flat [MP], covariance is [MP, MP]
        assert pred.mean.shape == (M * P,)
        assert pred.covariance().shape == (M * P, M * P)

        # Finite checks
        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))

    def test_different_base_kernels(self):
        """ICM works with different base kernels."""
        import gpjax as gpx

        key = jax.random.PRNGKey(0)
        X = jnp.linspace(0, 1, 15).reshape(-1, 1)
        y = jnp.column_stack([jnp.sin(X.squeeze()), X.squeeze() ** 2])
        D = gpx.Dataset(X=X, y=y)

        for base_cls in [gpx.kernels.RBF, gpx.kernels.Matern32, gpx.kernels.Matern52]:
            coreg = gpx.parameters.CoregionalizationMatrix(
                num_outputs=2, rank=1, key=key
            )
            kernel = gpx.kernels.ICMKernel(
                base_kernel=base_cls(),
                coregionalization_matrix=coreg,
            )
            prior = gpx.gps.Prior(
                mean_function=gpx.mean_functions.Zero(), kernel=kernel
            )
            lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=15, num_outputs=2)
            posterior = prior * lik

            mll = gpx.objectives.conjugate_mll(posterior, D)
            assert jnp.isfinite(mll), f"MLL not finite for {base_cls.__name__}"

    def test_three_outputs(self):
        """ICM with 3 outputs and rank-2 coregionalization."""
        import gpjax as gpx

        key = jax.random.PRNGKey(0)
        N, P = 20, 3
        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack(
            [
                jnp.sin(X.squeeze()),
                jnp.cos(X.squeeze()),
                X.squeeze() ** 2,
            ]
        )
        D = gpx.Dataset(X=X, y=y)

        coreg = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=2, key=key)
        kernel = gpx.kernels.ICMKernel(
            base_kernel=gpx.kernels.RBF(),
            coregionalization_matrix=coreg,
        )
        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        posterior = prior * lik

        mll = gpx.objectives.conjugate_mll(posterior, D)
        assert jnp.isfinite(mll)

        M = 8
        Xtest = jnp.linspace(0, 1, M).reshape(-1, 1)
        pred = posterior.predict(Xtest, D)
        assert pred.mean.shape == (M * P,)
        assert pred.covariance().shape == (M * P, M * P)


class TestLCMEndToEnd:
    def test_fit_and_predict(self):
        """Full LCM workflow: build, fit, predict with Q=2 components."""
        import gpjax as gpx

        key = jax.random.PRNGKey(42)
        N, P = 30, 2
        k1, k2, _k3 = jax.random.split(key, 3)

        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack(
            [jnp.sin(2 * jnp.pi * X.squeeze()), jnp.cos(2 * jnp.pi * X.squeeze())]
        )
        D = gpx.Dataset(X=X, y=y)

        coreg1 = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=k1)
        coreg2 = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=k2)
        kernel = gpx.kernels.LCMKernel(
            kernels=[gpx.kernels.RBF(), gpx.kernels.Matern52()],
            coregionalization_matrices=[coreg1, coreg2],
        )
        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        posterior = prior * lik

        mll = gpx.objectives.conjugate_mll(posterior, D)
        assert jnp.isfinite(mll)

        M = 10
        Xtest = jnp.linspace(0, 1, M).reshape(-1, 1)
        pred = posterior.predict(Xtest, D)
        assert pred.mean.shape == (M * P,)
        assert pred.covariance().shape == (M * P, M * P)
        assert jnp.all(jnp.isfinite(pred.mean))
        assert jnp.all(jnp.isfinite(pred.covariance()))

    def test_lcm_from_icm_components(self):
        """LCM built from ICM components matches direct construction."""
        import gpjax as gpx

        key = jax.random.PRNGKey(0)
        N, P = 20, 2
        k1, k2, _k3 = jax.random.split(key, 3)

        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack([jnp.sin(X.squeeze()), X.squeeze() ** 2])
        D = gpx.Dataset(X=X, y=y)

        coreg1 = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=k1)
        coreg2 = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=k2)
        icm1 = gpx.kernels.ICMKernel(
            base_kernel=gpx.kernels.RBF(), coregionalization_matrix=coreg1
        )
        icm2 = gpx.kernels.ICMKernel(
            base_kernel=gpx.kernels.Matern52(), coregionalization_matrix=coreg2
        )
        kernel = gpx.kernels.LCMKernel.from_icm_components([icm1, icm2])

        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        posterior = prior * lik

        mll = gpx.objectives.conjugate_mll(posterior, D)
        assert jnp.isfinite(mll)

    def test_three_outputs_three_components(self):
        """LCM with P=3, Q=3 — more components than outputs is valid."""
        import gpjax as gpx

        key = jax.random.PRNGKey(0)
        N, P, Q = 20, 3, 3
        keys = jax.random.split(key, Q + 1)

        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack(
            [jnp.sin(X.squeeze()), jnp.cos(X.squeeze()), X.squeeze() ** 2]
        )
        D = gpx.Dataset(X=X, y=y)

        kernel_classes = [gpx.kernels.RBF, gpx.kernels.Matern32, gpx.kernels.Matern52]
        coregs = [
            gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=keys[q])
            for q in range(Q)
        ]
        kernels = [cls() for cls in kernel_classes]
        kernel = gpx.kernels.LCMKernel(
            kernels=kernels, coregionalization_matrices=coregs
        )

        prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
        lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        posterior = prior * lik

        mll = gpx.objectives.conjugate_mll(posterior, D)
        assert jnp.isfinite(mll)

        M = 8
        Xtest = jnp.linspace(0, 1, M).reshape(-1, 1)
        pred = posterior.predict(Xtest, D)
        assert pred.mean.shape == (M * P,)

    def test_q1_lcm_matches_icm(self):
        """Q=1 LCM produces identical results to equivalent ICM."""
        import gpjax as gpx

        key = jax.random.PRNGKey(99)
        N, P = 15, 2

        X = jnp.linspace(0, 1, N).reshape(-1, 1)
        y = jnp.column_stack([jnp.sin(X.squeeze()), X.squeeze()])
        D = gpx.Dataset(X=X, y=y)

        coreg = gpx.parameters.CoregionalizationMatrix(num_outputs=P, rank=1, key=key)

        # ICM path
        icm_kernel = gpx.kernels.ICMKernel(
            base_kernel=gpx.kernels.RBF(), coregionalization_matrix=coreg
        )
        icm_prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(), kernel=icm_kernel
        )
        icm_lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        icm_post = icm_prior * icm_lik
        icm_mll = gpx.objectives.conjugate_mll(icm_post, D)

        # LCM (Q=1) path — reuse same coreg and kernel objects
        lcm_kernel = gpx.kernels.LCMKernel(
            kernels=[icm_kernel.base_kernel],
            coregionalization_matrices=[coreg],
        )
        lcm_prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(), kernel=lcm_kernel
        )
        lcm_lik = gpx.likelihoods.MultiOutputGaussian(num_datapoints=N, num_outputs=P)
        lcm_post = lcm_prior * lcm_lik
        lcm_mll = gpx.objectives.conjugate_mll(lcm_post, D)

        assert jnp.allclose(icm_mll, lcm_mll, atol=1e-6)
