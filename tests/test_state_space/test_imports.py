"""Smoke test: the state_space sub-package exists and re-exports its public API."""


def test_state_space_package_imports():
    import gpjax.state_space as gpx_ss

    expected_public = {
        "StateSpaceConjugatePosterior",
        "StateSpacePrior",
        "TruncatedPeriodic",
        "fit",
        "fit_lbfgs",
        "fit_scipy",
        "state_space_mll",
        "to_sde",
    }
    assert set(gpx_ss.__all__) == expected_public
    for name in expected_public:
        assert hasattr(gpx_ss, name), f"gpjax.state_space is missing {name}"
