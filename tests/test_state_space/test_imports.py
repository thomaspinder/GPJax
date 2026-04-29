"""Smoke test: the state_space sub-package exists and can be imported."""


def test_state_space_package_imports():
    import gpjax.state_space as gpx_ss

    assert gpx_ss.__all__ == []
