import jax.numpy as jnp
import pytest

from gpjax.types import Dataset, NoneType, verify_dataset, SparseDataset


def test_nonetype():
    assert isinstance(None, NoneType)


@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
def test_dataset(n, outd, ind):
    x = jnp.ones((n, ind))
    y = jnp.ones((n, outd))
    d = Dataset(X=x, y=y)
    verify_dataset(d)
    assert d.n == n
    assert d.in_dim == ind
    assert d.out_dim == outd


@pytest.mark.parametrize("nx, ny", [(1, 2), (2, 1), (10, 5), (5, 10)])
def test_dataset_assertions(nx, ny):
    x = jnp.ones((nx, 1))
    y = jnp.ones((ny, 1))
    with pytest.raises(AssertionError):
        ds = Dataset(X=x, y=y)
        verify_dataset(ds)


def test_y_none():
    x = jnp.ones((10, 1))
    d = Dataset(X=x, y=None)
    verify_dataset(d)
    assert d.y is None


@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("outd", [1, 2, 10])
@pytest.mark.parametrize("ind", [1, 2, 10])
def test_sparse(n, outd, ind):
    x = jnp.ones((n*2, ind))
    y = jnp.ones((n*2, outd))
    z = jnp.ones((n, ind))
    d = SparseDataset(X=x, y=y, Z=z)
    verify_dataset(d)
    assert d.n == n*2
    assert d.n_inducing == n
    assert d.in_dim == ind
    assert d.out_dim == outd


@pytest.mark.parametrize('z_val', [None, 10])
def test_z_none(z_val):
    z = jnp.ones(shape=(z_val, 1)) if z_val else z_val
    x = jnp.ones((10, 1))
    d = SparseDataset(X=x, Z=z, y=None)
    verify_dataset(d)
    assert d.y is None
    if not z_val:
        assert d.Z is None
