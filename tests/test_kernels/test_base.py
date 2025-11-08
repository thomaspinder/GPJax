# Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
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

from flax import nnx
from jax import config
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)
import pytest

from gpjax.kernels.base import (
    AbstractKernel,
    CombinationKernel,
    ProductKernel,
    SumKernel,
)
from gpjax.kernels.nonstationary import (
    Linear,
    Polynomial,
)
from gpjax.kernels.stationary import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
)
from gpjax.parameters import (
    PositiveReal,
    Real,
)

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

TESTED_KERNELS = [
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    Linear,
    RationalQuadratic,
]


@pytest.mark.parametrize("kernel", TESTED_KERNELS)
@pytest.mark.parametrize(
    "active_dims, n_dims",
    (p := [([3], 1), ([2, 3, 4], 3), (slice(1, 3), 2), (None, None)]),
    ids=[f"active_dims={x[0]}-n_dims={x[1]}" for x in p],
)
def test_init_dims(kernel: type[AbstractKernel], active_dims, n_dims):
    # initialize with active_dims, check if n_dims is inferred correctly
    k = kernel(active_dims=active_dims)
    assert k.active_dims == active_dims or slice(None)
    assert k.n_dims == n_dims

    # initialize with n_dims, check that active_dims is set to full slice
    k = kernel(n_dims=n_dims)
    assert k.active_dims == slice(None)
    assert k.n_dims == n_dims

    # initialize with both, no errors should be raised for mismatch
    k = kernel(active_dims=active_dims, n_dims=n_dims)
    assert k.active_dims == active_dims or slice(None)
    assert k.n_dims == n_dims

    # test that error is raised if they are incompatible
    with pytest.raises(ValueError):
        kernel(active_dims=[3], n_dims=2)

    with pytest.raises(ValueError):
        kernel(active_dims=slice(2), n_dims=1)

    # test that error is raised if types are wrong
    with pytest.raises(TypeError):
        kernel(active_dims="3", n_dims=2)

    with pytest.raises(TypeError):
        kernel(active_dims=[3], n_dims="2")


@pytest.mark.parametrize("combination_type", [SumKernel, ProductKernel])
@pytest.mark.parametrize("kernel", TESTED_KERNELS)
@pytest.mark.parametrize("n_kerns", [2, 3, 4])
def test_combination_kernel(
    combination_type: type[CombinationKernel],
    kernel: type[AbstractKernel],
    n_kerns: int,
) -> None:
    # Create inputs
    n = 20
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create list of kernels
    kernels = [kernel() for _ in range(n_kerns)]

    # Create combination kernel
    combination_kernel = combination_type(kernels=kernels)

    # Check params are a list of dictionaries
    assert combination_kernel.kernels == nnx.List(kernels)

    # Check combination kernel set
    assert len(combination_kernel.kernels) == n_kerns
    assert isinstance(combination_kernel.kernels, nnx.List)
    assert isinstance(combination_kernel.kernels[0], AbstractKernel)

    # Compute gram matrix
    Kxx = combination_kernel.gram(x)

    # Check shapes
    assert Kxx.shape[0] == Kxx.shape[1]
    assert Kxx.shape[1] == n

    # Check positive definiteness
    jitter = 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense() + jnp.eye(n) * jitter)
    assert (eigen_values > 0).all()


@pytest.mark.parametrize("k1", TESTED_KERNELS)
@pytest.mark.parametrize("k2", TESTED_KERNELS)
def test_sum_kern_value(k1: type[AbstractKernel], k2: type[AbstractKernel]) -> None:
    k1 = k1()
    k2 = k2()

    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create sum kernel
    sum_kernel = SumKernel(kernels=[k1, k2])

    # Compute gram matrix
    Kxx = sum_kernel.gram(x)

    # Compute gram matrix
    Kxx_k1 = k1.gram(x)
    Kxx_k2 = k2.gram(x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.to_dense() == Kxx_k1.to_dense() + Kxx_k2.to_dense())


@pytest.mark.parametrize("k1", TESTED_KERNELS)
@pytest.mark.parametrize("k2", TESTED_KERNELS)
def test_prod_kern_value(k1: AbstractKernel, k2: AbstractKernel) -> None:
    k1 = k1()
    k2 = k2()

    # Create inputs
    n = 10
    x = jnp.linspace(0.0, 1.0, num=n).reshape(-1, 1)

    # Create product kernel
    prod_kernel = ProductKernel(kernels=[k1, k2])

    # Compute gram matrix
    Kxx = prod_kernel.gram(x)

    # Compute gram matrix
    Kxx_k1 = k1.gram(x)
    Kxx_k2 = k2.gram(x)

    # Check manual and automatic gram matrices are equal
    assert jnp.all(Kxx.to_dense() == Kxx_k1.to_dense() * Kxx_k2.to_dense())


def test_kernel_subclassing():
    # Test initialising abstract kernel raises TypeError with unimplemented __call__ method:
    with pytest.raises(TypeError):
        AbstractKernel()

    # Create a dummy kernel class with __call__ implemented:
    class DummyKernel(AbstractKernel):
        def __init__(
            self,
            active_dims=None,
            test_a: Float[Array, "1"] = jnp.array([1.0]),
            test_b: Float[Array, "1"] = jnp.array([2.0]),
        ):
            self.test_a = Real(test_a)
            self.test_b = PositiveReal(test_b)

            super().__init__(active_dims)

        def __call__(
            self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
        ) -> Float[Array, "1"]:
            return x * self.test_b.value * y

    # Initialise dummy kernel class and test __call__ method:
    dummy_kernel = DummyKernel()
    assert dummy_kernel.test_a.value == jnp.array([1.0])
    assert dummy_kernel.test_b.value == jnp.array([2.0])
    assert dummy_kernel(jnp.array([1.0]), jnp.array([2.0])) == 4.0
