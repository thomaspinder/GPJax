# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
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

# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)


from gpjax.dataset import Dataset
from gpjax.fit import fit
from gpjax.gps import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import (
    AbstractMeanFunction,
    CombinationMeanFunction,
    Constant,
    Zero,
)
from gpjax.objectives import conjugate_mll
from gpjax.parameters import (
    Real,
    _val,
)
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    Num,
)
import optax as ox
from paramax import (
    AbstractUnwrappable,
    NonTrainable,
    unwrap,
)
import pytest


def test_abstract() -> None:
    # Check abstract mean function cannot be instantiated, as the `__call__` method is not defined.
    with pytest.raises(TypeError):
        AbstractMeanFunction()

    # Check a "dummy" mean function with defined abstract method, `__call__`, can be instantiated.
    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, x: Float[Array, " D"]) -> Float[Array, "1"]:
            return jnp.array([1.0])

    mf = DummyMeanFunction()
    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jnp.array([1.0])) == jnp.array([1.0])).all()
    assert (mf(jnp.array([2.0, 3.0])) == jnp.array([1.0])).all()


@pytest.mark.parametrize(
    "constant", [jnp.array([0.0]), jnp.array([1.0]), jnp.array([3.0])]
)
def test_constant(constant: Float[Array, " Q"]) -> None:
    mf = Constant(constant=constant)

    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jnp.array([[1.0]])) == jnp.array([constant])).all()
    assert (mf(jnp.array([[2.0, 3.0]])) == jnp.array([constant])).all()
    assert (mf(jnp.array([[1.0], [2.0]])) == jnp.array([constant, constant])).all()
    assert (
        mf(jnp.array([[1.0, 2.0], [3.0, 4.0]])) == jnp.array([constant, constant])
    ).all()


def test_zero_mean_initialises_at_zero() -> None:
    """Zero mean function should initialise its constant at 0.0."""
    meanf = Zero()
    assert jnp.allclose(_val(meanf.constant), 0.0)


def test_zero_mean_remains_zero() -> None:
    """The zero mean must stay at zero after fitting data with a non-zero mean.

    Regression test for #330/#712. The constant is frozen, so gradient descent
    must not be able to move it towards the data mean.
    """
    x = jnp.linspace(0.0, 1.0, 20, dtype=jnp.float64).reshape(-1, 1)
    y = jnp.full((20, 1), 50.0, dtype=jnp.float64)  # dataset with a non-zero mean
    train_data = Dataset(X=x, y=y)

    posterior = Prior(mean_function=Zero(), kernel=RBF()) * Gaussian(
        num_datapoints=train_data.n
    )

    optimised, _ = fit(
        model=posterior,
        objective=lambda model, data: -conjugate_mll(model, data),
        train_data=train_data,
        optim=ox.adam(0.1),
        num_iters=100,
        key=jr.PRNGKey(42),
        verbose=False,
    )

    mean_function = unwrap(optimised.prior.mean_function)
    assert jnp.allclose(mean_function.constant, 0.0)
    assert jnp.allclose(mean_function(x), 0.0)


def test_initialising_zero_mean_with_constant_raises_error():
    with pytest.raises(TypeError):
        Zero(constant=jnp.array([1.0]))


@pytest.fixture
def dummy_mean_function() -> AbstractMeanFunction:
    """Create a simple mean function for testing."""

    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
            return jnp.ones((x.shape[0], 1))

    return DummyMeanFunction()


@pytest.fixture
def constant_mean_function() -> AbstractMeanFunction:
    """Create a constant mean function for testing."""
    return Constant(constant=jnp.array([2.0]))


def test_mean_function_addition(
    dummy_mean_function: AbstractMeanFunction,
    constant_mean_function: AbstractMeanFunction,
) -> None:
    """Test addition of two mean functions."""
    # Test adding two mean functions
    sum_mean = dummy_mean_function + constant_mean_function

    # Check the result is a CombinationMeanFunction with sum operator
    assert isinstance(sum_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = sum_mean(x)

    # Expected: dummy returns ones, constant returns 2.0, sum should be 3.0
    expected = jnp.array([[3.0], [3.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_radd(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test right addition of mean function with a constant."""
    # Test adding constant to mean function
    constant_value = jnp.array([5.0])
    sum_mean = constant_value + dummy_mean_function

    # Check the result is a CombinationMeanFunction with sum operator
    assert isinstance(sum_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = sum_mean(x)

    # Expected: dummy returns ones, constant returns 5.0, sum should be 6.0
    expected = jnp.array([[6.0], [6.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_add_constant(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test addition of mean function with a constant."""
    # Test adding mean function to constant
    constant_value = jnp.array([3.0])
    sum_mean = dummy_mean_function + constant_value

    # Check the result is a CombinationMeanFunction with sum operator
    assert isinstance(sum_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = sum_mean(x)

    # Expected: dummy returns ones, constant returns 3.0, sum should be 4.0
    expected = jnp.array([[4.0], [4.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_multiplication(
    dummy_mean_function: AbstractMeanFunction,
    constant_mean_function: AbstractMeanFunction,
) -> None:
    """Test multiplication of two mean functions."""
    # Test multiplying two mean functions
    product_mean = dummy_mean_function * constant_mean_function

    # Check the result is a CombinationMeanFunction with product operator
    assert isinstance(product_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = product_mean(x)

    # Expected: dummy returns ones, constant returns 2.0, product should be 2.0
    expected = jnp.array([[2.0], [2.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_rmul(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test right multiplication of mean function with a constant."""
    # Test multiplying constant with mean function
    constant_value = jnp.array([5.0])
    product_mean = constant_value * dummy_mean_function

    # Check the result is a CombinationMeanFunction with product operator
    assert isinstance(product_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = product_mean(x)

    # Expected: dummy returns ones, constant returns 5.0, product should be 5.0
    expected = jnp.array([[5.0], [5.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_mul_constant(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test multiplication of mean function with a constant."""
    # Test multiplying mean function with constant
    constant_value = jnp.array([3.0])
    product_mean = dummy_mean_function * constant_value

    # Check the result is a CombinationMeanFunction with product operator
    assert isinstance(product_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = product_mean(x)

    # Expected: dummy returns ones, constant returns 3.0, product should be 3.0
    expected = jnp.array([[3.0], [3.0]])
    assert jnp.allclose(result, expected)


def test_chained_operations(
    dummy_mean_function: AbstractMeanFunction,
    constant_mean_function: AbstractMeanFunction,
) -> None:
    """Test chained operations between mean functions."""
    # Test a combination of addition and multiplication
    # Use jnp.array instead of float 3.0
    combined = dummy_mean_function + constant_mean_function * jnp.array([3.0])

    # Check the structure is correct
    assert isinstance(combined, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = combined(x)

    # dummy returns 1.0, constant returns 2.0, 2.0 * 3.0 = 6.0, 1.0 + 6.0 = 7.0
    expected = jnp.array([[7.0], [7.0]])
    assert jnp.allclose(result, expected)


def test_constant_mean_function_with_parameter():
    """Test Constant mean function with a Parameter object."""
    from gpjax.parameters import Real

    # Create Constant with a Parameter
    param = Real(2.5)
    meanf = Constant(constant=param)

    # Check that the constant is stored as a Parameter
    assert isinstance(meanf.constant, Real)
    assert jnp.allclose(meanf.constant.unwrap(), 2.5)

    # Test evaluation
    x = jnp.array([[1.0], [2.0], [3.0]])
    result = meanf(x)
    expected = jnp.array([[2.5], [2.5], [2.5]])
    assert jnp.allclose(result, expected)


def test_constant_mean_function_with_raw_value():
    """Test Constant mean function with a raw float/array value."""
    # Create Constant with raw value
    meanf = Constant(constant=3.7)

    # Check that the constant is stored as a raw array, not a Parameter
    assert not isinstance(meanf.constant, AbstractUnwrappable)
    assert isinstance(meanf.constant, jnp.ndarray)
    assert jnp.allclose(meanf.constant, 3.7)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = meanf(x)
    expected = jnp.array([[3.7], [3.7]])
    assert jnp.allclose(result, expected)


def test_constant_mean_function_with_array():
    """Test Constant mean function with an array value."""
    # Create Constant with array value
    value = jnp.array([1.5, 2.5])
    meanf = Constant(constant=value)

    # Check that the constant is stored as a raw array
    assert not isinstance(meanf.constant, AbstractUnwrappable)
    assert isinstance(meanf.constant, jnp.ndarray)
    assert jnp.allclose(meanf.constant, value)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = meanf(x)
    expected = jnp.array([[1.5, 2.5], [1.5, 2.5]])
    assert jnp.allclose(result, expected)


def test_zero_mean_function_constant_is_frozen():
    """The zero constant is wrapped so that optimisers cannot update it.

    ``fit`` treats every array leaf as trainable, so a bare array would be
    optimised. The ``NonTrainable`` wrapper is what keeps the zero at zero.
    """
    meanf = Zero()

    assert isinstance(meanf.constant, NonTrainable)
    assert jnp.allclose(_val(meanf.constant), 0.0)

    # The wrapper must stop gradients reaching the constant. `eqx.filter` alone
    # does not hide it -- `unwrap` is what applies `stop_gradient`.
    x = jnp.array([[1.0], [2.0], [3.0]])
    grads = jax.grad(lambda mf: jnp.sum((unwrap(mf)(x) - 50.0) ** 2))(meanf)
    assert jnp.allclose(grads.constant.tree, 0.0)

    # Test evaluation
    x = jnp.array([[1.0], [2.0], [3.0]])
    result = meanf(x)
    expected = jnp.array([[0.0], [0.0], [0.0]])
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_constant_dtype_preservation_raw(dtype):
    """Test that Constant mean function preserves dtype when given a raw array."""
    x = jnp.arange(5, dtype=dtype).reshape(-1, 1)
    constant = jnp.array(3.0, dtype=dtype)
    mean_fn = Constant(constant)
    mean = mean_fn(x)
    assert mean.dtype == dtype


def test_constant_dtype_preservation_real():
    """Real parameter always stores float64, so output is float64."""
    x = jnp.arange(5, dtype=jnp.float64).reshape(-1, 1)
    constant = Real(jnp.array(3.0, dtype=jnp.float64))
    mean_fn = Constant(constant)
    mean = mean_fn(x)
    assert mean.dtype == jnp.float64
