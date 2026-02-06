# # Copyright 2022 The thomaspinder Contributors. All Rights Reserved.
# #
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from gpjax.kernels.non_euclidean import GraphKernel
from gpjax.linalg.operators import Identity
from jax import (
    config,
    jit,
    vmap,
)
import jax.numpy as jnp
import networkx as nx

# # Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)


def test_graph_kernel():
    # Create a random graph, G, and verice labels, x,
    n_verticies = 20
    n_edges = 40
    G = nx.gnm_random_graph(n_verticies, n_edges, seed=123)
    x = jnp.arange(n_verticies).reshape(-1, 1)

    # Compute graph laplacian
    L = nx.laplacian_matrix(G).toarray() + jnp.eye(n_verticies) * 1e-12

    # Create graph kernel
    kern = GraphKernel(
        laplacian=L,
    )
    assert isinstance(kern, GraphKernel)
    assert kern.num_vertex == n_verticies
    assert kern.eigenvalues.shape == (n_verticies, 1)
    assert kern.eigenvectors.shape == (n_verticies, n_verticies)

    # Compute gram matrix
    Kxx = kern.gram(x)
    assert Kxx.shape == (n_verticies, n_verticies)

    # Check positive definiteness
    Kxx += Identity(Kxx.shape[0]) * 1e-6
    eigen_values = jnp.linalg.eigvalsh(Kxx.to_dense())
    assert all(eigen_values > 0)


def _build_test_kernel(n_vertices: int = 10) -> GraphKernel:
    graph = nx.path_graph(n_vertices)
    laplacian = nx.laplacian_matrix(graph).toarray() + jnp.eye(n_vertices) * 1e-12
    return GraphKernel(laplacian=laplacian)


def test_graph_kernel_accepts_vector_indices():
    kernel = _build_test_kernel()
    col_indices = jnp.arange(5).reshape(-1, 1)
    vector_indices = col_indices.squeeze()

    matrix_eval = kernel(col_indices, col_indices)
    vector_eval = kernel(vector_indices, vector_indices)

    assert matrix_eval.shape == (5, 5)
    assert vector_eval.shape == (5, 5)
    assert jnp.allclose(matrix_eval, vector_eval)


def test_graph_kernel_vmappable_over_single_indices():
    kernel = _build_test_kernel()
    idx = jnp.arange(4)

    diag_entries = vmap(lambda z: kernel(z, z))(idx)
    assert diag_entries.shape == (4,)
    assert jnp.all(diag_entries >= 0.0)


def test_graph_kernel_vmappable_over_pairs():
    kernel = _build_test_kernel()
    x = jnp.arange(5)
    y = jnp.array([4, 3, 2, 1, 0])

    vectorised_eval = vmap(lambda a, b: kernel(a, b))(x, y)
    baseline = jnp.asarray([kernel(a, b) for a, b in zip(x, y, strict=False)])

    assert vectorised_eval.shape == (5,)
    assert jnp.allclose(vectorised_eval, baseline)


def test_graph_kernel_is_jittable():
    kernel = _build_test_kernel()
    jit_kernel = jit(lambda a, b: kernel(a, b))

    column = jnp.arange(5).reshape(-1, 1)
    vector = jnp.arange(5)
    pairs = [
        (0, 0),
        (0, column),
        (vector, 1),
        (vector, vector),
        (column, column),
    ]

    for x, y in pairs:
        expected = kernel(x, y)
        result = jit_kernel(x, y)
        assert result.shape == expected.shape
        assert jnp.allclose(result, expected)
