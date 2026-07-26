# Copyright 2026 The thomaspinder Contributors. All Rights Reserved.
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

from pathlib import Path
import re
import tomllib


def test_runtime_dependency_metadata() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]

    assert any(dependency.startswith("scipy") for dependency in dependencies)
    assert not any(dependency.startswith("optimistix") for dependency in dependencies)


def test_no_unused_runtime_dependencies() -> None:
    """Runtime dependencies must actually be imported somewhere in the package.

    `tensorstore` was declared for macOS but imported nowhere, forcing a ~14 MB
    wheel onto every macOS install of a GP library (issue #675).
    """
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]

    package_root = Path(__file__).parents[1] / "gpjax"
    sources = "\n".join(path.read_text() for path in package_root.rglob("*.py"))

    # Distribution name -> module name where they differ.
    import_names = {"jaxlib": "jax", "numpy": "numpy"}

    for dependency in dependencies:
        name = re.split(r"[<>=!;\[ ]", dependency, maxsplit=1)[0].strip()
        module = import_names.get(name, name.replace("-", "_"))
        assert re.search(
            rf"^\s*(import|from)\s+{re.escape(module)}\b", sources, re.M
        ), (
            f"runtime dependency {name!r} is declared in pyproject.toml but never "
            f"imported in gpjax/ — drop it or document why it is load-bearing"
        )
