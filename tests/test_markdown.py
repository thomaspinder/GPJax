"""Execute the Python code chunks embedded in the project's markdown.

`docs/` is now a Sphinx source tree. Its markdown carries MyST directive fences
(```` ```{eval-rst} ````, ```` ```{toctree} ````) and the reference pages under
`docs/reference/` are nothing but those fences, with `docs/reference/generated/`
written by autosummary at build time. Only the hand-written prose pages contain
runnable Python, so the Sphinx-owned subtrees are skipped explicitly rather than
by accident of a non-recursive glob, and paths are anchored to the repository
rather than to the CWD.
"""

import pathlib

from mktestdocs import check_md_file
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Sphinx-owned subtrees of docs/: directive-only pages, autosummary output,
# jinja templates and the build directory.
DOCS_EXCLUDED_DIRS = ("_build", "_templates", "reference", "examples")


def _docs_markdown():
    return sorted(
        p
        for p in (REPO_ROOT / "docs").rglob("*.md")
        if not any(part in DOCS_EXCLUDED_DIRS for part in p.parts)
    )


# Ensure that code chunks within any markdown files execute without error
@pytest.mark.parametrize("fpath", sorted((REPO_ROOT / "gpjax").glob("*.md")), ids=str)
def test_source_good(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.parametrize("fpath", _docs_markdown(), ids=str)
def test_docs_good(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.parametrize("fpath", sorted(REPO_ROOT.glob("*.md")), ids=str)
def test_root_good(fpath):
    check_md_file(fpath=fpath, memory=True)
