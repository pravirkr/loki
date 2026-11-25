"""Which loki ran a unit: its commit, as far as it can be told."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from loki_survey.unit import loki_commit, versions

if TYPE_CHECKING:
    from pathlib import Path

GIT = shutil.which("git")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        [GIT, "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_the_commit_of_the_checkout_loki_was_imported_from(tmp_path: Path) -> None:
    package = tmp_path / "src" / "loki"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "loki")
    head = git(tmp_path, "rev-parse", "HEAD")
    no_bindings = SimpleNamespace()

    assert loki_commit(package, no_bindings) == head
    (package / "__init__.py").write_text("changed = True\n")
    assert loki_commit(package, no_bindings) == f"{head}-dirty"
    # A commit the bindings report wins.
    assert loki_commit(package, SimpleNamespace(git_commit="abc123")) == "abc123"


def test_an_installed_loki_has_no_known_commit(tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "loki"
    package.mkdir(parents=True)
    assert loki_commit(package, SimpleNamespace()) == "unknown"


def test_versions_name_the_loki_commit() -> None:
    assert versions()["loki_commit"]
