"""The source a loki build was made from."""

from loki import libloki


def test_git_commit_names_the_source_of_the_build() -> None:
    commit = libloki.git_commit
    assert isinstance(commit, str)
    assert commit
    # A `git describe` result or "unknown": one word.
    assert commit.split() == [commit]
