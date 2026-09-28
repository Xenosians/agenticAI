from pathlib import Path

from learning.code_corpus.snapshot import (
    resolve_source_snapshot,
)


def test_unversioned_source_is_inspection_only_by_default(
    tmp_path: Path,
):
    (
        tmp_path
        / "sample.py"
    ).write_text(
        "def hello():\n    return 'world'\n",
        encoding="utf-8",
    )

    snapshot = resolve_source_snapshot(
        root=tmp_path,
        source_kind="general",
        logical_repository="general-python",
    )

    assert snapshot.revision_kind == "content_snapshot"
    assert snapshot.training_eligible is False
    assert len(snapshot.revision_id) == 64


def test_unversioned_source_requires_explicit_training_opt_in(
    tmp_path: Path,
):
    (
        tmp_path
        / "sample.py"
    ).write_text(
        "def hello():\n    return 'world'\n",
        encoding="utf-8",
    )

    snapshot = resolve_source_snapshot(
        root=tmp_path,
        source_kind="general",
        logical_repository="general-python",
        allow_unversioned=True,
    )

    assert snapshot.training_eligible is True
