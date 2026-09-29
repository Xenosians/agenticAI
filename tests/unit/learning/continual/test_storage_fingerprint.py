from pathlib import Path

from learning.continual.storage import (
    fingerprint_directory,
)


def test_directory_fingerprint_is_stable_for_same_contents(
    tmp_path: Path,
):
    root = (
        tmp_path
        / "model"
    )

    nested = (
        root
        / "subdir"
    )

    nested.mkdir(
        parents=True
    )

    (
        root
        / "config.json"
    ).write_text(
        '{"model":"test"}\n',
        encoding="utf-8",
    )

    (
        nested
        / "weights.bin"
    ).write_bytes(
        b"abc123"
    )

    first = fingerprint_directory(
        root
    )

    second = fingerprint_directory(
        root
    )

    assert first == second
