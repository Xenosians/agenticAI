from pathlib import Path

from config.path_portability import (
    resolve_portable_path,
    translate_portable_path_text,
)


def test_wsl_drive_path_translates_to_windows():
    assert (
        translate_portable_path_text(
            "/mnt/c/project/agenticAI",
            target_os="nt",
        )
        == r"C:\project\agenticAI"
    )


def test_windows_drive_path_translates_to_wsl():
    assert (
        translate_portable_path_text(
            r"C:\project\agenticAI",
            target_os="posix",
        )
        == "/mnt/c/project/agenticAI"
    )


def test_relative_path_is_not_rewritten():
    assert (
        translate_portable_path_text(
            ".runtime/learning/example",
            target_os="nt",
        )
        == ".runtime/learning/example"
    )


def test_native_relative_resolution_uses_explicit_base(
    tmp_path: Path,
):
    result = resolve_portable_path(
        "child/artifact.json",
        base=tmp_path,
    )

    assert (
        result
        == (
            tmp_path
            / "child"
            / "artifact.json"
        ).resolve()
    )
