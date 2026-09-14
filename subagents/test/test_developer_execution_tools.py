import sys

from pathlib import Path

import tools.developer_execution as developer

from services.process_runner import (
    evaluate_process_policy,
)

from tools.registry import (
    get_tool,
)


def configure_workspace(
    monkeypatch,
    root: Path,
):
    monkeypatch.setattr(
        developer,
        "resolve_cwd",
        lambda relative_path: (
            root
            if relative_path
            in {
                ".",
                "",
            }
            else (
                root
                / relative_path
            ).resolve()
        ),
    )


def test_detects_python_project(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "requirements.txt"
    ).write_text(
        "pytest\n",
        encoding="utf-8",
    )

    result = (
        developer
        .workspace_project_info()
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "project_type"
    ] == "python"

    assert (
        "requirements.txt"
        in result[
            "markers"
        ]
    )


def test_detects_elixir_project(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "mix.exs"
    ).write_text(
        "defmodule Demo.MixProject do\nend\n",
        encoding="utf-8",
    )

    result = (
        developer
        .workspace_project_info()
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "project_type"
    ] == "elixir"


def test_detects_nim_project(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "demo.nimble"
    ).write_text(
        'version = "0.1.0"\n',
        encoding="utf-8",
    )

    result = (
        developer
        .workspace_project_info()
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "project_type"
    ] == "nim"


def test_rejects_ambiguous_project(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "requirements.txt"
    ).write_text(
        "pytest\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / "mix.exs"
    ).write_text(
        "",
        encoding="utf-8",
    )

    result = (
        developer
        .workspace_project_info()
    )

    assert result[
        "ok"
    ] is False

    assert (
        "Multiple project types"
        in result[
            "error"
        ]
    )


def test_python_tests_use_fixed_command(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "requirements.txt"
    ).write_text(
        "pytest\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run_trusted_process(
        *,
        executable,
        args,
        cwd,
        timeout_seconds,
    ):
        calls.append(
            {
                "executable":
                    executable,

                "args":
                    args,

                "cwd":
                    cwd,

                "timeout_seconds":
                    timeout_seconds,
            }
        )

        return {
            "ok": True,
            "status": "success",
            "executable":
                executable,
            "args":
                args,
            "cwd":
                cwd,
            "exit_code": 0,
            "stdout": "10 passed",
            "stderr": "",
            "timed_out": False,
            "duration_ms": 100,
        }

    monkeypatch.setattr(
        developer,
        "run_trusted_process",
        fake_run_trusted_process,
    )

    result = (
        developer
        .workspace_run_tests()
    )

    assert result[
        "ok"
    ] is True

    assert calls[
        0
    ][
        "executable"
    ] == sys.executable

    assert calls[
        0
    ][
        "args"
    ] == [
        "-m",
        "pytest",
        "-q",
    ]

    assert result[
        "operation"
    ] == "tests"


def test_python_build_uses_compileall(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "requirements.txt"
    ).write_text(
        "",
        encoding="utf-8",
    )

    calls = []

    def fake_run_trusted_process(
        *,
        executable,
        args,
        cwd,
        timeout_seconds,
    ):
        calls.append(
            {
                "executable":
                    executable,

                "args":
                    args,

                "cwd":
                    cwd,

                "timeout_seconds":
                    timeout_seconds,
            }
        )

        return {
            "ok": True,
            "status": "success",
            "executable":
                executable,
            "args":
                args,
            "cwd":
                cwd,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "duration_ms": 100,
        }

    monkeypatch.setattr(
        developer,
        "run_trusted_process",
        fake_run_trusted_process,
    )

    result = (
        developer
        .workspace_run_build()
    )

    assert result[
        "ok"
    ] is True

    assert calls[
        0
    ][
        "args"
    ] == [
        "-m",
        "compileall",
        "-q",
        ".",
    ]


def test_generic_process_exec_cannot_bypass_python_tool():
    result = (
        evaluate_process_policy(
            executable=(
                sys.executable
            ),
            args=[
                "-m",
                "pytest",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_generic_process_exec_cannot_bypass_mix_tool():
    result = (
        evaluate_process_policy(
            executable="mix",
            args=[
                "test",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_tests_require_medium_risk_approval():
    tool = (
        get_tool(
            "workspace_run_tests"
        )
    )

    assert tool is not None

    assert tool[
        "risk"
    ] == "medium"

    assert tool[
        "requires_approval"
    ] is True


def test_build_requires_medium_risk_approval():
    tool = (
        get_tool(
            "workspace_run_build"
        )
    )

    assert tool is not None

    assert tool[
        "risk"
    ] == "medium"

    assert tool[
        "requires_approval"
    ] is True