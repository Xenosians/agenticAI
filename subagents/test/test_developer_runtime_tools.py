import json

from pathlib import Path

import tools.developer_runtime as runtime

from services.process_runner import (
    evaluate_process_policy,
)

from tools.developer_mcp import (
    register_developer_tools,
)

from tools.registry import (
    get_tool,
)


def configure_workspace(
    monkeypatch,
    root: Path,
):
    def fake_resolve_cwd(
        relative_path,
    ):
        if (
            relative_path is None
            or relative_path == "."
            or relative_path == ""
        ):
            return root

        candidate = (
            root
            / relative_path
        ).resolve()

        if (
            candidate != root
            and root not in candidate.parents
        ):
            raise ValueError(
                "Working directory is outside "
                "the approved workspace."
            )

        return candidate

    monkeypatch.setattr(
        runtime,
        "resolve_cwd",
        fake_resolve_cwd,
    )


def test_process_snapshot_uses_fixed_safe_shape(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    calls = []

    def fake_runner(
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
            "stdout": (
                "  100     1 S 20 python\n"
                "  200   100 R  5 docker\n"
            ),
            "stderr": "",
        }

    monkeypatch.setattr(
        runtime,
        "run_trusted_process",
        fake_runner,
    )

    result = (
        runtime
        .workspace_process_snapshot()
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "count"
    ] == 2

    assert calls[
        0
    ][
        "executable"
    ] == "ps"

    assert calls[
        0
    ][
        "args"
    ] == [
        "-eo",
        (
            "pid=,ppid=,stat=,"
            "etimes=,comm="
        ),
    ]

    assert result[
        "processes"
    ][
        0
    ][
        "executable"
    ] == "python"


def test_service_status_uses_fixed_compose_commands(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    calls = []

    def fake_runner(
        *,
        executable,
        args,
        cwd,
        timeout_seconds,
    ):
        calls.append(
            list(
                args
            )
        )

        if args == [
            "compose",
            "config",
            "--services",
        ]:
            return {
                "ok": True,
                "status": "success",
                "stdout": (
                    "samba-ad\n"
                    "api\n"
                ),
                "stderr": "",
            }

        if args == [
            "compose",
            "ps",
            "--all",
            "--format",
            "json",
        ]:
            return {
                "ok": True,
                "status": "success",
                "stdout": json.dumps(
                    {
                        "Service":
                            "samba-ad",

                        "Name":
                            "itsm-samba-ad",

                        "State":
                            "running",

                        "Status":
                            "Up 2 minutes",

                        "Health":
                            "",
                    }
                ),
                "stderr": "",
            }

        raise AssertionError(
            f"Unexpected command: {args}"
        )

    monkeypatch.setattr(
        runtime,
        "run_trusted_process",
        fake_runner,
    )

    result = (
        runtime
        .workspace_service_status()
    )

    assert result[
        "ok"
    ] is True

    assert calls == [
        [
            "compose",
            "config",
            "--services",
        ],
        [
            "compose",
            "ps",
            "--all",
            "--format",
            "json",
        ],
    ]

    services = {
        service[
            "service"
        ]:
            service
        for service
        in result[
            "services"
        ]
    }

    assert services[
        "samba-ad"
    ][
        "state"
    ] == "running"

    assert services[
        "api"
    ][
        "state"
    ] == "not_created"


def test_service_logs_validate_service_and_redact(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    calls = []

    def fake_runner(
        *,
        executable,
        args,
        cwd,
        timeout_seconds,
    ):
        calls.append(
            list(
                args
            )
        )

        if args == [
            "compose",
            "config",
            "--services",
        ]:
            return {
                "ok": True,
                "status": "success",
                "stdout":
                    "samba-ad\n",
                "stderr": "",
            }

        if args == [
            "compose",
            "logs",
            "--no-color",
            "--tail",
            "100",
            "samba-ad",
        ]:
            return {
                "ok": True,
                "status": "success",
                "stdout": (
                    "server started\n"
                    "token=abcdef123\n"
                    "Authorization: "
                    "Bearer abc.def.ghi\n"
                    "ldap://admin:"
                    "SuperSecret@host\n"
                ),
                "stderr": "",
            }

        raise AssertionError(
            f"Unexpected command: {args}"
        )

    monkeypatch.setattr(
        runtime,
        "run_trusted_process",
        fake_runner,
    )

    result = (
        runtime
        .workspace_service_logs(
            "samba-ad"
        )
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "redaction_applied"
    ] is True

    assert (
        "abcdef123"
        not in result[
            "logs"
        ]
    )

    assert (
        "abc.def.ghi"
        not in result[
            "logs"
        ]
    )

    assert (
        "SuperSecret"
        not in result[
            "logs"
        ]
    )

    assert calls[
        1
    ] == [
        "compose",
        "logs",
        "--no-color",
        "--tail",
        "100",
        "samba-ad",
    ]


def test_service_logs_reject_unknown_service(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    calls = []

    def fake_runner(
        *,
        executable,
        args,
        cwd,
        timeout_seconds,
    ):
        calls.append(
            list(
                args
            )
        )

        return {
            "ok": True,
            "status": "success",
            "stdout":
                "samba-ad\n",
            "stderr": "",
        }

    monkeypatch.setattr(
        runtime,
        "run_trusted_process",
        fake_runner,
    )

    result = (
        runtime
        .workspace_service_logs(
            "not-real"
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"

    assert len(
        calls
    ) == 1


def test_generic_process_cannot_bypass_ps_tool():
    result = (
        evaluate_process_policy(
            executable="ps",
            args=[
                "-ef",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_generic_process_cannot_bypass_docker_tool():
    result = (
        evaluate_process_policy(
            executable="docker",
            args=[
                "compose",
                "ps",
            ],
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_runtime_tools_are_read_only():
    for tool_name in [
        "workspace_process_snapshot",
        "workspace_service_status",
        "workspace_service_logs",
    ]:
        tool = (
            get_tool(
                tool_name
            )
        )

        assert tool is not None

        assert tool[
            "risk"
        ] == "read"

        assert tool[
            "requires_approval"
        ] is False


def test_service_logs_ground_service_name():
    tool = (
        get_tool(
            "workspace_service_logs"
        )
    )

    assert tool is not None

    assert tool[
        "grounded_arguments"
    ] == [
        "service_name"
    ]


class FakeMCPServer:
    def __init__(
        self,
    ):
        self.tool_names = []

    def tool(
        self,
    ):
        def decorator(
            function,
        ):
            self.tool_names.append(
                function.__name__
            )

            return function

        return decorator


def test_developer_mcp_registers_complete_feature_group():
    server = (
        FakeMCPServer()
    )

    register_developer_tools(
        server
    )

    assert set(
        server.tool_names
    ) == {
        "workspace_project_info",
        "workspace_run_tests",
        "workspace_run_build",
        "workspace_process_snapshot",
        "workspace_service_status",
        "workspace_service_logs",
    }