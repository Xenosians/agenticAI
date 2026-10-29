"""Protocol evidence is distinct from a tool's actual domain outcome.

The stdio fixtures exercise the installed SDK with only temporary files. They
do not connect a provider or certify any real ITSM operation.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager
from agent.mcp_client import MCPRuntime, PROJECT_ROOT


def _server(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    marker = tmp_path / "effects.txt"
    script = tmp_path / "fixture_server.py"
    return_type = "str" if mode == "unstructured" else "dict[str, object]"
    response = {
        "raised": 'raise RuntimeError("synthetic post-effect error")',
        "unstructured": 'return "unstructured acknowledgement"',
        "denied": 'return {"ok": False, "status": "denied", "changed": False}',
        "success": 'return {"ok": True, "status": "success", "changed": True}',
    }[mode]
    effect = "" if mode == "denied" else f'Path({str(marker)!r}).open("a").write("effect\\n")'
    script.write_text(
        "from pathlib import Path\n"
        "from mcp.server import MCPServer\n"
        'server = MCPServer("isolated-tool-outcome-fixture")\n'
        "@server.tool()\n"
        f"def fixture_action(target: str) -> {return_type}:\n"
        '    assert target == "temporary-fixture"\n'
        f"    {effect or 'pass'}\n"
        f"    {response}\n"
        "server.run()\n",
        encoding="utf-8",
    )
    return script, marker


@pytest.mark.parametrize("mode", ["raised", "unstructured"])
def test_real_sdk_post_effect_failure_requires_reconciliation(tmp_path, mode):
    script, marker = _server(tmp_path, mode)
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()

    async def run():
        runtime = MCPRuntime(server_path=script)
        manager = ApprovalManager(
            store=store,
            mcp=runtime,
            tool_lookup=lambda _: {"requires_approval": True, "risk": "medium"},
        )
        approval = manager.create_approval(
            "fixture_action", {"target": "temporary-fixture"}, risk="medium",
        )
        try:
            await runtime.start()
            first = await manager.approve_approval(approval["id"])
            second = await manager.approve_approval(approval["id"])
            return first, second, store.get(approval["id"])
        finally:
            await runtime.stop()

    first, second, persisted = asyncio.run(run())
    assert marker.read_text() == "effect\n"
    assert first["ok"] is False
    assert first["result"]["status"] == "outcome_unknown"
    assert first["approval"]["status"] == "executing"
    assert persisted.status == "executing"
    assert persisted.result == first["result"]
    assert second["replayed"] is True
    assert second["result"] == first["result"]


@pytest.mark.parametrize("mode", ["denied", "success"])
def test_real_sdk_preserves_actual_typed_domain_receipts(tmp_path, mode):
    script, marker = _server(tmp_path, mode)

    async def run():
        runtime = MCPRuntime(server_path=script)
        try:
            await runtime.start()
            return await runtime.call_tool("fixture_action", {"target": "temporary-fixture"})
        finally:
            await runtime.stop()

    result = asyncio.run(run())
    expected = mode == "success"
    assert result == {"ok": expected, "status": mode, "changed": expected}
    assert marker.exists() is expected


@pytest.mark.parametrize("reason", ["transport", "invalid_shape", "unstructured", "tool_error"])
def test_unconfirmed_protocol_outcomes_are_sanitized_and_never_retried(reason):
    calls = []

    class Client:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            if reason == "transport":
                raise RuntimeError("synthetic-secret-must-not-leak")
            return SimpleNamespace(
                is_error=reason == "tool_error",
                structured_content=None if reason == "unstructured" else [],
            )

    runtime = MCPRuntime()
    runtime._client = Client()
    result = asyncio.run(runtime.call_tool("fixture_action", {}))
    assert result["ok"] is False
    assert result["status"] == "outcome_unknown"
    assert result["error_code"].startswith("mcp_")
    assert "synthetic-secret" not in str(result)
    assert len(calls) == 1


def test_cancelled_call_is_not_converted_to_a_terminal_failure():
    class Client:
        async def call_tool(self, *_):
            raise asyncio.CancelledError

    runtime = MCPRuntime()
    runtime._client = Client()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runtime.call_tool("fixture_action", {}))


def test_registered_production_schemas_and_lua_scoped_read_use_real_sdk(tmp_path, monkeypatch):
    """Metadata parity is not live acceptance of the registered providers."""
    from mcp_server import register_capability_signatures
    from tools.capability_contract import normalize_capability_catalog
    from tools.registry import TOOLS
    from tools import runtime_schema

    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True, capture_output=True)
    (workspace / "readme.txt").write_text("isolated registered read\n")
    (workspace / ".env").write_text("SYNTHETIC_SECRET=must-not-be-read\n")
    monkeypatch.setenv("PROCESS_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("GIT_REPOSITORIES", json.dumps({"fixture": str(workspace)}))
    monkeypatch.setenv("GIT_DEFAULT_REPOSITORY", "fixture")
    # No child may discover the shared deployment dotenv, even through a
    # workspace helper's lazily loaded Settings instance.
    script = tmp_path / "registered_fixture.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from config import Settings\n"
        "Settings.model_config['env_file'] = None\n"
        "from mcp_server import create_mcp_server\n"
        "settings = Settings(_env_file=None, directory_backend='mock', "
        "ticketing_backend='mock', asset_backend='mock', knowledge_backend='mock')\n"
        "create_mcp_server(settings=settings).run()\n",
        encoding="utf-8",
    )

    async def run():
        runtime = MCPRuntime(server_path=script)
        try:
            await runtime.start()
            listing = await runtime._client.list_tools()
            read = await runtime.call_tool(
                "workspace_read_text", {"relative_path": "readme.txt", "repository": "fixture"},
            )
            denied = await runtime.call_tool(
                "workspace_read_text", {"relative_path": ".env", "repository": "fixture"},
            )
            return listing, read, denied
        finally:
            await runtime.stop()

    listing, read, denied = asyncio.run(run())
    register_capability_signatures()
    catalog = normalize_capability_catalog(TOOLS)
    assert listing.next_cursor is None
    assert {tool.name for tool in listing.tools} == set(catalog)
    for tool in listing.tools:
        contract = catalog[tool.name]
        assert set(tool.input_schema["properties"]) == set(contract["execution_parameters"])
        assert set(contract["required_arguments"]) <= set(tool.input_schema.get("required", []))
        assert tool.output_schema is not None, tool.name
    assert read["ok"] is True, read.get("error")
    assert read["repository"] == "fixture"
    assert read["content"] == "isolated registered read\n"
    assert denied["ok"] is False
    assert denied["content"] is None
    assert "must-not-be-read" not in str(denied)
