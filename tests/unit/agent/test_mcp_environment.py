"""The trusted MCP server needs app settings; Lua children inherit no credentials."""
import os
import asyncio
from pathlib import Path

from agent import mcp_client
from services.extensions.transform import worker_environment


def test_env_only_deployment_reaches_trusted_server_but_not_lua(tmp_path, monkeypatch):
    server_file = tmp_path / "server.py"
    server_file.write_text("# fixture only\n")
    monkeypatch.setenv("JIRA_API_TOKEN", "synthetic-provider-credential")
    monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", "/fixture/machine.lua")
    monkeypatch.setenv("ITSM_LUA_ADAPTER_SHA256", "a" * 64)
    captured = []

    class Client:
        def __init__(self, parameters):
            captured.append(parameters)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            return None

    monkeypatch.setattr(mcp_client, "Client", Client)
    runtime = mcp_client.MCPRuntime(server_path=server_file)
    async def run():
        try:
            await runtime.start()
        finally:
            await runtime.stop()
    asyncio.run(run())
    parameters = captured[0]
    assert parameters.env["JIRA_API_TOKEN"] == "synthetic-provider-credential"
    assert parameters.env["ITSM_LUA_ADAPTER_FILE"] == "/fixture/machine.lua"
    assert parameters.env["ITSM_LUA_ADAPTER_SHA256"] == "a" * 64
    assert parameters.env is not os.environ
    assert Path(parameters.cwd) == mcp_client.PROJECT_ROOT
    assert not set(parameters.env) & {"JIRA_API_TOKEN", "ITSM_LUA_ADAPTER_FILE"} & set(worker_environment())


def test_actual_sdk_stdio_child_receives_env_only_configuration(tmp_path, monkeypatch):
    """Exercise the SDK's real environment filtering, without an app provider."""
    script = tmp_path / "probe_server.py"
    script.write_text('''import os
from pathlib import Path
from mcp.server import MCPServer
server = MCPServer("configuration-fixture")
@server.tool()
def probe_environment() -> dict[str, bool]:
    return {"configured": os.environ.get("ITSM_SYNTHETIC_SETTING") == "fixture-only",
            "cwd_matches": str(Path.cwd()) == os.environ.get("ITSM_SYNTHETIC_ROOT")}
server.run()
''')
    monkeypatch.setenv("ITSM_SYNTHETIC_SETTING", "fixture-only")
    monkeypatch.setenv("ITSM_SYNTHETIC_ROOT", str(mcp_client.PROJECT_ROOT))

    async def run():
        runtime = mcp_client.MCPRuntime(server_path=script)
        try:
            await runtime.start()
            return await runtime.call_tool("probe_environment", {})
        finally:
            await runtime.stop()
    result = asyncio.run(run())
    assert result == {"configured": True, "cwd_matches": True}
