"""The account a tool runs for crosses into the MCP subprocess (owner decision, 17 November).

Each Janus account connects its own Proxmox and Tailscale, so a tool must know
whose request it serves. The trusted host sends the owner as request metadata
(never as a model argument), and the server makes it the tool's current owner.
These fixtures exercise the installed SDK over stdio with temporary files only.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agent.mcp_client import MCPRuntime, OWNER_META_KEY
from services.extensions.context import request_owner

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _server(tmp_path: Path) -> Path:
    script = tmp_path / "owner_fixture.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from tools.runtime_schema import ContractAwareMCPServer\n"
        "from services.extensions.context import current_owner\n"
        'server = ContractAwareMCPServer("owner-fixture")\n'
        "@server.tool()\n"
        "def whose_request(probe: str) -> dict[str, object]:\n"
        '    return {"owner": current_owner(), "probe": probe}\n'
        "@server.tool()\n"
        "async def whose_async_request(probe: str) -> dict[str, object]:\n"
        '    return {"owner": current_owner(), "probe": probe}\n'
        "server.run()\n",
        encoding="utf-8",
    )
    return script


def test_the_requesting_owner_reaches_sync_and_async_tools_and_never_leaks_between_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", "")
    script = _server(tmp_path)

    async def run():
        runtime = MCPRuntime(server_path=script)
        try:
            await runtime.start()
            token = request_owner.set("alice")
            try:
                alice = await runtime.call_tool("whose_request", {"probe": "a"})
                alice_async = await runtime.call_tool("whose_async_request", {"probe": "b"})
            finally:
                request_owner.reset(token)
            nobody = await runtime.call_tool("whose_request", {"probe": "c"})
            token = request_owner.set("bob")
            try:
                bob = await runtime.call_tool("whose_request", {"probe": "d"})
            finally:
                request_owner.reset(token)
            return alice, alice_async, nobody, bob
        finally:
            await runtime.stop()

    alice, alice_async, nobody, bob = asyncio.run(run())
    assert alice == {"owner": "alice", "probe": "a"}
    assert alice_async == {"owner": "alice", "probe": "b"}
    assert nobody == {"owner": None, "probe": "c"}
    assert bob == {"owner": "bob", "probe": "d"}


def test_an_explicit_owner_wins_over_the_ambient_one(tmp_path, monkeypatch):
    """An approved action runs for the owner who approved it, whatever request decides it."""
    monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", "")
    script = _server(tmp_path)

    async def run():
        runtime = MCPRuntime(server_path=script)
        try:
            await runtime.start()
            token = request_owner.set("bob")
            try:
                return await runtime.call_tool("whose_request", {"probe": "e"}, owner="alice")
            finally:
                request_owner.reset(token)
        finally:
            await runtime.stop()

    assert asyncio.run(run()) == {"owner": "alice", "probe": "e"}


def test_the_owner_travels_only_as_metadata_never_as_an_argument():
    sent = []

    class Client:
        async def call_tool(self, name, arguments, *, meta=None):
            sent.append((name, dict(arguments), meta))

            class Result:
                is_error = False
                structured_content = {"ok": True}
            return Result()

    runtime = MCPRuntime()
    runtime._client = Client()
    token = request_owner.set("alice")
    try:
        asyncio.run(runtime.call_tool("whose_request", {"probe": "f"}))
    finally:
        request_owner.reset(token)
    asyncio.run(runtime.call_tool("whose_request", {"probe": "g"}))
    assert sent == [("whose_request", {"probe": "f"}, {OWNER_META_KEY: "alice"}),
                    ("whose_request", {"probe": "g"}, None)]
    assert OWNER_META_KEY.startswith("janus.")
    assert sys.modules["agent.mcp_client"]
