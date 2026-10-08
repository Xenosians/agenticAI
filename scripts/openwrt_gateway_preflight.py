"""Real ToolGateway -> MCP -> OpenWrt read contract check, without model inference."""
import asyncio
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.mcp_client import MCPRuntime
from subagents.core.definitions.loader import load_agent_definition
from subagents.core.tooling.gateway import ToolGateway


async def main():
    def approval(*args, **kwargs):
        raise RuntimeError("Read-only preflight must never create approval.")
    mcp = MCPRuntime()
    agent = load_agent_definition(ROOT / "subagents/agents/networking-specialist.md")
    gateway = ToolGateway(approval_creator=approval, mcp=mcp)
    results = []
    await mcp.start()
    try:
        for tool, args, request in [
            ("network_system_info", {}, "Read router identity."),
            ("network_interface_status", {"interface":"lan"}, "Inspect logical interface lan."),
            ("network_firewall_config", {}, "Read configured firewall policy."),
        ]:
            result = await gateway.execute(agent, request, tool, args)
            results.append({"tool":tool,"ok":result.get("ok"),"decision_code":result.get("decision_code")})
        rejected = await gateway.execute(agent, "Read router identity.", "network_system_info", {"device_id":"invented"})
        results.append({"tool":"invalid_argument_proposal", "ok":rejected.get("ok") is False and rejected.get("decision_code") == "unknown_arguments", "decision_code":rejected.get("decision_code")})
    finally:
        await mcp.stop()
    report = {"schema":"openwrt-gateway-preflight.v1", "model_inference":False,
        "live_provider":True, "cases":results, "passed":all(r["ok"] for r in results)}
    path = ROOT / ".runtime/openwrt-gateway-preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
