import inspect

import httpx

from config import Settings
from services.operations import OperationsService
from subagents.core.definitions.loader import load_agent_directory
from tools.operations.catalog import (
    OPERATIONS_TOOLS,
    format_container_logs_result,
    format_container_status_result,
    format_service_health_result,
)
from tools.operations.mcp import register_operations_tools
from tools.registry import TOOLS


READ_TOOLS = {"container_status", "container_logs", "service_health"}


class FakeMCPServer:

    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(function):
            self.functions[function.__name__] = function
            return function
        return decorator


def registered(service=None, **overrides):
    server = FakeMCPServer()
    settings = Settings(_env_file=None, **overrides)
    register_operations_tools(server, settings, service)
    return server.functions


def test_read_tools_are_registered_with_exact_signatures():
    functions = registered()
    assert READ_TOOLS <= set(functions)
    signatures = {name: {key: p.default for key, p in inspect.signature(functions[name]).parameters.items()}
                  for name in READ_TOOLS}
    assert signatures == {
        "container_status": {"container": None},
        "container_logs": {"container": inspect.Parameter.empty, "lines": None},
        "service_health": {},
    }
    notify = {key: p.default for key, p in inspect.signature(functions["notify_user"]).parameters.items()}
    # dedupe_key is the trusted policy argument; the model never supplies it.
    assert notify == {"email": inspect.Parameter.empty, "message": inspect.Parameter.empty, "subject": None,
                      "dedupe_key": None}
    assert set(functions) == READ_TOOLS | {"notify_user"}


def test_catalog_entries_are_reads_without_approval_and_owned_by_operations():
    agents = {agent.name: agent for agent in load_agent_directory("subagents/agents")}
    owners = {name: sorted(agent.name for agent in agents.values() if name in agent.tools) for name in OPERATIONS_TOOLS}
    assert owners["notify_user"] == ["operations-specialist"]
    for name in READ_TOOLS:
        spec = OPERATIONS_TOOLS[name]
        assert TOOLS[name] is spec
        assert spec["risk"] == "read" and spec["requires_approval"] is False and spec["effect"] == "read"
        assert spec["description"].startswith("Read only.")
        assert "approval_formatter" not in spec and "policy_resolver" not in spec
        assert callable(spec["result_formatter"])
        assert owners[name] == ["operations-specialist"]
    assert TOOLS["container_logs"]["grounded_arguments"] == ["container"]
    agent = agents["operations-specialist"]
    assert agent.model == "hub-main" and agent.max_steps == 2
    assert not any(name in agent.system_prompt for name in OPERATIONS_TOOLS)


def test_unconfigured_tools_return_structured_errors():
    # No allowlist, no Phoenix URL and no AI host: nothing may be probed.
    functions = registered(ai_host="")
    status = functions["container_status"]()
    assert status.ok is False and status.error_code == "not_configured"
    logs = functions["container_logs"](container="itsm-surrealdb")
    assert logs.ok is False and logs.error_code == "not_configured"
    health = functions["service_health"]()
    assert health.ok is False and health.error_code == "not_configured"
    assert format_container_status_result(status.model_dump()).startswith("The container status read did not succeed")


def test_mcp_results_are_structured():
    def docker(request):
        if request.url.path.endswith("/logs"):
            payload = b"ready\n"
            return httpx.Response(200, content=bytes([1, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload)
        return httpx.Response(200, json={"State": {"Status": "running", "Running": True,
                                                    "StartedAt": "2026-10-18T00:00:00Z", "Health": {"Status": "healthy"}},
                                          "RestartCount": 0, "Config": {"Image": "surrealdb/surrealdb:v2", "Tty": False}})

    settings = Settings(_env_file=None, operations_containers="itsm-surrealdb", phoenix_base_url="http://127.0.0.1:4000")
    service = OperationsService(settings, docker_transport=httpx.MockTransport(docker),
                                health_transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    functions = registered(service)
    status = functions["container_status"]().model_dump()
    assert status["data"]["containers"][0]["health"] == "healthy"
    assert "itsm-surrealdb: running, health healthy" in format_container_status_result(status)
    logs = functions["container_logs"](container="itsm-surrealdb", lines=5).model_dump()
    assert format_container_logs_result(logs) == "Last 1 log line(s) of container itsm-surrealdb:\nready"
    health = functions["service_health"]().model_dump()
    text = format_service_health_result(health)
    assert text.startswith("2 of 2 platform service(s) up.") and "phoenix: up (HTTP 200" in text


def test_formatters_report_down_services_and_redaction():
    text = format_service_health_result({"ok": True, "data": {"count": 2, "up": 0, "services": [
        {"name": "phoenix", "up": False, "http_status": 503, "latency_ms": 3.2},
        {"name": "ai", "up": False, "http_status": None, "error": "connection_failed"}]}})
    assert "phoenix: down (HTTP 503, 3.2 ms)" in text and "ai: down (connection_failed)" in text
    logs = format_container_logs_result({"ok": True, "data": {"container": "c", "line_count": 0, "logs": "",
                                                              "truncated": True, "redaction_applied": True}})
    assert "older output was cut" in logs and "redacted" in logs and "(no log output)" in logs
