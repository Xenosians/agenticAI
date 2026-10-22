import json

import httpx
import pytest
from pydantic import ValidationError

from config import Settings
from services.operations import MAX_LOG_BYTES, OperationsService, _log_payload, health_targets


ALLOWED = "itsm-surrealdb,itsm-git-lab"
STARTED = "2026-10-18T01:00:00.123456789Z"


def settings(**overrides):
    values = {"operations_containers": ALLOWED, "phoenix_base_url": "http://127.0.0.1:4000",
              "itsm_internal_job_token": "internal-secret"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def container(name, *, status="running", health=None, tty=False, restarts=0, image="surrealdb/surrealdb:v2"):
    state = {"Status": status, "Running": status == "running", "StartedAt": STARTED, "ExitCode": 0 if status == "running" else 137}
    if health:
        state["Health"] = {"Status": health, "Log": [{"Output": "token=leak"}]}
    return {"Name": "/" + name, "State": state, "RestartCount": restarts,
            "Config": {"Image": image, "Tty": tty, "Env": ["PASSWORD=never-returned"]}}


def frames(*parts):
    out = b""
    for stream, text in parts:
        payload = text.encode()
        out += bytes([stream, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload
    return out


def docker(records, logs=None, calls=None, status=200):
    def handler(request):
        if calls is not None:
            calls.append((request.url.path, dict(request.url.params)))
        if status != 200:
            return httpx.Response(status, json={"message": "boom"})
        parts = request.url.path.split("/")
        name = parts[2]
        if name not in records:
            return httpx.Response(404, json={"message": "No such container"})
        if parts[3] == "logs":
            return httpx.Response(200, content=logs[name])
        return httpx.Response(200, json=records[name])
    return httpx.MockTransport(handler)


def test_container_status_reads_only_allow_listed_containers():
    calls = []
    transport = docker({"itsm-surrealdb": container("itsm-surrealdb", health="healthy", restarts=2),
                        "itsm-git-lab": container("itsm-git-lab", status="exited")}, calls=calls)
    result = OperationsService(settings(), docker_transport=transport).container_status()
    assert result["ok"] is True and result["provider"] == "docker"
    data = result["data"]
    assert (data["count"], data["running"], data["unhealthy"], data["not_found"]) == (2, 1, 0, 0)
    surreal, git = data["containers"]
    assert surreal["name"] == "itsm-surrealdb" and surreal["state"] == "running" and surreal["health"] == "healthy"
    assert surreal["restart_count"] == 2 and surreal["image"] == "surrealdb/surrealdb:v2"
    assert surreal["started_at"] == "2026-10-18T01:00:00.123456+00:00" and surreal["uptime_seconds"] >= 0
    assert git["state"] == "exited" and git["uptime_seconds"] is None and git["exit_code"] == 137
    # Only the two allow-listed names are inspected; no container listing, no env or health logs returned.
    assert [path for path, _ in calls] == ["/containers/itsm-surrealdb/json", "/containers/itsm-git-lab/json"]
    assert "never-returned" not in json.dumps(result) and "leak" not in json.dumps(result)


def test_container_status_one_container_and_missing_container():
    transport = docker({})
    result = OperationsService(settings(), docker_transport=transport).container_status("itsm-git-lab")
    assert result["ok"] is True
    assert result["data"]["containers"][0]["found"] is False and result["data"]["not_found"] == 1


def test_container_outside_allowlist_is_denied_before_any_docker_call():
    calls = []
    service = OperationsService(settings(), docker_transport=docker({}, calls=calls))
    for result in (service.container_status("frosty_faraday"), service.container_logs("frosty_faraday")):
        assert result["ok"] is False and result["status"] == "denied"
        assert result["error_code"] == "container_not_allowed"
    assert calls == []


def test_tools_report_not_configured_without_allowlist():
    service = OperationsService(settings(operations_containers=""), docker_transport=docker({}))
    result = service.container_status()
    assert result["ok"] is False and result["error_code"] == "not_configured"
    assert "OPERATIONS_CONTAINERS" in result["error"]


@pytest.mark.parametrize("transport, code", [
    (docker({}, status=500), "provider_error"),
    (httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused"))), "provider_unavailable"),
    (httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))), "provider_timeout"),
])
def test_provider_failures_are_errors_not_missing_containers(transport, code):
    result = OperationsService(settings(), docker_transport=transport).container_status()
    assert result["ok"] is False and result["status"] == "error" and result["error_code"] == code


def test_container_logs_demultiplexes_redacts_and_bounds_lines():
    calls = []
    stream = frames((1, "starting\n"), (2, "db password=hunter2 ready\n"), (1, "url https://bot:s3cret@example.test/x\n"))
    transport = docker({"itsm-surrealdb": container("itsm-surrealdb")}, logs={"itsm-surrealdb": stream}, calls=calls)
    result = OperationsService(settings(), docker_transport=transport).container_logs("itsm-surrealdb", 20)
    data = result["data"]
    assert result["ok"] is True and data["line_count"] == 3 and data["truncated"] is False
    assert data["logs"].splitlines() == ["starting", "db password=[REDACTED] ready", "url https://bot:[REDACTED]@example.test/x"]
    assert data["redaction_applied"] is True
    assert calls[-1] == ("/containers/itsm-surrealdb/logs", {"stdout": "1", "stderr": "1", "tail": "20"})


def test_container_logs_default_lines_and_tty_stream():
    calls = []
    transport = docker({"itsm-git-lab": container("itsm-git-lab", tty=True)},
                       logs={"itsm-git-lab": b"\x1b[32mgreen\x1b[0m line\nnext\n"}, calls=calls)
    result = OperationsService(settings(), docker_transport=transport).container_logs("itsm-git-lab")
    assert result["data"]["logs"] == "green line\nnext" and result["data"]["redaction_applied"] is False
    assert calls[-1][1]["tail"] == "50"


@pytest.mark.parametrize("lines", [0, 201, True, "20"])
def test_container_logs_rejects_out_of_range_line_counts(lines):
    calls = []
    service = OperationsService(settings(), docker_transport=docker({}, calls=calls))
    result = service.container_logs("itsm-surrealdb", lines)
    assert result["ok"] is False and result["error_code"] == "invalid_arguments" and calls == []


def test_container_logs_for_missing_container_is_not_found():
    result = OperationsService(settings(), docker_transport=docker({})).container_logs("itsm-git-lab")
    assert result["ok"] is False and result["error_code"] == "not_found"


def test_log_payload_keeps_newest_bytes_and_drops_the_cut_line():
    lines = "".join(f"line {index:05d} token=abcdef\n" for index in range(2000))
    payload, clipped = _log_payload([frames((1, lines))[i:i + 777] for i in range(0, len(frames((1, lines))), 777)],
                                    False, MAX_LOG_BYTES)
    assert clipped and len(payload) == MAX_LOG_BYTES and payload.endswith(b"line 01999 token=abcdef\n")
    transport = docker({"itsm-git-lab": container("itsm-git-lab")}, logs={"itsm-git-lab": frames((1, lines))})
    data = OperationsService(settings(), docker_transport=transport).container_logs("itsm-git-lab", 200)["data"]
    assert data["truncated"] is True and len(data["logs"].encode()) <= MAX_LOG_BYTES
    assert all(line.startswith("line ") and line.endswith("token=[REDACTED]") for line in data["logs"].splitlines())


def test_service_health_derives_targets_and_reports_each_service():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.port == 4000:
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503)

    service = OperationsService(settings(ai_host="0.0.0.0", ai_port=8000), health_transport=httpx.MockTransport(handler))
    result = service.service_health()
    assert result["ok"] is True and result["provider"] == "http"
    phoenix, ai = result["data"]["services"]
    assert phoenix["name"] == "phoenix" and phoenix["url"] == "http://127.0.0.1:4000/api/health"
    assert phoenix["up"] is True and phoenix["http_status"] == 200 and phoenix["latency_ms"] >= 0
    assert ai["url"] == "http://127.0.0.1:8000/health" and ai["up"] is False and ai["error"] == "unexpected_status"
    assert (result["data"]["up"], result["data"]["down"]) == (1, 1)
    # The internal job token and any other credential stay out of health probes.
    for request in seen:
        assert not {"authorization", "x-internal-token", "cookie"} & {key.lower() for key in request.headers}
        assert "internal-secret" not in str(request.headers)


def test_service_health_uses_configured_urls_and_reports_unreachable():
    def handler(request):
        if request.url.host == "slow.test":
            raise httpx.ConnectTimeout("slow")
        raise httpx.ConnectError("refused")

    configured = settings(operations_health_urls="phoenix=http://phoenix.test:4000/api/health,ai=http://slow.test/ready")
    assert health_targets(configured) == {"phoenix": "http://phoenix.test:4000/api/health", "ai": "http://slow.test/ready"}
    result = OperationsService(configured, health_transport=httpx.MockTransport(handler)).service_health()
    assert [(item["name"], item["up"], item["error"]) for item in result["data"]["services"]] == [
        ("phoenix", False, "connection_failed"), ("ai", False, "timeout")]


def test_service_health_without_any_endpoint_is_not_configured():
    result = OperationsService(settings(phoenix_base_url=None, ai_host="")).service_health()
    assert result["ok"] is False and result["error_code"] == "not_configured"


def test_operations_settings_parse_env_lists(monkeypatch):
    monkeypatch.setenv("OPERATIONS_CONTAINERS", " itsm-surrealdb, itsm-git-lab ,itsm-surrealdb")
    monkeypatch.setenv("OPERATIONS_HEALTH_URLS", "phoenix=http://127.0.0.1:4000/api/health")
    loaded = Settings(_env_file=None)
    assert loaded.operations_containers == ["itsm-surrealdb", "itsm-git-lab"]
    assert loaded.operations_health_urls == {"phoenix": "http://127.0.0.1:4000/api/health"}
    assert Settings(_env_file=None, operations_containers="").operations_containers == []


@pytest.mark.parametrize("field, value", [
    ("operations_containers", "bad name;rm"),
    ("operations_health_urls", "phoenix=http://user:pw@127.0.0.1:4000/api/health"),
    ("operations_health_urls", "phoenix=file:///etc/passwd"),
    ("operations_health_urls", "no-separator"),
])
def test_operations_settings_reject_unsafe_values(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
