"""Observation through the fixed read adapters only: synthetic services, no network."""
from datetime import datetime, timezone

import httpx
import pytest

from services.engineering.observe import observe

NOW = datetime(2026, 10, 29, 1, 0, tzinfo=timezone.utc)


class Containers:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def container_status(self):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Endpoints:
    def __init__(self, result):
        self.result = result

    def service_health(self):
        return self.result


class Router:
    def __init__(self, interfaces=None, board=None, fail_board=None):
        self.interfaces = interfaces or {}
        self.board = board or {"hostname": "lab-router", "kernel": "6.6.0",
                               "release": {"version": "24.10.0", "target": "x86/64"}, "system": "lab"}
        self.fail_board, self.asked = fail_board, []

    def system_info(self):
        if self.fail_board:
            raise self.fail_board
        return {"ok": True, "status": "success", "provider": "openwrt", "observed_at": "x", "data": self.board}

    def interface_status(self, name):
        self.asked.append(name)
        if name not in self.interfaces:
            raise RuntimeError("OpenWrt rejected the read or returned an invalid RPC response.")
        return {"ok": True, "status": "success", "provider": "openwrt", "observed_at": "x", "data": self.interfaces[name]}


def container_result(*records):
    return {"ok": True, "status": "success", "provider": "docker", "observed_at": "x",
            "data": {"containers": list(records), "count": len(records)}}


def record(name, *, found=True, state="running", health="healthy"):
    if not found:
        return {"name": name, "found": False, "state": "not_found", "health": None, "started_at": None,
                "uptime_seconds": None, "exit_code": None, "restart_count": None, "image": None}
    return {"name": name, "found": True, "state": state, "health": health, "started_at": "2026-10-29T00:00:00+00:00",
            "uptime_seconds": 60, "exit_code": None, "restart_count": 2, "image": "surrealdb/surrealdb:v2"}


HEALTH = {"ok": True, "status": "success", "provider": "http", "observed_at": "x",
          "data": {"services": [{"name": "phoenix", "url": "http://127.0.0.1:4000/api/health", "up": True,
                                 "http_status": 200, "latency_ms": 3.1, "error": None},
                                {"name": "ai", "url": "http://127.0.0.1:8000/health", "up": False,
                                 "http_status": None, "latency_ms": None, "error": "connection_failed"}]}}
LAN = {"up": True, "available": True, "proto": "static", "device": "br-lan", "uptime": 300,
       "ipv4-address": [{"address": "192.168.1.1", "mask": 24}]}


def run(**overrides):
    values = dict(containers=Containers(container_result(record("itsm-surrealdb"), record("itsm-git-lab", found=False))),
                  endpoints=Endpoints(HEALTH), router=Router({"lan": LAN}), topology=None,
                  default_interfaces=["lan", "wan"], now=NOW)
    values.update(overrides)
    return observe(**values)


def by_id(result):
    return {node["id"]: node for node in result["nodes"]}


def test_reads_every_source_into_labelled_observed_state():
    result = run()
    assert result["schema"] == "infrastructure-observation.v1" and result["state_kind"] == "observed"
    assert result["observed_at"] == "2026-10-29T01:00:00+00:00"
    assert [(row["id"], row["state"], row["error_code"]) for row in result["sources"]] == [
        ("containers", "observed", None), ("endpoints", "observed", None), ("router", "observed", None)]
    nodes = by_id(result)
    assert nodes["container:itsm-surrealdb"]["properties"] == {
        "state": "running", "health": "healthy", "restart_count": 2, "image": "surrealdb/surrealdb:v2"}
    assert nodes["endpoint:phoenix"]["properties"] == {"up": True, "http_status": 200}
    assert nodes["endpoint:ai"]["properties"] == {"up": False, "http_status": None}
    assert nodes["router:lab-router"]["properties"] == {
        "hostname": "lab-router", "kernel": "6.6.0", "release": "24.10.0", "target": "x86/64"}
    assert nodes["interface:lan"]["properties"] == {
        "up": True, "available": True, "proto": "static", "device": "br-lan", "ipv4_address": "192.168.1.1/24"}
    assert result["edges"] == [{"from": "interface:lan", "to": "router:lab-router", "relation": "hosted_on"}]
    # Docker said the allow-listed container does not exist; the refused interface is unread, not absent.
    assert result["absent"] == ["container:itsm-git-lab"]
    assert result["unread"] == ["interface:wan"]


def test_endpoint_urls_never_leave_the_observation():
    assert "127.0.0.1" not in str(run())


def test_designed_interfaces_are_read_in_addition_to_the_defaults():
    router = Router({"lan": LAN, "guest": {**LAN, "device": "br-guest"}})
    topology = {"nodes": [{"id": "interface:guest", "kind": "interface", "label": "", "properties": {}}], "edges": []}
    result = run(router=router, topology=topology)
    assert sorted(router.asked) == ["guest", "lan", "wan"]
    assert "interface:guest" in by_id(result)


def test_invalid_interface_names_are_never_sent_to_the_router():
    router = Router({"lan": LAN})
    topology = {"nodes": [{"id": "interface:bad.name", "kind": "interface", "label": "", "properties": {}}], "edges": []}
    result = run(router=router, topology=topology)
    assert "bad.name" not in router.asked
    assert "interface:bad.name" in result["unread"]


def test_interface_reads_are_bounded():
    topology = {"nodes": [{"id": f"interface:if{i}", "kind": "interface", "label": "", "properties": {}}
                          for i in range(40)], "edges": []}
    router = Router({})
    result = run(router=router, topology=topology, default_interfaces=[])
    assert len(router.asked) == 16
    assert len(result["unread"]) == 40


@pytest.mark.parametrize("failure,code", [
    (httpx.ConnectTimeout("slow"), "provider_timeout"),
    (httpx.ConnectError("refused http://user:pw@host"), "provider_unavailable"),
    (RuntimeError("OpenWrt rejected the read"), "provider_unavailable"),
])
def test_a_failed_router_read_marks_only_that_source_unavailable(failure, code):
    result = run(router=Router({"lan": LAN}, fail_board=failure))
    sources = {row["id"]: row for row in result["sources"]}
    assert sources["router"] == {"id": "router", "state": "unavailable", "error_code": code}
    assert sources["containers"]["state"] == "observed"
    assert not any(node["id"].startswith(("router:", "interface:")) for node in result["nodes"])
    assert result["unread"] == []
    assert "user:pw" not in str(result)


def test_unconfigured_sources_are_reported_as_not_configured():
    not_configured = {"ok": False, "status": "error", "provider": "docker", "error_code": "not_configured",
                      "error": "OPERATIONS_CONTAINERS is not configured."}
    result = run(containers=Containers(not_configured), router=None,
                 endpoints=Endpoints({**not_configured, "provider": "http"}))
    assert [(row["id"], row["state"]) for row in result["sources"]] == [
        ("containers", "not_configured"), ("endpoints", "not_configured"), ("router", "not_configured")]
    assert result["nodes"] == [] and result["absent"] == []


def test_a_provider_error_keeps_its_stable_code_and_drops_its_message():
    failed = {"ok": False, "status": "error", "provider": "docker", "error_code": "provider_error",
              "error": "Docker Engine refused the read (HTTP 500) token=abc"}
    result = run(containers=Containers(failed))
    assert result["sources"][0] == {"id": "containers", "state": "unavailable", "error_code": "provider_error"}
    assert "token=abc" not in str(result)


def test_an_adapter_exception_is_contained_to_its_source():
    result = run(containers=Containers(OSError("socket path /var/run/docker.sock")))
    assert result["sources"][0] == {"id": "containers", "state": "unavailable", "error_code": "provider_unavailable"}
    assert "docker.sock" not in str(result)
    assert "endpoint:phoenix" in by_id(result)


def test_unknown_error_codes_are_not_forwarded():
    failed = {"ok": False, "status": "error", "provider": "docker", "error_code": "x" * 300, "error": "boom"}
    assert run(containers=Containers(failed))["sources"][0]["error_code"] == "provider_error"


def test_malformed_adapter_records_are_skipped_not_invented():
    malformed = container_result({"name": "../escape", "found": True, "state": "running"},
                                 {"found": True}, "not-a-record", record("itsm-surrealdb"))
    result = run(containers=Containers(malformed))
    assert [node["id"] for node in result["nodes"] if node["kind"] == "container"] == ["container:itsm-surrealdb"]


def test_observed_text_is_bounded_and_single_line():
    board = {"hostname": "lab-router", "kernel": "k" * 500, "release": {"version": "24\n10", "target": "x86/64"}}
    props = by_id(run(router=Router({"lan": LAN}, board=board)))["router:lab-router"]["properties"]
    assert len(props["kernel"]) == 200 and props["release"] == "24 10"


def test_the_observation_hash_covers_its_content():
    first, second = run(), run(endpoints=Endpoints({**HEALTH, "data": {"services": HEALTH["data"]["services"][:1]}}))
    assert first["observation_sha256"] != second["observation_sha256"]
    assert run()["observation_sha256"] == first["observation_sha256"]
    assert len(first["observation_sha256"]) == 64
