"""Engineering's internal HTTP boundary: only Phoenix may call it; it only reads."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import engineering, extensions
from tests.unit.services.test_engineering_netlist import KICAD_7
from tests.unit.services.test_engineering_topology import design

TOKEN = "test-internal-secret"
HEADERS = {"x-internal-token": TOKEN}
DRIFT, NETLIST = "/internal/v1/engineering/topology/drift", "/internal/v1/engineering/netlist"


class Containers:
    def container_status(self):
        return {"ok": True, "status": "success", "provider": "docker", "observed_at": "x", "data": {"containers": [
            {"name": "itsm-surrealdb", "found": True, "state": "exited", "health": None, "restart_count": 1,
             "image": "surrealdb/surrealdb:v2"}]}}


class Endpoints:
    def service_health(self):
        return {"ok": True, "status": "success", "provider": "http", "observed_at": "x", "data": {"services": [
            {"name": "phoenix", "url": "http://127.0.0.1:4000/api/health", "up": True, "http_status": 200}]}}


@pytest.fixture
def client(monkeypatch):
    settings = SimpleNamespace(require_internal_job_token=lambda: TOKEN)
    monkeypatch.setattr(extensions, "get_settings", lambda: settings)
    monkeypatch.setattr(engineering, "observation_sources",
                        lambda: {"containers": Containers(), "endpoints": Endpoints(), "router": None,
                                 "default_interfaces": ["lan"]})
    app = FastAPI()
    app.include_router(engineering.router)
    return TestClient(app)


def drift_body(**changes):
    body = {"topology_id": "a" * 32, "version": 2, "topology": design()}
    body.update(changes)
    return body


def test_both_routes_require_the_internal_token(client):
    for path, body in ((DRIFT, drift_body()), (NETLIST, {"netlist": KICAD_7})):
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"x-internal-token": "wrong"}).status_code == 401


def test_drift_observes_and_compares_the_supplied_version(client):
    response = client.post(DRIFT, json=drift_body(), headers=HEADERS)
    assert response.status_code == 200
    report = response.json()
    assert report["schema"] == "topology-drift.v1" and report["state_kind"] == "simulated"
    assert report["topology_id"] == "a" * 32 and report["topology_version"] == 2
    assert report["execution_enabled"] is False and report["apply_to_live"] is False
    rows = {row["id"]: row for row in report["nodes"]}
    assert rows["container:itsm-surrealdb"]["status"] == "drifted"
    assert rows["endpoint:phoenix"]["status"] == "matches"
    assert rows["router:openwrt"]["reason"] == "source_not_configured"
    assert "127.0.0.1" not in response.text


def test_drift_without_a_topology_returns_the_observation_only(client):
    response = client.post(DRIFT, json=drift_body(topology={"nodes": [], "edges": []}), headers=HEADERS)
    assert response.json()["not_in_design"] == ["container:itsm-surrealdb", "endpoint:phoenix"]


@pytest.mark.parametrize("body", [
    drift_body(topology_id="../x"), drift_body(version=0), drift_body(version=True), drift_body(owner="alice"),
    drift_body(topology={"nodes": [{"id": "bad id", "kind": "x", "label": "", "properties": {}}], "edges": []}),
    {"topology_id": "a" * 32, "version": 1},
])
def test_invalid_drift_requests_are_refused_with_a_stable_code(client, body):
    response = client.post(DRIFT, json=body, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["detail"] in {"invalid_topology", "invalid_engineering_request"}


def test_private_topology_content_has_its_own_code(client):
    topology = design()
    topology["nodes"][0]["properties"]["admin_password"] = "x"
    response = client.post(DRIFT, json=drift_body(topology=topology), headers=HEADERS)
    assert response.status_code == 422 and response.json()["detail"] == "private_topology_content"


def test_netlist_inspection(client):
    response = client.post(NETLIST, json={"netlist": KICAD_7}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["summary"]["components"] == 3
    assert response.json()["actuation"] == "none"


@pytest.mark.parametrize("body", [{"netlist": "(not a netlist"}, {"netlist": 3}, {}, {"netlist": KICAD_7, "path": "/etc"}])
def test_invalid_netlists_are_refused(client, body):
    response = client.post(NETLIST, json=body, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["detail"] in {"invalid_netlist", "invalid_engineering_request"}


def test_observation_sources_are_built_from_the_fixed_read_adapters(monkeypatch):
    settings = SimpleNamespace(openwrt_base_url=None, openwrt_username=None, openwrt_password=None,
                               operations_containers=["itsm-surrealdb"], engineering_router_interfaces=["lan", "wan"])
    monkeypatch.setattr(engineering, "get_settings", lambda: settings)
    built = []
    monkeypatch.setattr(engineering, "OperationsService", lambda value: built.append(value) or "operations")
    monkeypatch.setattr(engineering, "build_openwrt_service", lambda value: built.append(value) or None)
    sources = engineering.observation_sources()
    assert sources == {"containers": "operations", "endpoints": "operations", "router": None,
                       "default_interfaces": ["lan", "wan"]}
    assert built == [settings, settings]


@pytest.mark.parametrize("value,expected", [("lan, wan,lan", ["lan", "wan"]), ("", [])])
def test_router_interface_setting_parses_names(value, expected):
    from config import Settings
    assert Settings(_env_file=None, engineering_router_interfaces=value).engineering_router_interfaces == expected


@pytest.mark.parametrize("value", ["lan;reboot", "a" * 33, ",".join(f"if{i}" for i in range(17))])
def test_router_interface_setting_refuses_invalid_names(value):
    from pydantic import ValidationError
    from config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None, engineering_router_interfaces=value)
