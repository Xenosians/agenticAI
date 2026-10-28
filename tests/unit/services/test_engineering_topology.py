"""Desired topologies and advisory drift (FR-093, V19-09): synthetic data only."""
from copy import deepcopy

import pytest

from services.engineering.topology import TopologyError, compare, parse_topology


def design():
    return {
        "nodes": [
            {"id": "router:openwrt", "kind": "router", "label": "Edge router", "properties": {"release": "24.10.0"}},
            {"id": "interface:lan", "kind": "interface", "label": "LAN", "properties": {"up": True}},
            {"id": "container:itsm-surrealdb", "kind": "container", "label": "Database",
             "properties": {"state": "running", "health": "healthy"}},
            {"id": "endpoint:phoenix", "kind": "service_endpoint", "label": "Backend", "properties": {"up": True}},
            {"id": "endpoint:ai", "kind": "service_endpoint", "label": "AI", "properties": {"up": True}},
            {"id": "service:ticket-desk", "kind": "service", "label": "Ticket desk", "properties": {}},
        ],
        "edges": [
            {"from": "interface:lan", "to": "router:openwrt", "relation": "hosted_on"},
            {"from": "endpoint:phoenix", "to": "container:itsm-surrealdb", "relation": "depends_on"},
            {"from": "endpoint:ai", "to": "endpoint:phoenix", "relation": "depends_on"},
            {"from": "service:ticket-desk", "to": "endpoint:ai", "relation": "depends_on"},
        ],
    }


def observation(**changes):
    nodes = {
        "router:openwrt": {"id": "router:openwrt", "kind": "router", "label": "OpenWrt",
                           "properties": {"hostname": "lab-router", "release": "24.10.0"}},
        "interface:lan": {"id": "interface:lan", "kind": "interface", "label": "lan",
                          "properties": {"up": True, "proto": "static"}},
        "container:itsm-surrealdb": {"id": "container:itsm-surrealdb", "kind": "container", "label": "itsm-surrealdb",
                                     "properties": {"state": "running", "health": "healthy", "restart_count": 3}},
        "container:itsm-git-lab": {"id": "container:itsm-git-lab", "kind": "container", "label": "itsm-git-lab",
                                   "properties": {"state": "running", "health": None, "restart_count": 0}},
        "endpoint:phoenix": {"id": "endpoint:phoenix", "kind": "service_endpoint", "label": "phoenix",
                             "properties": {"up": True, "http_status": 200}},
        "endpoint:ai": {"id": "endpoint:ai", "kind": "service_endpoint", "label": "ai",
                        "properties": {"up": True, "http_status": 200}},
    }
    observed = {
        "schema": "infrastructure-observation.v1", "state_kind": "observed",
        "observed_at": "2026-10-29T01:00:00+00:00", "observation_sha256": "a" * 64,
        "sources": [{"id": "containers", "state": "observed", "error_code": None},
                    {"id": "endpoints", "state": "observed", "error_code": None},
                    {"id": "router", "state": "observed", "error_code": None}],
        "nodes": list(nodes.values()), "edges": [], "absent": [], "unread": [],
    }
    for key, value in changes.items():
        observed[key] = value(observed[key]) if callable(value) else value
    return observed


def rows(report):
    return {row["id"]: row for row in report["nodes"]}


def test_matching_design_reports_no_drift_and_never_enables_execution():
    report = compare(parse_topology(design()), observation(), topology_id="t" * 32, version=3)
    assert report["state_kind"] == "simulated"
    assert report["execution_enabled"] is False and report["apply_to_live"] is False
    assert report["topology_id"] == "t" * 32 and report["topology_version"] == 3
    assert report["observation_sha256"] == "a" * 64
    statuses = {key: row["status"] for key, row in rows(report).items()}
    assert statuses == {"router:openwrt": "matches", "interface:lan": "matches", "container:itsm-surrealdb": "matches",
                        "endpoint:phoenix": "matches", "endpoint:ai": "matches", "service:ticket-desk": "unknown"}
    assert report["potentially_affected"] == []
    assert report["not_in_design"] == ["container:itsm-git-lab"]
    assert report["counts"] == {"matches": 5, "drifted": 0, "missing": 0, "unknown": 1}


def test_only_declared_properties_are_compared():
    # restart_count and http_status are observed but not declared, so they never count as drift.
    report = compare(parse_topology(design()), observation(), topology_id="t" * 32, version=1)
    assert rows(report)["container:itsm-surrealdb"]["differences"] == []


def test_drift_lists_expected_and_observed_values_and_reaches_dependents():
    def stopped(nodes):
        nodes = deepcopy(nodes)
        for node in nodes:
            if node["id"] == "container:itsm-surrealdb":
                node["properties"].update(state="exited", health=None)
        return nodes

    report = compare(parse_topology(design()), observation(nodes=stopped), topology_id="t" * 32, version=1)
    row = rows(report)["container:itsm-surrealdb"]
    assert row["status"] == "drifted"
    assert row["differences"] == [
        {"property": "health", "expected": "healthy", "observed": None, "observed_present": True},
        {"property": "state", "expected": "running", "observed": "exited", "observed_present": True},
    ]
    # depends_on propagates from the dependency to everything that depends on it.
    assert report["potentially_affected"] == ["endpoint:ai", "endpoint:phoenix", "service:ticket-desk"]


def test_json_types_are_not_coerced():
    document = design()
    document["nodes"][1]["properties"] = {"up": 1}
    report = compare(parse_topology(document), observation(), topology_id="t" * 32, version=1)
    assert rows(report)["interface:lan"]["status"] == "drifted"


def test_an_undeclared_observed_property_is_reported_as_not_present():
    document = design()
    document["nodes"][0]["properties"] = {"model": "lab"}
    row = rows(compare(parse_topology(document), observation(), topology_id="t" * 32, version=1))["router:openwrt"]
    assert row["differences"] == [{"property": "model", "expected": "lab", "observed": None, "observed_present": False}]


def test_missing_requires_a_source_that_confirmed_absence():
    absent = observation(nodes=lambda nodes: [n for n in nodes if n["id"] != "container:itsm-surrealdb"],
                         absent=["container:itsm-surrealdb"])
    report = compare(parse_topology(design()), absent, topology_id="t" * 32, version=1)
    assert rows(report)["container:itsm-surrealdb"]["status"] == "missing"
    assert report["potentially_affected"] == ["endpoint:ai", "endpoint:phoenix", "service:ticket-desk"]


@pytest.mark.parametrize("source,node,reason", [
    ("containers", "container:itsm-surrealdb", "source_unavailable"),
    ("endpoints", "endpoint:phoenix", "source_unavailable"),
    ("router", "interface:lan", "source_unavailable"),
])
def test_an_unreadable_source_makes_its_nodes_unknown_not_missing(source, node, reason):
    def broken(sources):
        return [{**row, "state": "unavailable", "error_code": "provider_unavailable"} if row["id"] == source else row
                for row in sources]

    def without(nodes):
        prefix = node.split(":")[0] + ":"
        return [n for n in nodes if not n["id"].startswith(prefix)]

    report = compare(parse_topology(design()), observation(sources=broken, nodes=without),
                     topology_id="t" * 32, version=1)
    assert rows(report)[node]["status"] == "unknown"
    assert rows(report)[node]["reason"] == reason
    # Unknown state never seeds impact.
    assert report["potentially_affected"] == []


def test_unconfigured_sources_and_out_of_scope_nodes_are_unknown_with_their_reason():
    def unconfigured(sources):
        return [{**row, "state": "not_configured", "error_code": "not_configured"} if row["id"] == "router" else row
                for row in sources]

    document = design()
    document["nodes"].append({"id": "container:not-allow-listed", "kind": "container", "label": "", "properties": {}})
    document["nodes"].append({"id": "interface:guest", "kind": "interface", "label": "", "properties": {}})
    report = compare(parse_topology(document),
                     observation(sources=unconfigured,
                                 nodes=lambda nodes: [n for n in nodes if not n["id"].startswith(("router", "interface"))]),
                     topology_id="t" * 32, version=1)
    found = rows(report)
    assert found["router:openwrt"]["reason"] == "source_not_configured"
    assert found["container:not-allow-listed"]["reason"] == "not_in_source_scope"
    assert found["service:ticket-desk"]["reason"] == "no_source"
    assert found["interface:guest"]["reason"] == "source_not_configured"


def test_a_named_interface_the_router_would_not_read_is_unknown():
    document = design()
    document["nodes"].append({"id": "interface:guest", "kind": "interface", "label": "", "properties": {}})
    report = compare(parse_topology(document), observation(unread=["interface:guest"]), topology_id="t" * 32, version=1)
    assert rows(report)["interface:guest"] == {"id": "interface:guest", "status": "unknown", "reason": "not_read",
                                               "differences": []}


@pytest.mark.parametrize("relation,affected", [
    ("connected_to", ["interface:lan", "router:openwrt"]),
    ("associated_with", []),
])
def test_relation_semantics(relation, affected):
    document = design()
    document["edges"] = [{"from": "router:openwrt", "to": "container:itsm-surrealdb", "relation": relation},
                         {"from": "interface:lan", "to": "router:openwrt", "relation": "hosted_on"}]
    stopped = observation(nodes=lambda nodes: [
        {**n, "properties": {**n["properties"], "state": "exited"}} if n["id"] == "container:itsm-surrealdb" else n
        for n in nodes])
    assert compare(parse_topology(document), stopped, topology_id="t" * 32, version=1)["potentially_affected"] == affected


def test_dependency_cycles_terminate():
    document = design()
    document["edges"].append({"from": "container:itsm-surrealdb", "to": "service:ticket-desk", "relation": "depends_on"})
    stopped = observation(nodes=lambda nodes: [
        {**n, "properties": {**n["properties"], "up": False}} if n["id"] == "endpoint:ai" else n for n in nodes])
    report = compare(parse_topology(document), stopped, topology_id="t" * 32, version=1)
    assert report["potentially_affected"] == ["container:itsm-surrealdb", "endpoint:phoenix", "service:ticket-desk"]


@pytest.mark.parametrize("mutate", [
    lambda d: d["nodes"].append(deepcopy(d["nodes"][0])),
    lambda d: d["edges"].append(deepcopy(d["edges"][0])),
    lambda d: d["edges"].append({"from": "router:openwrt", "to": "router:openwrt", "relation": "depends_on"}),
    lambda d: d["edges"].append({"from": "router:openwrt", "to": "unknown:node", "relation": "depends_on"}),
    lambda d: d["edges"][0].update(relation="executes"),
    lambda d: d["nodes"][0].update(id="../outside"),
    lambda d: d["nodes"][0].update(kind="Router Kind"),
    lambda d: d["nodes"][0].update(label="x" * 121),
    lambda d: d["nodes"][0].pop("label"),
    lambda d: d["nodes"][0].update(command="reboot"),
    lambda d: d.update(apply=True),
    lambda d: d["nodes"][0]["properties"].update({"Bad Key": "x"}),
    lambda d: d["nodes"][0]["properties"].update(nested={"a": 1}),
    lambda d: d["nodes"][0]["properties"].update(values=[1, 2]),
    lambda d: d["nodes"][0]["properties"].update(large=2 ** 53 + 1),
    lambda d: d["nodes"][0]["properties"].update(ratio=float("inf")),
    lambda d: d["nodes"][0]["properties"].update(text="x" * 201),
    lambda d: d["nodes"][0]["properties"].update(text="line\nbreak"),
    lambda d: d.update(nodes=[{"id": f"host:n{i}", "kind": "host", "label": "", "properties": {}} for i in range(301)]),
    lambda d: d["nodes"][0].update(properties={f"p{i}": i for i in range(41)}),
])
def test_invalid_topologies_are_refused(mutate):
    document = design()
    mutate(document)
    with pytest.raises(TopologyError) as caught:
        parse_topology(document)
    assert caught.value.code == "invalid_topology"


@pytest.mark.parametrize("key,value", [
    ("admin_password", "x"), ("api_key", "x"), ("session_token", "x"),
    ("note", "Bearer abcdefghijklmnop.qrstu"), ("note", "password=hunter2"), ("note", "api-key: abc"),
    ("note", "-----BEGIN PRIVATE KEY"), ("note", "https://admin:pw@router.lab"), ("owner", "ops@example.org"),
])
def test_private_values_are_refused_without_echoing_them(key, value):
    document = design()
    document["nodes"][0]["properties"][key] = value
    with pytest.raises(TopologyError) as caught:
        parse_topology(document)
    assert caught.value.code == "private_topology_content"
    assert value not in str(caught.value)


@pytest.mark.parametrize("label", ["Basic switch", "Bearer service", "Token ring segment", "Secret garden VLAN"])
def test_ordinary_words_are_not_mistaken_for_credentials(label):
    document = design()
    document["nodes"][0]["label"] = label
    assert parse_topology(document)["nodes"][0]["label"] == label


def test_empty_design_is_a_valid_topology():
    report = compare(parse_topology({"nodes": [], "edges": []}), observation(), topology_id="t" * 32, version=1)
    assert report["nodes"] == [] and len(report["not_in_design"]) == 6


def test_report_does_not_include_actions_or_commands():
    report = compare(parse_topology(design()), observation(), topology_id="t" * 32, version=1)
    assert not ({"commands", "actions", "tool_calls", "approvals", "apply"} & report.keys())
