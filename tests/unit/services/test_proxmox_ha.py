"""Proxmox high availability (HA) state, read-only.

One read of /cluster/ha/status/current (Sys.Audit on /). Proxmox reports the
HA manager's and each node's state only inside a status text such as
"pve-a (active, watchdog active, Mon Nov 16 11:20:01 2026)", worked out on
the node with its own clock (an entry older than 30 s reads "old timestamp -
dead?"). The service maps the state word in that text to a fixed word and
never passes the text on: it carries the node's local time. A word it does
not know is "unknown", never healthy. Resource comments are operator free
text and stay in the service.
"""
import json

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"
PATH = "/api2/json/cluster/ha/status/current"

QUORUM = {"id": "quorum", "type": "quorum", "node": "pve-a", "status": "OK", "quorate": 1}
CLUSTER = [
    QUORUM,
    {"id": "master", "type": "master", "node": "pve-a", "status": "pve-a (active, Mon Nov 16 11:20:00 2026)",
     "timestamp": 1_794_802_800},
    {"id": "fencing", "type": "fencing", "node": "pve-a", "status": "armed (CRM watchdog active)",
     "armed-state": "armed"},
    {"id": "lrm:pve-a", "type": "lrm", "node": "pve-a",
     "status": "pve-a (active, watchdog active, Mon Nov 16 11:20:01 2026)", "timestamp": 1_794_802_801},
    {"id": "lrm:pve-b", "type": "lrm", "node": "pve-b",
     "status": "pve-b (old timestamp - dead?, watchdog standby, Mon Nov 16 10:02:11 2026)",
     "timestamp": 1_794_798_131},
    {"id": "lrm:pve-c", "type": "lrm", "node": "pve-c",
     "status": "pve-c (maintenance mode, watchdog standby, Mon Nov 16 11:19:58 2026)", "timestamp": 1_794_802_798},
    {"id": "service:vm:100", "type": "service", "sid": "vm:100", "node": "pve-a", "crm_state": "started",
     "state": "started", "request_state": "started", "status": "vm:100 (pve-a, started)",
     "comment": "orders primary; page Bob first", "max_restart": 1, "max_relocate": 1, "failback": 1},
    {"id": "service:ct:101", "type": "service", "sid": "ct:101", "node": "pve-b", "crm_state": "error",
     "state": "error", "request_state": "started", "status": "ct:101 (pve-b, error)"},
    {"id": "service:vm:102", "type": "service", "sid": "vm:102", "node": "pve-c", "crm_state": "stopped",
     "state": "stopped", "request_state": "stopped", "status": "vm:102 (pve-c, stopped)"},
]


def service(routes, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        status, data = routes[request.url.path]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client), client


def read(rows):
    client_service, client = service({PATH: (200, rows)})
    try:
        return client_service.ha_status()["data"]
    finally:
        client.close()


def test_ha_status_reads_quorum_manager_nodes_and_resources_in_fixed_words():
    calls = []
    client_service, client = service({PATH: (200, CLUSTER)}, calls)
    try:
        result = client_service.ha_status()
    finally:
        client.close()
    assert calls == [("GET", PATH, {})]
    data = result["data"]
    assert data["schema"] == "proxmox-ha-status.v1"
    assert data["quorate"] is True and data["in_use"] is True and data["fencing"] == "armed"
    assert data["manager"] == {"node": "pve-a", "state": "active", "problem": False}
    assert data["nodes"] == [{"node": "pve-a", "state": "active", "problem": False},
                             {"node": "pve-b", "state": "not_responding", "problem": True},
                             {"node": "pve-c", "state": "maintenance", "problem": False}]
    # A resource in trouble first, then by VMID.
    assert data["resources"] == [
        {"sid": "ct:101", "type": "lxc", "vmid": 101, "node": "pve-b", "state": "error", "requested": "started",
         "problem": True},
        {"sid": "vm:100", "type": "qemu", "vmid": 100, "node": "pve-a", "state": "started", "requested": "started",
         "problem": False},
        {"sid": "vm:102", "type": "qemu", "vmid": 102, "node": "pve-c", "state": "stopped", "requested": "stopped",
         "problem": False}]
    assert data["count"] == 3 and data["truncated"] is False
    assert data["summary"] == {"nodes": 3, "nodes_with_problems": 1, "resources": 3, "resources_with_problems": 1}
    text = json.dumps(result)
    for leaked in ("Mon Nov", "watchdog", "page Bob", "comment", "timestamp"):
        assert leaked not in text, leaked


def test_ha_that_is_not_in_use_reads_plainly():
    # The live hummingbird answer: no HA resource was ever configured.
    data = read([QUORUM, {"id": "fencing", "type": "fencing", "node": "pve-a",
                          "status": "standby (CRM watchdog standby)", "armed-state": "standby"}])
    assert data["in_use"] is False and data["quorate"] is True and data["fencing"] == "standby"
    assert data["manager"] is None and data["nodes"] == [] and data["resources"] == []


def test_no_quorum_is_reported():
    data = read([{"id": "quorum", "type": "quorum", "node": "pve-a", "status": "No quorum on node 'pve-a'!",
                  "quorate": 0}])
    assert data["quorate"] is False


# Every word pve-ha-manager's status handler can put in an LRM status text.
# Deliberate operator modes (maintenance, disarm) are not problems; a node in
# transition, without its lock, unreadable or unknown is.
@pytest.mark.parametrize("status,state,problem", [
    ("pve-d (unable to read lrm status)", "unreadable", True),
    ("pve-d (idle, watchdog standby, Mon Nov 16 11:20:01 2026)", "idle", False),
    ("pve-d (wait_for_agent_lock, watchdog standby, Mon Nov 16 11:20:01 2026)", "waiting_for_lock", True),
    ("pve-d (lost_agent_lock, watchdog standby, Mon Nov 16 11:20:01 2026)", "lost_lock", True),
    ("pve-d (maintenance, watchdog active, Mon Nov 16 11:20:01 2026)", "maintenance", False),
    ("pve-d (disarm mode, watchdog released, Mon Nov 16 11:20:01 2026)", "disarmed", False),
    ("pve-d (restart mode, watchdog standby, Mon Nov 16 11:20:01 2026)", "restarting", True),
    ("pve-d (shutdown mode, watchdog standby, Mon Nov 16 11:20:01 2026)", "shutting_down", True),
    ("pve-d (detected time drift!, watchdog standby, Mon Nov 16 11:20:01 2026)", "clock_drift", True),
    ("pve-d (unknown, watchdog standby, Mon Nov 16 11:20:01 2026)", "unknown", True),
    ("pve-d (a state from a later release, watchdog active, Mon Nov 16 11:20:01 2026)", "unknown", True),
    ("something else entirely", "unknown", True),
])
def test_each_node_state_word_is_fixed_and_an_unknown_one_is_never_healthy(status, state, problem):
    node = read([QUORUM, {"id": "lrm:pve-d", "type": "lrm", "node": "pve-d", "status": status}])["nodes"][0]
    assert node == {"node": "pve-d", "state": state, "problem": problem}


@pytest.mark.parametrize("state,problem", [("started", False), ("stopped", False), ("migrate", False),
                                           ("error", True), ("fence", True), ("recovery", True),
                                           ("not found", True)])
def test_a_resource_in_error_fencing_recovery_or_missing_is_a_problem(state, problem):
    resource = read([QUORUM, {"id": "service:vm:100", "type": "service", "sid": "vm:100", "node": "pve-a",
                              "state": state, "request_state": "started"}])["resources"][0]
    assert resource["state"] == state and resource["problem"] is problem


@pytest.mark.parametrize("status,state,problem", [
    ("pve-a (idle, Mon Nov 16 11:20:00 2026)", "idle", False),
    ("pve-a (old timestamp - dead?, Mon Nov 16 10:20:00 2026)", "not_responding", True),
    ("pve-a (active, Mon Nov 16 11:20:00 2026) - static load CRS", "active", False),
])
def test_the_ha_manager_state_word_is_fixed(status, state, problem):
    manager = read([QUORUM, {"id": "master", "type": "master", "node": "pve-a", "status": status}])["manager"]
    assert manager == {"node": "pve-a", "state": state, "problem": problem}


def test_an_entry_of_a_type_this_version_does_not_know_is_left_out():
    data = read([QUORUM, {"id": "planner", "type": "planner", "node": "pve-a", "status": "anything"}])
    assert data["quorate"] is True and data["nodes"] == [] and data["resources"] == []


@pytest.mark.parametrize("row", [
    {"id": "service:vm:abc", "type": "service", "sid": "vm:abc", "node": "pve-a", "state": "started"},
    {"id": "service:zz:100", "type": "service", "sid": "zz:100", "node": "pve-a", "state": "started"},
    {"id": "service:vm:0", "type": "service", "sid": "vm:0", "node": "pve-a", "state": "started"},
    {"id": "service:vm:100", "type": "service", "sid": "vm:100", "node": "pve-a", "state": "started\nnow"},
    {"id": "service:vm:100", "type": "service", "sid": "vm:100", "node": "pve-a", "state": "Started; rm"},
    {"id": "lrm:../x", "type": "lrm", "node": "../x", "status": "../x (active, w, t)"},
    {"id": "quorum", "type": "quorum", "node": "pve-a", "status": "OK", "quorate": 2},
    {"id": "fencing", "type": "fencing", "node": "pve-a", "status": "armed", "armed-state": "armed; rm -rf"},
    "not an object",
])
def test_a_malformed_ha_entry_is_refused(row):
    with pytest.raises(RuntimeError, match="invalid answer"):
        read([QUORUM, row])
