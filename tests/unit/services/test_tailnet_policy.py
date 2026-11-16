"""Tailnet policy and device posture reads (direction note plan item A/TS), read-only.

The deployment's own trust credential holds policy_file and
devices:posture_attributes:read since 16 November. The policy read summarizes
who may reach what: tags, groups, autogroups and "*" stay as written, while
users (emails), host aliases and addresses become fixed words, so no identity
or address leaves the service. Posture is read per device (bounded), and only
the fixed node attributes leave the service; custom and integration attributes
are counted, never repeated.
"""
import json
from datetime import datetime, timezone

import httpx
import pytest

from services.tailnet import MAX_POSTURE_DEVICES, TailnetConfig, TailnetService

NOW = datetime(2026, 11, 16, 7, 0, tzinfo=timezone.utc)
# The live tailnet's policy shape (16 November), with a fixture user.
POLICY = {
    "grants": [{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]}],
    "ssh": [{"action": "check", "src": ["autogroup:member"], "dst": ["autogroup:self"],
             "users": ["autogroup:nonroot", "root", "deploy"]}],
    "tagOwners": {"tag:janus-lab": ["autogroup:admin"]},
    "tests": [{"src": "owner@example.com", "accept": ["tag:janus-lab:22"]},
              {"src": "tag:janus-lab", "deny": ["100.64.0.1:22"]}],
}
DEVICES = [{"nodeId": "nAAA1", "hostname": "hummingbird"}, {"nodeId": "nBBB2", "hostname": "Blackhawk"}]
POSTURE = {
    "nAAA1": {"attributes": {"node:os": "linux", "node:osVersion": "7.0.14-28-pve", "node:tsAutoUpdate": True,
                             "node:tsReleaseTrack": "stable", "node:tsStateEncrypted": False,
                             "node:tsVersion": "1.104.7"}},
    "nBBB2": {"attributes": {"node:os": "windows", "node:osVersion": "10.0.26300.9974", "node:tsAutoUpdate": False,
                             "node:tsReleaseTrack": "stable", "node:tsStateEncrypted": True,
                             "node:tsVersion": "1.104.4", "custom:owner": "alice@example.com",
                             "intune:complianceState": "compliant"}},
}


def service(routes, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/api/v2/oauth/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600, "token_type": "Bearer"})
        status, body = routes[request.url.path]
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TailnetService(TailnetConfig("client-id", "client-secret"), client, clock=lambda: NOW)


def policy_routes(policy=POLICY, settings=None):
    return {"/api/v2/tailnet/-/acl": (200, policy),
            "/api/v2/tailnet/-/settings": (200, settings if settings is not None else {"aclsExternallyManagedOn": False})}


def test_the_policy_reads_who_may_reach_what_with_users_and_addresses_masked():
    calls = []
    result = service(policy_routes(), calls).policy()
    assert [path for _, path, _ in calls if path != "/api/v2/oauth/token"] == ["/api/v2/tailnet/-/acl",
                                                                              "/api/v2/tailnet/-/settings"]
    assert all(method == "GET" for method, path, _ in calls if not path.endswith("/oauth/token"))
    data = result["data"]
    assert data["schema"] == "tailnet-policy.v1" and result["provider"] == "tailscale"
    assert data["externally_managed"] is False
    assert data["rules"] == [{"kind": "grant", "sources": ["autogroup:member"], "destinations": ["*"],
                              "ports": ["*"], "app_capabilities": 0}]
    # root is a well-known account and a security fact; other local account names are masked.
    assert data["ssh"] == [{"action": "check", "sources": ["autogroup:member"], "destinations": ["autogroup:self"],
                            "users": ["autogroup:nonroot", "root", "a named user"]}]
    assert data["tag_owners"] == [{"tag": "tag:janus-lab", "owners": ["autogroup:admin"]}]
    assert data["groups"] == [] and data["hosts"] == 0 and data["tests"] == 2
    assert data["other_sections"] == []
    assert data["summary"] == {"rules": 1, "ssh_rules": 1, "tags": 1, "groups": 0, "open_to_all_members": True}
    text = json.dumps(result)
    for leaked in ("example.com", "100.64.0.1", "owner@"):
        assert leaked not in text, leaked


def test_acl_rules_split_destination_ports_and_mask_users_hosts_and_addresses():
    policy = {
        "groups": {"group:ops": ["alice@example.com", "bob@example.com"]},
        "hosts": {"nas": "192.168.100.20"},
        "acls": [{"action": "accept", "src": ["group:ops", "carol@example.com"],
                  "dst": ["tag:db:5432", "nas:445", "10.0.0.0/8:*", "*:22"]}],
        "nodeAttrs": [{"target": ["*"], "attr": ["funnel"]}],
        "autoApprovers": {"routes": {"10.0.0.0/8": ["tag:router"]}},
    }
    data = service(policy_routes(policy, {"aclsExternallyManagedOn": True})).policy()["data"]
    assert data["externally_managed"] is True
    assert data["rules"] == [{"kind": "acl", "sources": ["group:ops", "a user"],
                              "destinations": ["tag:db", "a host alias", "an address", "*"],
                              "ports": ["5432", "445", "*", "22"], "app_capabilities": 0}]
    assert data["groups"] == [{"name": "group:ops", "members": 2}]
    assert data["hosts"] == 1
    assert data["other_sections"] == ["autoApprovers", "nodeAttrs"]
    assert data["summary"]["open_to_all_members"] is False
    text = json.dumps(data)
    assert "alice" not in text and "192.168" not in text and "10.0.0.0" not in text and '"nas"' not in text


@pytest.mark.parametrize("policy", [
    [],
    {"grants": "all"},
    {"grants": [{"src": "autogroup:member", "dst": ["*"]}]},
    {"acls": [{"action": "accept", "src": [5], "dst": ["*:*"]}]},
    {"tagOwners": {"tag:x": "autogroup:admin"}},
    {"groups": {"group:x": [1]}},
])
def test_a_malformed_policy_is_refused(policy):
    with pytest.raises(RuntimeError, match="invalid answer"):
        service(policy_routes(policy)).policy()


def test_a_refused_settings_read_leaves_external_management_unknown():
    routes = policy_routes()
    routes["/api/v2/tailnet/-/settings"] = (403, {"message": "forbidden"})
    assert service(routes).policy()["data"]["externally_managed"] is None


def test_device_posture_reads_fixed_node_attributes_and_counts_the_rest():
    calls = []
    routes = {"/api/v2/tailnet/-/devices": (200, {"devices": DEVICES}),
              **{f"/api/v2/device/{node}/attributes": (200, body) for node, body in POSTURE.items()}}
    result = service(routes, calls).device_posture()
    data = result["data"]
    assert data["schema"] == "tailnet-posture.v1"
    assert (data["count"], data["truncated"]) == (2, False)
    assert data["devices"] == [
        {"name": "Blackhawk", "os": "windows", "os_version": "10.0.26300.9974", "tailscale_version": "1.104.4",
         "auto_update": False, "release_track": "stable", "state_encrypted": True, "other_attributes": 2},
        {"name": "hummingbird", "os": "linux", "os_version": "7.0.14-28-pve", "tailscale_version": "1.104.7",
         "auto_update": True, "release_track": "stable", "state_encrypted": False, "other_attributes": 0},
    ]
    assert data["summary"] == {"auto_update_off": 1, "state_not_encrypted": 1}
    text = json.dumps(result)
    assert "alice" not in text and "compliant" not in text and "nAAA1" not in text


def test_posture_is_read_for_a_bounded_number_of_devices():
    many = [{"nodeId": f"n{index:04d}", "hostname": f"host-{index:04d}"} for index in range(MAX_POSTURE_DEVICES + 5)]
    calls = []
    routes = {"/api/v2/tailnet/-/devices": (200, {"devices": many}),
              **{f"/api/v2/device/{row['nodeId']}/attributes": (200, {"attributes": {}}) for row in many}}
    data = service(routes, calls).device_posture()["data"]
    assert (data["count"], data["truncated"], len(data["devices"])) == (MAX_POSTURE_DEVICES + 5, True,
                                                                         MAX_POSTURE_DEVICES)
    assert sum(path.endswith("/attributes") for _, path, _ in calls) == MAX_POSTURE_DEVICES


def test_a_device_whose_posture_cannot_be_read_is_unknown_not_healthy():
    routes = {"/api/v2/tailnet/-/devices": (200, {"devices": DEVICES[:1]}),
              "/api/v2/device/nAAA1/attributes": (404, {"message": "not found"})}
    row = service(routes).device_posture()["data"]["devices"][0]
    assert row == {"name": "hummingbird", "os": None, "os_version": None, "tailscale_version": None,
                   "auto_update": None, "release_track": None, "state_encrypted": None, "other_attributes": None}


@pytest.mark.parametrize("devices,attributes", [
    ([{"nodeId": "../x", "hostname": "a"}], {}),
    ([{"nodeId": "nAAA1", "hostname": "a"}], {"attributes": {"node:tsAutoUpdate": "yes"}}),
    ([{"nodeId": "nAAA1", "hostname": "a"}], {"attributes": []}),
])
def test_malformed_posture_answers_are_refused(devices, attributes):
    routes = {"/api/v2/tailnet/-/devices": (200, {"devices": devices}),
              "/api/v2/device/nAAA1/attributes": (200, attributes)}
    with pytest.raises(RuntimeError, match="invalid answer"):
        service(routes).device_posture()
