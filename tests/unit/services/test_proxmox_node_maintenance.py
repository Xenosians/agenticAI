"""One Proxmox node's maintenance state, read-only (direction note H1).

Seven reads, each allowed by Sys.Audit: /nodes/{node}/status (version,
running kernel, boot mode), /nodes/{node}/services, /nodes/{node}/apt/versions
(installed kernels), /nodes/{node}/certificates/info, /nodes/{node}/subscription,
/nodes/{node}/apt/repositories and /cluster/status (whether the node is in a
cluster, which decides whether a stopped corosync is expected: corosync starts
only when /etc/corosync/corosync.conf exists). Certificate subjects, issuers,
SANs, fingerprints and PEM, the subscription key and server id, and repository
file paths never leave the service. A section Proxmox refuses is unreadable,
never healthy. Pending package updates need Sys.Modify (GET apt/update), which
Janus does not hold, so they are not read.
"""
import json
import time

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

NOW = 1_794_800_000
DAY = 86_400
CA = "/CN=Proxmox Virtual Environment/OU=0b3c5d7e-1111-2222-3333-444455556666/O=PVE Cluster Manager CA"
STATUS = {"pveversion": "pve-manager/9.2.27/9583db5e450078e7",
          "current-kernel": {"sysname": "Linux", "release": "7.0.14-28-pve", "machine": "x86_64",
                             "version": "#2 SMP PREEMPT_DYNAMIC PMX 7.0.14-28 (2026-11-01T10:00Z)"},
          "kversion": "Linux 7.0.14-28-pve #2 SMP PREEMPT_DYNAMIC PMX 7.0.14-28 (2026-11-01T10:00Z)",
          "boot-info": {"mode": "efi", "secureboot": 1}, "uptime": DAY, "cpu": 0.02}


def unit(name, active, unit_state, state=None):
    return {"name": name, "service": name, "desc": f"{name} daemon", "active-state": active,
            "state": state or ("running" if active == "active" else "dead"), "unit-state": unit_state}


SERVICES = [unit("pveproxy", "active", "enabled"), unit("pvestatd", "failed", "enabled", "failed"),
            unit("corosync", "inactive", "enabled"), unit("spiceproxy", "inactive", "disabled"),
            unit("syslog", "inactive", "not-found"), unit("systemd-journald", "active", "static")]


def package(name, version, state="Installed"):
    return {"Package": name, "Version": version, "CurrentState": state, "Title": name, "Origin": "Proxmox"}


VERSIONS = [package("proxmox-ve", "9.2.0"), package("pve-manager", "9.2.27"),
            package("proxmox-kernel-7.0", "7.0.14-30"),  # meta package: not a kernel
            package("proxmox-kernel-7.0.14-29-pve-signed", "7.0.14-29"),
            package("proxmox-kernel-7.0.14-28-pve-signed", "7.0.14-28"),
            package("proxmox-kernel-7.0.15-1-pve-signed", "7.0.15-1", "ConfigFiles"),  # removed
            package("proxmox-kernel-helper", "9.2.1")]


def certificate(filename, days, issuer=CA, subject=CA, key=("rsaEncryption", 2048)):
    return {"filename": filename, "notafter": NOW + days, "notbefore": NOW - 30 * DAY, "issuer": issuer,
            "subject": subject, "san": ["127.0.0.1", "pve-a.lan", "192.168.1.5"], "fingerprint": "AA:BB:CC",
            "pem": "-----BEGIN CERTIFICATE-----", "public-key-type": key[0], "public-key-bits": key[1]}


CERTIFICATES = [certificate("pve-root-ca.pem", 3645 * DAY, key=("rsaEncryption", 4096)),
                certificate("pve-ssl.pem", 20 * DAY + 3600,
                            subject="/OU=PVE Cluster Node/O=Proxmox Virtual Environment/CN=pve-a.lan")]
SUBSCRIPTION = {"status": "notfound", "message": "There is no subscription key", "serverid": "0123456789ABCDEF",
                "url": "https://www.proxmox.com/en/proxmox-virtual-environment/pricing"}


def repositories(**statuses):
    rows = {"enterprise": 1, "no-subscription": 0, "test": None, "ceph-squid-no-subscription": 1, **statuses}
    return {"errors": [], "infos": [], "digest": "ab",
            "files": [{"path": "/etc/apt/sources.list.d/pve-enterprise.sources", "repositories": []}],
            "standard-repos": [{"handle": handle, "name": handle.title(), "status": status, "description": "x"}
                               for handle, status in rows.items()]}


STANDALONE = [{"type": "node", "id": "node/pve-a", "name": "pve-a", "online": 1, "local": 1, "nodeid": 0,
               "ip": "192.168.1.5", "level": ""}]
CLUSTERED = [{"type": "cluster", "id": "cluster", "name": "lab", "nodes": 3, "quorate": 1, "version": 5}, *STANDALONE]


def routes(**overrides):
    base = {"status": (200, STATUS), "services": (200, SERVICES), "apt/versions": (200, VERSIONS),
            "certificates/info": (200, CERTIFICATES), "subscription": (200, SUBSCRIPTION),
            "apt/repositories": (200, repositories()), "cluster": (200, STANDALONE)}
    base.update(overrides)
    return {("/cluster/status" if name == "cluster" else f"/nodes/pve-a/{name}"): value for name, value in base.items()}


def read(route_map, calls=None, node="pve-a"):
    def handler(request):
        path = request.url.path.removeprefix("/api2/json")
        if calls is not None:
            calls.append((request.method, path))
        status, data = route_map[path]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", "reporter@pam!readonly", "secret"), client)
    try:
        return service.node_maintenance(node)["data"]
    finally:
        client.close()


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)


def test_one_node_reads_version_kernel_services_certificates_and_updates_without_identifiers():
    calls = []
    data = read(routes(), calls)
    assert sorted(calls) == sorted(("GET", path) for path in routes())
    assert data == {
        "schema": "proxmox-node-maintenance.v1", "node": "pve-a", "version": "9.2.27",
        "kernel": {"running": "7.0.14-28-pve", "newest_installed": "7.0.14-29-pve", "installed": 2},
        "boot": {"mode": "efi", "secure_boot": True},
        "in_cluster": False,
        "services": {"installed": 5, "running": 2, "not_running": [{"name": "pvestatd", "state": "failed"}],
                     "expected_stopped": ["corosync"], "turned_off": ["spiceproxy"], "not_installed": 1},
        "certificates": [
            {"file": "pve-ssl.pem", "role": "node", "served": True, "issuer": "proxmox_ca", "key": "RSA 2048",
             "days_left": 20},
            {"file": "pve-root-ca.pem", "role": "cluster_ca", "served": False, "issuer": "self_signed",
             "key": "RSA 4096", "days_left": 3645}],
        "subscription": "none",
        "repositories": {"enabled": ["enterprise", "ceph-squid-no-subscription"], "turned_off": ["no-subscription"],
                         "errors": 0},
        "issues": ["service_not_running", "certificate_expiring", "enterprise_repository_without_subscription",
                   "reboot_for_newer_kernel"],
        "unreadable": [],
    }
    text = json.dumps(data)
    for hidden in ("pve-a.lan", "192.168", "AA:BB", "BEGIN", "0123456789ABCDEF", "0b3c5d7e", ".sources", "PMX"):
        assert hidden not in text


def test_a_node_in_a_cluster_needs_corosync_running():
    data = read(routes(cluster=(200, CLUSTERED)))
    assert data["in_cluster"] is True
    assert data["services"]["not_running"] == [{"name": "corosync", "state": "inactive"},
                                               {"name": "pvestatd", "state": "failed"}]
    assert data["services"]["expected_stopped"] == []


def test_a_section_proxmox_refuses_is_unreadable_never_healthy():
    data = read(routes(services=(403, None), **{"certificates/info": (403, None)}, subscription=(403, None)))
    assert data["unreadable"] == ["services", "certificates", "subscription"]
    assert (data["services"], data["certificates"], data["subscription"]) == (None, None, None)
    # The enterprise repository is enabled, but without the subscription Janus cannot say updates fail.
    assert data["issues"] == ["reboot_for_newer_kernel"]
    unknown = read(routes(cluster=(403, None)))
    assert unknown["in_cluster"] is None and unknown["unreadable"] == ["cluster"]
    assert {"name": "corosync", "state": "inactive"} in unknown["services"]["not_running"]


def test_a_section_older_proxmox_does_not_have_is_unreadable_too():
    # apt/repositories exists since Proxmox VE 7; an older node answers 404 for it.
    data = read(routes(**{"apt/repositories": (404, None)}))
    assert data["unreadable"] == ["repositories"] and data["repositories"] is None
    assert "no_update_repository" not in data["issues"]


def test_the_newest_installed_kernel_running_needs_no_reboot_and_kversion_is_the_fallback():
    versions = [row for row in VERSIONS if row["Package"] != "proxmox-kernel-7.0.14-29-pve-signed"]
    status = {key: value for key, value in STATUS.items() if key != "current-kernel"}
    data = read(routes(status=(200, status), **{"apt/versions": (200, versions)}))
    assert data["kernel"] == {"running": "7.0.14-28-pve", "newest_installed": "7.0.14-28-pve", "installed": 1}
    assert "reboot_for_newer_kernel" not in data["issues"]


def test_repository_subscription_and_served_certificate_rules():
    active = read(routes(subscription=(200, dict(SUBSCRIPTION, status="active", level="c"))))
    assert active["subscription"] == "active" and "enterprise_repository_without_subscription" not in active["issues"]
    ceph_only = read(routes(**{"apt/repositories": (200, repositories(enterprise=0))}))
    assert "no_update_repository" in ceph_only["issues"]
    testing = read(routes(**{"apt/repositories": (200, repositories(enterprise=None, test=1))}))
    assert testing["issues"][-1] == "test_repository_enabled"
    custom = CERTIFICATES + [certificate("pveproxy-ssl.pem", -3600, issuer="/C=US/O=Let's Encrypt/CN=R11",
                                         subject="/CN=pve-a.example.com", key=("id-ecPublicKey", 256))]
    served = read(routes(**{"certificates/info": (200, custom)}))["certificates"]
    assert served[0] == {"file": "pveproxy-ssl.pem", "role": "custom", "served": True, "issuer": "other_ca",
                         "key": "EC 256", "days_left": -1}
    assert [row["served"] for row in served] == [True, False, False]
    assert read(routes(**{"certificates/info": (200, custom)}))["issues"][:3] == [
        "service_not_running", "certificate_expired", "certificate_expiring"]


@pytest.mark.parametrize("node", ["../pve-a", "pve a", "", None])
def test_the_node_name_must_be_exact(node):
    with pytest.raises(ValueError, match="node"):
        read(routes(), node=node)


@pytest.mark.parametrize("override", [
    {"services": (200, [{"name": "pveproxy", "active-state": "active"}])},
    {"certificates/info": (200, [dict(CERTIFICATES[0], notafter="soon")])},
    {"apt/repositories": (200, repositories(enterprise=2))},
    {"status": (200, dict(STATUS, pveversion="pve-manager"))},
])
def test_malformed_answers_are_refused(override):
    with pytest.raises(RuntimeError, match="invalid answer"):
        read(routes(**override))
