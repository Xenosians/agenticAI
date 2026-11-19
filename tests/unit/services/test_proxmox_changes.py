"""Governed Proxmox guest changes: exact target, one request, task, read-back.

Each change re-reads the frozen guest, refuses a different identity or a
concurrently edited configuration, sends exactly one request, follows the
Proxmox task and verifies the expected state. An unconfirmable answer is an
unknown outcome, never a retried change.
"""
import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

TOKEN_ID = "janus@pve!dashboard"
TOKEN_SECRET = "fixture-token-secret"
UPID = "UPID:pve-a:000F1C2B:0A1B2C3D:6738A1B2:vzsnapshot:990:janus@pve!dashboard:"


def ok(request, data):
    return httpx.Response(200, json={"data": data}, request=request)


class FakeCluster:
    """A small Proxmox: inventory, configs, snapshots, tasks, storage and new guests."""

    def __init__(self, *, digests=None, task_polls=1, exitstatus="OK"):
        self.guests = {
            990: {"type": "lxc", "node": "pve-a", "name": "janus-lab", "status": "stopped",
                  "config": {"cores": 1, "memory": 128, "onboot": 0, "tags": "lab",
                             "rootfs": "local-lvm:vm-990-disk-0,size=1G"}},
            101: {"type": "qemu", "node": "pve-a", "name": "db", "status": "running",
                  "config": {"cores": 4, "memory": "current=4096", "onboot": 1,
                             "scsi0": "local-lvm:vm-101-disk-0,size=8G"}},
        }
        self.digests = dict(digests or {990: "d990", 101: "d101"})
        self.snapshots = {990: [{"name": "current", "parent": None}], 101: [{"name": "current"}]}
        self.volumes = []
        self.task_polls = task_polls
        self.exitstatus = exitstatus
        self.calls = []
        self.created_forms = []

    def rows(self):
        return [{"id": f"{guest['type']}/{vmid}", "type": guest["type"], "vmid": vmid, "node": guest["node"],
                 "name": guest["name"], "status": guest["status"]} for vmid, guest in self.guests.items()]

    def __call__(self, request):
        path = request.url.path
        method = request.method
        body = {}
        if method in ("POST", "PUT") and request.content:
            body = dict(httpx.QueryParams(request.content.decode()))
        params = dict(request.url.params)
        self.calls.append((method, path, body or params))

        if path == "/api2/json/cluster/resources":
            return ok(request, self.rows())
        if path == "/api2/json/nodes":
            return ok(request, [{"node": "pve-a", "status": "online"}])
        if path == "/api2/json/cluster/nextid":
            return ok(request, "992")
        if "/tasks/" in path and path.endswith("/status"):
            if len([call for call in self.calls if call[1] == path]) < self.task_polls:
                return ok(request, {"status": "running"})
            return ok(request, {"status": "stopped", "exitstatus": self.exitstatus})
        if path.startswith("/api2/json/nodes/pve-a/storage/") and path.endswith("/content"):
            return ok(request, self.volumes)

        parts = path.strip("/").split("/")
        # /api2/json/nodes/pve-a/{type}/{vmid}[/...]
        if len(parts) >= 6 and parts[:3] == ["api2", "json", "nodes"] and parts[3] == "pve-a":
            guest_type = parts[4]
            vmid = int(parts[5]) if parts[5].isdigit() else None
            tail = parts[6:]
            if guest_type in ("lxc", "qemu") and vmid in self.guests:
                guest = self.guests[vmid]
                if tail == [] and method == "DELETE":
                    del self.guests[vmid]
                    return ok(request, UPID)
                if tail == ["status", "current"]:
                    live = {"status": guest["status"], "vmid": vmid}
                    if guest.get("lock"):
                        live["lock"] = guest["lock"]
                    return ok(request, live)
                if tail == ["config"] and method == "GET":
                    return ok(request, {**guest["config"], "digest": self.digests.get(vmid)})
                if tail == ["config"] and method == "PUT":
                    for key, value in body.items():
                        guest["config"][key] = int(value) if key in {"cores", "memory", "onboot"} else value
                    self.digests[vmid] = f"d{vmid}-{len(self.calls)}"
                    return ok(request, None)
                if tail == ["snapshot"] and method == "GET":
                    return ok(request, self.snapshots.get(vmid, [{"name": "current"}]))
                if tail == ["snapshot"] and method == "POST":
                    self.snapshots.setdefault(vmid, [{"name": "current"}]).append(
                        {"name": body.get("snapname"), "parent": "current",
                         "description": body.get("description")})
                    return ok(request, UPID)
                if len(tail) == 2 and tail[0] == "snapshot" and method == "DELETE":
                    self.snapshots[vmid] = [row for row in self.snapshots[vmid] if row["name"] != tail[1]]
                    return ok(request, UPID)
                if len(tail) == 3 and tail[0] == "snapshot" and tail[2] == "rollback":
                    self.snapshots[vmid] = [{"name": "current", "parent": tail[1]}]
                    return ok(request, UPID)
                if tail == ["resize"] and method == "PUT":
                    name = body["disk"]
                    value = guest["config"][name]
                    previous = int(value.split("size=")[1].split("G")[0])
                    grown = previous + int(body["size"].lstrip("+").rstrip("G"))
                    guest["config"][name] = value.split(",size=")[0] + f",size={grown}G"
                    self.digests[vmid] = f"d{vmid}-{len(self.calls)}"
                    return ok(request, None)
                if tail == ["clone"] and method == "POST":
                    newid = int(body["newid"])
                    self.guests[newid] = {"type": guest_type, "node": guest["node"],
                                          "name": body.get("name") or body.get("hostname"),
                                          "status": "stopped",
                                          "config": dict(guest["config"])}
                    self.digests[newid] = f"d{newid}"
                    return ok(request, UPID)
        if path == "/api2/json/nodes/pve-a/vzdump" and method == "POST":
            self.volumes.append({"volid": f"local:backup/vzdump-lxc-{body['vmid']}-2026_11_17.tar.zst",
                                 "vmid": int(body["vmid"]), "ctime": int(__import__("time").time()),
                                 "format": "tar.zst"})
            return ok(request, UPID)
        if path == "/api2/json/nodes/pve-a/lxc" and method == "POST":
            vmid = int(body["vmid"])
            if "restore" in body:
                self.guests[vmid] = {"type": "lxc", "node": "pve-a", "name": body.get("hostname"),
                                     "status": "stopped", "config": {"rootfs": "local-lvm:vm-x,size=1G"}}
            else:
                self.created_forms.append(dict(body))
                self.guests[vmid] = {"type": "lxc", "node": "pve-a", "name": body.get("hostname"),
                                     "status": "stopped", "config": {"corefs": body.get("rootfs")}}
            self.digests[vmid] = f"d{vmid}"
            return ok(request, UPID)
        return httpx.Response(404, json={"data": None, "message": "not found"}, request=request)

    def writes(self):
        return [(method, path) for method, path, _ in self.calls if method in ("POST", "PUT", "DELETE")]


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def service_for(fake):
    client = httpx.Client(transport=httpx.MockTransport(fake))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test", TOKEN_ID, TOKEN_SECRET), client), client


def run(fake, call):
    service, client = service_for(fake)
    clock = Clock()
    try:
        return call(service, clock)
    finally:
        client.close()


def test_set_resources_writes_once_and_reads_the_new_values_back():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_set_resources(
        "pve-a", 990, cores=2, memory_mib=256, expected_type="lxc", expected_name="janus-lab",
        expected_digest="d990", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["status"], receipt["verification_ok"]) == (True, "executed", True)
    # Proxmox accepted the write, so resending it is never "safe" (as for power actions).
    assert receipt["changed"] is True and receipt["retry_safe"] is False
    assert receipt["previous"] == {"cores": 1, "memory_mib": 128}
    assert receipt["observed"] == {"cores": 2, "memory_mib": 256}
    assert fake.writes() == [("PUT", "/api2/json/nodes/pve-a/lxc/990/config")]


def test_a_concurrently_changed_configuration_is_refused_before_any_write():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_set_resources(
        "pve-a", 990, cores=2, memory_mib=None, expected_type="lxc", expected_name="janus-lab",
        expected_digest="stale", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["status"], receipt["retry_safe"]) == (False, "conflict", True)
    assert "configuration changed" in receipt["error"]
    assert fake.writes() == []


def test_set_options_normalizes_tags_and_verifies_start_at_boot():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_set_options(
        "pve-a", 990, tags="db, prod, db", on_boot=True, expected_type="lxc", expected_name="janus-lab",
        expected_digest="d990", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert receipt["tags"] == "db,prod" and receipt["on_boot"] is True
    assert fake.guests[990]["config"]["tags"] == "db,prod"
    assert fake.guests[990]["config"]["onboot"] == 1


def test_a_containers_console_returns_to_a_login_prompt_and_is_read_back():
    fake = FakeCluster()
    fake.guests[990]["config"]["cmode"] = "shell"
    receipt = run(fake, lambda service, clock: service.guest_set_options(
        "pve-a", 990, tags=None, on_boot=None, console_mode="tty", expected_type="lxc", expected_name="janus-lab",
        expected_digest="d990", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert receipt["observed"]["console_mode"] == "tty"
    assert fake.guests[990]["config"]["cmode"] == "tty"
    with pytest.raises(ValueError):                      # a VM has no container console mode
        run(fake, lambda service, clock: service.guest_set_options(
            "pve-a", 101, tags=None, on_boot=None, console_mode="tty", expected_type="qemu", expected_name="db",
            expected_digest="d101"))


def test_grow_only_disk_resize_reads_the_new_size_back():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_resize_disk(
        "pve-a", 101, disk="scsi0", grow_gib=4, expected_type="qemu", expected_name="db",
        expected_digest="d101", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert receipt["previous_bytes"] == 8 * 1024 ** 3 and receipt["new_bytes"] == 12 * 1024 ** 3
    assert fake.writes() == [("PUT", "/api2/json/nodes/pve-a/qemu/101/resize")]


def test_a_disk_that_is_not_part_of_the_guest_is_refused_before_any_write():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_resize_disk(
        "pve-a", 990, disk="scsi9", grow_gib=4, expected_type="lxc", expected_name="janus-lab",
        expected_digest="d990", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["status"]) == (False, "conflict")
    assert fake.writes() == []


def test_snapshot_create_follows_its_task_and_confirms_the_snapshot_is_listed():
    fake = FakeCluster(task_polls=2)
    receipt = run(fake, lambda service, clock: service.guest_snapshot_create(
        "pve-a", 990, snapshot="before-upgrade", description="test", expected_type="lxc",
        expected_name="janus-lab", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert receipt["upid"] == UPID and receipt["task_exit_status"] == "OK"
    assert [row["name"] for row in fake.snapshots[990]] == ["current", "before-upgrade"]


def test_snapshot_delete_is_confirmed_gone():
    fake = FakeCluster()
    fake.snapshots[990].append({"name": "old"})
    receipt = run(fake, lambda service, clock: service.guest_snapshot_delete(
        "pve-a", 990, snapshot="old", expected_type="lxc", expected_name="janus-lab",
        sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert [row["name"] for row in fake.snapshots[990]] == ["current"]


def test_rollback_is_confirmed_by_the_current_state_following_the_snapshot():
    fake = FakeCluster()
    fake.snapshots[990].append({"name": "known-good"})
    receipt = run(fake, lambda service, clock: service.guest_snapshot_rollback(
        "pve-a", 990, snapshot="known-good", expected_type="lxc", expected_name="janus-lab",
        sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert receipt["current_parent"] == "known-good"


def test_delete_refuses_a_running_guest_and_deletes_a_stopped_one():
    fake = FakeCluster()
    refused = run(fake, lambda service, clock: service.guest_delete(
        "pve-a", 101, expected_type="qemu", expected_name="db", sleep=clock.sleep, clock=clock.monotonic))
    assert (refused["ok"], refused["status"]) == (False, "conflict")
    assert fake.writes() == []

    receipt = run(fake, lambda service, clock: service.guest_delete(
        "pve-a", 990, expected_type="lxc", expected_name="janus-lab", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"], receipt["deleted"]) == (True, True, True)
    assert 990 not in fake.guests


def test_create_builds_the_container_and_confirms_it():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_create_lxc(
        "pve-a", 993, hostname="lab-2", template="local:vztmpl/alpine-3.23-default_20260116_amd64.tar.xz",
        cores=1, memory_mib=128, disk_gib=2, storage="local-lvm", pool="janus", description=None,
        sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert fake.guests[993]["name"] == "lab-2" and fake.guests[993]["type"] == "lxc"
    # Without the optional settings the request is exactly the minimal container.
    assert fake.created_forms == [{"vmid": "993", "ostemplate": "local:vztmpl/alpine-3.23-default_20260116_amd64.tar.xz",
                                   "hostname": "lab-2", "cores": "1", "memory": "128", "rootfs": "local-lvm:2",
                                   "unprivileged": "1", "pool": "janus"}]


PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdV8iaJ0lYvGbM3hJmW0sQyJ7dU6m8rM2r6QJ0Z1aBc janus-trainer"


def test_create_sends_the_approved_container_settings_and_nothing_else():
    # The trainer container (owner decision, 18 Nov 2026): swap, a fixed address on the NAT VNet, its own
    # nameservers, a public SSH key, nesting for systemd, a root-shell console for first boot, start at boot.
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_create_lxc(
        "pve-a", 991, hostname="janus-trainer", template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
        cores=4, memory_mib=3584, disk_gib=48, storage="local-lvm", pool=None, description="Janus training worker",
        swap_mib=4096, net0="name=eth0,bridge=janusnat,ip=10.77.0.10/24,gw=10.77.0.1,firewall=1",
        nameserver="1.1.1.1 9.9.9.9", ssh_public_keys=PUBLIC_KEY, nesting=True, console_mode="shell",
        on_boot=True, tags="janus-trainer", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    form = fake.created_forms[0]
    assert {key: form[key] for key in ("swap", "net0", "nameserver", "ssh-public-keys", "features", "cmode",
                                       "onboot", "tags")} == {
        "swap": "4096", "net0": "name=eth0,bridge=janusnat,ip=10.77.0.10/24,gw=10.77.0.1,firewall=1",
        "nameserver": "1.1.1.1 9.9.9.9", "ssh-public-keys": PUBLIC_KEY, "features": "nesting=1", "cmode": "shell",
        "onboot": "1", "tags": "janus-trainer"}
    assert form["unprivileged"] == "1" and "password" not in form


@pytest.mark.parametrize("field, value", [
    ("ssh_public_keys", "-----BEGIN OPENSSH " + "PRIVATE KEY-----"),        # only a public key is ever sent
    ("ssh_public_keys", PUBLIC_KEY + "\nssh-rsa AAAAB3Nza second"),        # one key, one line
    ("net0", "name=eth0,bridge=vmbr0,ip=dhcp;rm -rf /"),
    ("net0", "name=eth0,bridge=janusnat,ip=10.77.0.10/24,hwaddr=AA:BB"),    # only the fields Janus sets
    ("nameserver", "1.1.1.1 evil.example"),
    ("console_mode", "console"),
    ("swap_mib", -1),
    ("tags", "a;b"),
])
def test_create_refuses_settings_outside_their_shape_before_any_request(field, value):
    fake = FakeCluster()
    arguments = dict(hostname="janus-trainer", template="local:vztmpl/debian-13-standard_13.6-1_amd64.tar.zst",
                     cores=4, memory_mib=3584, disk_gib=48, storage="local-lvm", pool=None, description=None)
    arguments[field] = value
    service, client = service_for(fake)
    try:
        with pytest.raises(ValueError):
            service.guest_create_lxc("pve-a", 991, **arguments)
    finally:
        client.close()
    assert fake.calls == []


def test_clone_creates_the_new_guest_and_confirms_its_name():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_clone(
        "pve-a", 990, newid=994, name="lab-clone", full=True, pool=None, expected_type="lxc",
        expected_name="janus-lab", sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert fake.guests[994]["name"] == "lab-clone"


def test_backup_starts_one_run_and_confirms_a_new_volume():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.backup_now(
        "pve-a", 990, storage="local", expected_type="lxc", expected_name="janus-lab",
        sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert receipt["new_backups"] == 1 and receipt["storage"] == "local"


def test_restore_creates_a_new_guest_and_never_overwrites_one():
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.backup_restore(
        "pve-a", volume="local:backup/vzdump-lxc-990-2026_11_17.tar.zst", newid=995, name="restored",
        storage="local-lvm", pool=None, sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["verification_ok"]) == (True, True)
    assert fake.guests[995]["name"] == "restored"

    conflict = run(fake, lambda service, clock: service.backup_restore(
        "pve-a", volume="local:backup/vzdump-lxc-990-2026_11_17.tar.zst", newid=995, name="again",
        storage=None, pool=None, sleep=clock.sleep, clock=clock.monotonic))
    assert (conflict["ok"], conflict["status"]) == (False, "conflict")


def test_next_vmid_is_proxmox_own_answer_not_a_local_range():
    fake = FakeCluster()
    service, client = service_for(fake)
    try:
        assert service.next_vmid() == 992
    finally:
        client.close()


def test_an_unanswered_change_keeps_the_approved_target_and_nothing_is_retried():
    fake = FakeCluster()

    def answer(request):
        if request.method == "POST":
            raise httpx.ReadTimeout("no answer", request=request)
        return fake(request)

    client = httpx.Client(transport=httpx.MockTransport(answer))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test", TOKEN_ID, TOKEN_SECRET), client)
    try:
        receipt = service.guest_snapshot_create(
            "pve-a", 990, snapshot="before-upgrade", description=None, expected_type="lxc",
            expected_name="janus-lab", sleep=Clock().sleep, clock=Clock().monotonic)
    finally:
        client.close()
    assert (receipt["ok"], receipt["status"], receipt["retry_safe"]) == (False, "outcome_unknown", False)
    assert "fixture-token-secret" not in repr(receipt)


@pytest.mark.parametrize("expected_type,expected_name", [("qemu", "janus-lab"), ("lxc", "other")])
def test_a_guest_that_no_longer_matches_the_approval_is_not_touched(expected_type, expected_name):
    fake = FakeCluster()
    receipt = run(fake, lambda service, clock: service.guest_snapshot_create(
        "pve-a", 990, snapshot="before-upgrade", description=None, expected_type=expected_type,
        expected_name=expected_name, sleep=clock.sleep, clock=clock.monotonic))
    assert (receipt["ok"], receipt["status"]) == (False, "conflict")
    assert fake.writes() == []


def _readback_fails(fake, path_test):
    """Proxmox accepts the change, then every read of its result goes unanswered."""
    def answer(request):
        if request.method == "GET" and any(call[0] in ("POST", "PUT", "DELETE") for call in fake.calls) \
                and path_test(request.url.path) and "/tasks/" not in request.url.path:
            raise httpx.ReadTimeout("no answer", request=request)
        return fake(request)
    return answer


READBACK_CASES = {
    "set_resources": (lambda path: path.endswith("/config"), lambda service, clock: service.guest_set_resources(
        "pve-a", 990, cores=2, memory_mib=None, expected_type="lxc", expected_name="janus-lab",
        expected_digest="d990", sleep=clock.sleep, clock=clock.monotonic)),
    "set_options": (lambda path: path.endswith("/config"), lambda service, clock: service.guest_set_options(
        "pve-a", 990, tags="db", on_boot=None, expected_type="lxc", expected_name="janus-lab",
        expected_digest="d990", sleep=clock.sleep, clock=clock.monotonic)),
    "resize_disk": (lambda path: path.endswith("/config"), lambda service, clock: service.guest_resize_disk(
        "pve-a", 101, disk="scsi0", grow_gib=4, expected_type="qemu", expected_name="db",
        expected_digest="d101", sleep=clock.sleep, clock=clock.monotonic)),
    "rollback": (lambda path: path.endswith("/snapshot"), lambda service, clock: service.guest_snapshot_rollback(
        "pve-a", 990, snapshot="known-good", expected_type="lxc", expected_name="janus-lab",
        sleep=clock.sleep, clock=clock.monotonic)),
    "create": (lambda path: path.endswith("/cluster/resources"), lambda service, clock: service.guest_create_lxc(
        "pve-a", 993, hostname="lab-2", template="local:vztmpl/alpine-3.23-default_20260116_amd64.tar.xz",
        cores=1, memory_mib=128, disk_gib=2, storage="local-lvm", pool=None, description=None,
        sleep=clock.sleep, clock=clock.monotonic)),
    "clone": (lambda path: path.endswith("/cluster/resources"), lambda service, clock: service.guest_clone(
        "pve-a", 990, newid=994, name="lab-clone", full=True, pool=None, expected_type="lxc",
        expected_name="janus-lab", sleep=clock.sleep, clock=clock.monotonic)),
    "backup": (lambda path: path.endswith("/content"), lambda service, clock: service.backup_now(
        "pve-a", 990, storage="local", expected_type="lxc", expected_name="janus-lab",
        sleep=clock.sleep, clock=clock.monotonic)),
    "restore": (lambda path: path.endswith("/cluster/resources"), lambda service, clock: service.backup_restore(
        "pve-a", volume="local:backup/vzdump-lxc-990-2026_11_17.tar.zst", newid=995, name="restored",
        storage=None, pool=None, sleep=clock.sleep, clock=clock.monotonic)),
}


@pytest.mark.parametrize("case", sorted(READBACK_CASES))
def test_an_unreadable_result_after_an_accepted_change_is_unknown_and_never_retried(case):
    # Proxmox accepted the change (and finished its task); only the read-back
    # failed. The change probably happened, so the receipt must not say
    # "failed" or invite a retry: it is an unknown outcome to check by hand.
    fake = FakeCluster()
    fake.snapshots[990].append({"name": "known-good"})
    path_test, call = READBACK_CASES[case]
    client = httpx.Client(transport=httpx.MockTransport(_readback_fails(fake, path_test)))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test", TOKEN_ID, TOKEN_SECRET), client)
    try:
        receipt = call(service, Clock())
    finally:
        client.close()
    assert len(fake.writes()) == 1, fake.writes()
    assert (receipt["ok"], receipt["status"], receipt["retry_safe"]) == (False, "outcome_unknown", False)
    assert receipt["verification_ok"] is False
    assert "could not be read back" in receipt["error"]
