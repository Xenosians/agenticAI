"""Governed Proxmox guest power actions: exact target, one POST, task and read-back receipt."""
import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService
from services.provider_errors import ProviderResourceNotFound

TOKEN_ID = "janus@pve!dashboard"
TOKEN_SECRET = "fixture-token-secret"
UPID = "UPID:pve-a:000F1C2B:0A1B2C3D:6738A1B2:vzstart:990:janus@pve!dashboard:"

ROWS = [
    {"id": "lxc/990", "type": "lxc", "vmid": 990, "node": "pve-a", "name": "lab", "status": "stopped", "template": 0},
    {"id": "qemu/101", "type": "qemu", "vmid": 101, "node": "pve-a", "name": "db", "status": "running"},
    {"id": "lxc/991", "type": "lxc", "vmid": 991, "node": "pve-a", "name": "busy", "status": "running",
     "lock": "backup"},
    {"id": "qemu/900", "type": "qemu", "vmid": 900, "node": "pve-a", "name": "golden", "status": "stopped",
     "template": 1},
]


def ok(request, data):
    return httpx.Response(200, json={"data": data}, request=request)


class FakeProxmox:
    """Answers like Proxmox VE 9: the inventory, a power POST returning a task, the task and the guest."""

    def __init__(self, *, post=None, task_polls=1, exitstatus="OK", after=None, rows=ROWS, current=None):
        self.rows = rows
        self.post = post
        self.task_polls = task_polls
        self.exitstatus = exitstatus
        self.after = after
        # The guest's own live state before the action; the inventory row by default.
        self.current = current
        self.posted = False
        self.polls = 0
        self.calls = []

    def __call__(self, request):
        path = request.url.path
        body = request.content.decode() if request.method == "POST" else dict(request.url.params)
        self.calls.append((request.method, path, body))
        if path == "/api2/json/cluster/resources":
            return ok(request, self.rows)
        if request.method == "POST":
            self.posted = True
            if self.post is not None:
                return self.post(request)
            return ok(request, UPID)
        if "/tasks/" in path and path.endswith("/status"):
            self.polls += 1
            if self.polls < self.task_polls:
                return ok(request, {"status": "running", "upid": UPID})
            return ok(request, {"status": "stopped", "exitstatus": self.exitstatus, "upid": UPID})
        if path.endswith("/status/current"):
            vmid = int(path.split("/")[-3])
            if self.posted:
                return ok(request, {"status": self.after, "vmid": vmid})
            row = next(row for row in self.rows if row["vmid"] == vmid)
            live = {"status": self.current or row["status"], "vmid": vmid}
            if row.get("lock"):
                live["lock"] = row["lock"]
            return ok(request, live)
        return httpx.Response(404, json={"data": None}, request=request)

    def posts(self):
        return [(path, body) for method, path, body in self.calls if method == "POST"]


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def run(fake, action="start", node="pve-a", vmid=990, expected_type="lxc", expected_name="lab", clock=None):
    clock = clock or Clock()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test", TOKEN_ID, TOKEN_SECRET), client)
    try:
        return service.guest_power(node, vmid, action, expected_type=expected_type, expected_name=expected_name,
                                   sleep=clock.sleep, clock=clock.monotonic)
    finally:
        client.close()


def test_start_posts_once_waits_for_its_task_and_verifies_the_guest_runs():
    fake = FakeProxmox(task_polls=3, after="running")
    receipt = run(fake)
    assert receipt["ok"] is True
    assert receipt["status"] == "executed"
    assert receipt["verification_ok"] is True
    assert receipt["changed"] is True
    assert (receipt["node"], receipt["vmid"], receipt["type"], receipt["name"]) == ("pve-a", 990, "lxc", "lab")
    assert (receipt["action"], receipt["expected_status"], receipt["observed_status"]) == ("start", "running", "running")
    assert receipt["upid"] == UPID and receipt["task_exit_status"] == "OK"
    assert fake.posts() == [("/api2/json/nodes/pve-a/lxc/990/status/start", "")]
    paths = [path for _, path, _ in fake.calls]
    assert paths[0] == "/api2/json/cluster/resources"
    assert paths[-1] == "/api2/json/nodes/pve-a/lxc/990/status/current"
    assert fake.polls == 3


@pytest.mark.parametrize("action,vmid,kind,name,endpoint,body,after", [
    ("shutdown", 101, "qemu", "db", "shutdown", "timeout=60", "stopped"),
    ("reboot", 101, "qemu", "db", "reboot", "timeout=60", "running"),
    ("force_stop", 101, "qemu", "db", "stop", "", "stopped"),
])
def test_each_action_uses_its_own_proxmox_endpoint(action, vmid, kind, name, endpoint, body, after):
    fake = FakeProxmox(after=after)
    receipt = run(fake, action=action, vmid=vmid, expected_type=kind, expected_name=name)
    assert receipt["ok"] is True and receipt["observed_status"] == after
    assert fake.posts() == [(f"/api2/json/nodes/pve-a/{kind}/{vmid}/status/{endpoint}", body)]


@pytest.mark.parametrize("expected_type,expected_name", [("qemu", "lab"), ("lxc", "other"), ("", "lab")])
def test_a_target_that_no_longer_matches_the_approval_is_not_touched(expected_type, expected_name):
    fake = FakeProxmox()
    receipt = run(fake, expected_type=expected_type, expected_name=expected_name)
    assert (receipt["ok"], receipt["status"]) == (False, "conflict")
    assert receipt["retry_safe"] is True
    assert fake.posts() == []


@pytest.mark.parametrize("action,vmid,kind,name,reason", [
    ("start", 101, "qemu", "db", "already running"),
    ("shutdown", 990, "lxc", "lab", "stopped"),
    ("reboot", 991, "lxc", "busy", "locked (backup)"),
    ("start", 900, "qemu", "golden", "template"),
])
def test_an_action_that_cannot_apply_now_is_refused_without_a_post(action, vmid, kind, name, reason):
    fake = FakeProxmox()
    receipt = run(fake, action=action, vmid=vmid, expected_type=kind, expected_name=name)
    assert (receipt["ok"], receipt["status"]) == (False, "conflict")
    assert reason in receipt["error"]
    assert fake.posts() == []


def test_a_guest_that_is_gone_is_not_found_and_nothing_is_sent():
    fake = FakeProxmox()
    receipt = run(fake, vmid=989)
    assert (receipt["ok"], receipt["status"]) == (False, "not_found")
    assert fake.posts() == []


def test_the_live_state_decides_whether_an_action_applies_not_the_inventory_cache():
    # Proxmox refreshes /cluster/resources every few seconds, so right after a
    # start the inventory can still say "stopped" while the guest already runs.
    fake = FakeProxmox(current="running", after="stopped")
    receipt = run(fake, action="shutdown")
    assert receipt["ok"] is True
    assert fake.posts() == [("/api2/json/nodes/pve-a/lxc/990/status/shutdown", "timeout=60")]


def test_a_guest_the_inventory_still_shows_running_but_is_stopped_is_not_touched():
    fake = FakeProxmox(rows=[dict(ROWS[0], status="running")], current="stopped")
    receipt = run(fake, action="shutdown")
    assert (receipt["ok"], receipt["status"]) == (False, "conflict")
    assert "stopped" in receipt["error"]
    assert fake.posts() == []


@pytest.mark.parametrize("failure", ["refused", "unreachable"])
def test_a_failed_check_before_sending_is_a_plain_error_and_nothing_is_sent(failure):
    fake = FakeProxmox()

    def answer(request):
        if request.url.path.endswith("/status/current"):
            if failure == "refused":
                return httpx.Response(500, json={"data": None}, request=request)
            raise httpx.ConnectError("refused", request=request)
        return fake(request)

    client = httpx.Client(transport=httpx.MockTransport(answer))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test", TOKEN_ID, TOKEN_SECRET), client)
    try:
        receipt = service.guest_power("pve-a", 990, "start", expected_type="lxc", expected_name="lab",
                                      sleep=Clock().sleep, clock=Clock().monotonic)
    finally:
        client.close()
    assert (receipt["ok"], receipt["status"], receipt["retry_safe"]) == (False, "error", True)
    assert "nothing was sent" in receipt["error"].lower()
    assert fake.posts() == []
    assert "fixture-token-secret" not in repr(receipt)


def test_a_failed_task_is_a_failed_receipt_with_the_proxmox_exit_status():
    fake = FakeProxmox(exitstatus="startup for container '990' failed", after="stopped")
    receipt = run(fake)
    assert (receipt["ok"], receipt["status"]) == (False, "failed")
    assert receipt["task_exit_status"] == "startup for container '990' failed"
    assert "startup for container '990' failed" in receipt["error"]
    assert len(fake.posts()) == 1


def test_a_finished_task_whose_guest_reads_back_wrong_is_not_a_success():
    fake = FakeProxmox(after="stopped")
    receipt = run(fake)
    assert (receipt["ok"], receipt["status"], receipt["verification_ok"]) == (False, "failed", False)
    assert receipt["observed_status"] == "stopped"
    assert "stopped" in receipt["error"]


@pytest.mark.parametrize("status,message,code", [
    (403, "Permission check failed (/vms/990, VM.PowerMgmt)", "denied"),
    (500, "CT 990 already running", "failed"),
])
def test_a_refused_post_means_nothing_ran(status, message, code):
    fake = FakeProxmox(post=lambda request: httpx.Response(status, json={"data": None, "message": message},
                                                           request=request))
    receipt = run(fake)
    assert (receipt["ok"], receipt["status"]) == (False, code)
    assert f"HTTP {status}" in receipt["error"] and message in receipt["error"]
    assert receipt["retry_safe"] is True and receipt["upid"] is None


def test_an_unanswered_post_is_an_unknown_outcome_not_a_failure():
    def post(request):
        raise httpx.ReadTimeout("no answer", request=request)

    receipt = run(FakeProxmox(post=post))
    assert (receipt["ok"], receipt["status"]) == (False, "outcome_unknown")
    assert receipt["retry_safe"] is False
    assert "fixture-token-secret" not in repr(receipt)


def test_a_post_that_never_connected_sent_nothing():
    def post(request):
        raise httpx.ConnectError("refused", request=request)

    receipt = run(FakeProxmox(post=post))
    assert (receipt["ok"], receipt["status"], receipt["retry_safe"]) == (False, "error", True)


def test_a_task_still_running_at_the_deadline_is_unknown_and_keeps_its_upid():
    clock = Clock()
    receipt = run(FakeProxmox(task_polls=10_000), clock=clock)
    assert (receipt["ok"], receipt["status"]) == (False, "outcome_unknown")
    assert receipt["upid"] == UPID
    assert clock.now <= 125


def test_an_answer_that_is_not_a_task_of_this_node_is_an_unknown_outcome():
    fake = FakeProxmox(post=lambda request: ok(request, "UPID:other-node:0:0:0:vzstart:990:root@pam:"))
    receipt = run(fake)
    assert (receipt["ok"], receipt["status"]) == (False, "outcome_unknown")
    assert not any("/tasks/" in path for _, path, _ in fake.calls)


def test_power_target_names_the_exact_guest_and_whether_the_action_applies():
    client = httpx.Client(transport=httpx.MockTransport(FakeProxmox()))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test", TOKEN_ID, TOKEN_SECRET), client)
    try:
        target = service.power_target("pve-a", 990, "start")["data"]
        assert target == {"node": "pve-a", "vmid": 990, "type": "lxc", "name": "lab", "status": "stopped",
                          "lock": None, "template": False, "action": "start", "allowed": True, "reason": None}
        refused = service.power_target("pve-a", 990, "force_stop")["data"]
        assert refused["allowed"] is False and "stopped" in refused["reason"]
        with pytest.raises(ProviderResourceNotFound):
            service.power_target("pve-a", 989, "start")
    finally:
        client.close()


@pytest.mark.parametrize("node,vmid,action", [
    ("../x", 990, "start"), ("pve-a", "990", "start"), ("pve-a", True, "start"),
    ("pve-a", 0, "start"), ("pve-a", 990, "delete"), ("pve-a", 990, "stop"),
])
def test_power_arguments_cannot_change_the_request(node, vmid, action):
    fake = FakeProxmox()
    with pytest.raises(ValueError):
        run(fake, node=node, vmid=vmid, action=action)
    assert fake.calls == []
