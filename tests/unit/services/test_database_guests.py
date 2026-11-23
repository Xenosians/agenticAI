"""Database guests from OCI images on Proxmox (owner decisions 20 Nov 2026 item 5, 23 Nov 2026 item 2).

"database management - PostgreSQL/MariaDB guests from OCI images, health through a read-only login
stored per account, dumps and restores each approved by the owner"; 23 Nov: "create and start in one
approval, then read its health".

The fake cluster answers like Proxmox VE 9.2 did live:

* 20 Nov: ``POST /nodes/{node}/storage/{storage}/oci-registry-pull`` returns a task UPID and the template
  then lists as ``local:vztmpl/<normalized reference>.tar``; a new guest is listed without a name for a
  few seconds (inventory lag), so it is confirmed from its own config.
* 20 Nov, live matrix: a create that carried ``tags`` failed with HTTP 403 "Permission check failed
  (/vms/101, VM.Config.Options)". Read 23 Nov from hummingbird's PVE sources: ``assert_tag_permissions``
  checks ``/vms/<id>`` without the pool, and a guest that does not exist yet inherits nothing from the
  pool; once created in the pool, the same tags are accepted.
* 23 Nov, PVE sources: ``restore_oci_archive`` sets ``env`` (and the entrypoint) from the image, replacing
  any ``env`` in the create request; ``env`` is one NUL-separated string of ``NAME=value``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
from urllib.parse import unquote

import httpx
import pytest

from services.database_guests import (
    DATABASE_ENGINES,
    DatabaseGuestRefused,
    database_guest_plan,
    environment_names,
    first_boot_environment,
    generate_password,
    merged_environment,
    mysql_native_hash,
    oci_template_volume,
    scram_sha256_verifier,
)
from services.proxmox import ProxmoxConfig, ProxmoxService

NODE = "hummingbird"
# The official images' own environment, as restore_oci_archive copies it into the config.
IMAGE_ENV = {
    "postgres": "PATH=/usr/local/bin:/usr/bin:/bin\0GOSU_VERSION=1.17\0LANG=en_US.utf8\0PG_MAJOR=17\0"
                "PGDATA=/var/lib/postgresql/data",
    "mariadb": "PATH=/usr/local/bin:/usr/bin:/bin\0GOSU_VERSION=1.17\0LANG=C.UTF-8\0MARIADB_VERSION=1:11.8.2",
}


class Cluster:
    def __init__(self, *, templates=(), pull_status="OK", start_status="OK", pool_tags_allowed=True,
                 env_readback=True):
        self.templates = set(templates)
        self.pull_status = pull_status
        self.start_status = start_status
        self.pool_tags_allowed = pool_tags_allowed
        self.env_readback = env_readback
        self.guests: dict[int, dict] = {}
        self.posts: list[tuple[str, dict]] = []
        self.puts: list[tuple[str, dict]] = []
        self.tasks: dict[str, str] = {}
        self.next_id = 120

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api2/json")
        method = request.method
        form = dict(httpx.QueryParams(request.content.decode())) if request.content else {}
        if method == "POST":
            self.posts.append((path, form))
        if method == "PUT":
            self.puts.append((path, form))
        if path == f"/nodes/{NODE}/storage/local/oci-registry-pull" and method == "POST":
            # Live shape: the worker id is the normalized file name (no ':' or '/').
            worker = oci_template_volume("local", form["reference"]).split("/", 1)[1]
            upid = f"UPID:{NODE}:0000A1B2:0001C3D4:6720A000:ociregistrypull:{worker}:janus@pve!itsm:"
            self.tasks[upid] = self.pull_status
            if self.pull_status == "OK":
                self.templates.add(oci_template_volume("local", form["reference"]))
            return httpx.Response(200, json={"data": upid})
        if path.startswith(f"/nodes/{NODE}/tasks/") and path.endswith("/status"):
            upid = unquote(path.split("/tasks/", 1)[1].rsplit("/status", 1)[0])
            return httpx.Response(200, json={"data": {"status": "stopped", "exitstatus": self.tasks.get(upid, "OK")}})
        if path == f"/nodes/{NODE}/storage/local/content":
            return httpx.Response(200, json={"data": [{"volid": volid, "content": "vztmpl"}
                                                      for volid in sorted(self.templates)]})
        if path == "/cluster/nextid":
            return httpx.Response(200, json={"data": str(self.next_id)})
        if path == "/nodes":
            return httpx.Response(200, json={"data": [{"node": NODE, "status": "online"}]})
        if path == "/cluster/resources":
            # A brand-new guest is listed without its name for a few seconds (live, 20 Nov).
            return httpx.Response(200, json={"data": [
                {"type": "lxc", "vmid": vmid, "node": NODE, "status": guest["status"], "name": None,
                 "pool": guest.get("pool"), "tags": guest["config"].get("tags")}
                for vmid, guest in self.guests.items()]})
        if path == f"/nodes/{NODE}/lxc" and method == "POST":
            vmid = int(form["vmid"])
            if "tags" in form:
                # Live 20 Nov: tags are checked on /vms/<id> without the pool before the guest exists.
                return httpx.Response(403, json={"data": None, "message":
                                                 f"Permission check failed (/vms/{vmid}, VM.Config.Options)"})
            image = next(name for name in IMAGE_ENV if f"/{name}_" in form["ostemplate"])
            config = {key: value for key, value in form.items() if key not in ("vmid", "ostemplate", "pool")}
            # restore_oci_archive: the image's env and entrypoint replace anything sent.
            config.update(env=IMAGE_ENV[image], cmode="console", digest="d0")
            self.guests[vmid] = {"status": "stopped", "pool": form.get("pool"), "config": config}
            upid = f"UPID:{NODE}:0000A1B3:0001C3D5:6720A001:vzcreate:{vmid}:janus@pve!itsm:"
            self.tasks[upid] = "OK"
            return httpx.Response(200, json={"data": upid})
        match = re.fullmatch(rf"/nodes/{NODE}/lxc/(\d+)/(config|status/current|status/start)", path)
        if match:
            vmid, tail = int(match.group(1)), match.group(2)
            guest = self.guests.get(vmid)
            if guest is None:
                return httpx.Response(500, json={"data": None, "message": "Configuration file does not exist"})
            if tail == "config" and method == "GET":
                config = dict(guest["config"])
                if not self.env_readback:
                    config.pop("env", None)
                return httpx.Response(200, json={"data": config})
            if tail == "config" and method == "PUT":
                if "tags" in form and not (self.pool_tags_allowed and guest.get("pool")):
                    return httpx.Response(403, json={"data": None, "message":
                                                     f"Permission check failed (/vms/{vmid}, VM.Config.Options)"})
                if form.get("digest") not in (None, guest["config"]["digest"]):
                    return httpx.Response(500, json={"data": None, "message": "config digest mismatch"})
                for key, value in form.items():
                    if key != "digest":
                        # Proxmox stores tags semicolon-separated.
                        guest["config"][key] = value.replace(",", ";") if key == "tags" else value
                guest["config"]["digest"] = f"d{len(self.puts)}"
                return httpx.Response(200, json={"data": None})
            if tail == "status/current":
                return httpx.Response(200, json={"data": {"status": guest["status"],
                                                          "name": guest["config"].get("hostname")}})
            if tail == "status/start" and method == "POST":
                upid = f"UPID:{NODE}:0000A1B4:0001C3D6:6720A002:vzstart:{vmid}:janus@pve!itsm:"
                self.tasks[upid] = self.start_status
                if self.start_status == "OK":
                    guest["status"] = "running"
                return httpx.Response(200, json={"data": upid})
        raise AssertionError((method, path))

    def service(self) -> ProxmoxService:
        return ProxmoxService(ProxmoxConfig("https://pve.example:8006", "janus@pve!itsm", "secret"),
                              httpx.Client(transport=httpx.MockTransport(self.handler)))


class Reachable:
    """The database port answers (or not) the way a TCP connect would."""

    def __init__(self, ok=True):
        self.ok = ok
        self.asked = []

    def __call__(self, address: str, port: int) -> bool:
        self.asked.append((address, port))
        return self.ok


def plan(**changes):
    values = dict(engine="postgresql", version="17-alpine", hostname="db-pg-1", node=NODE, vmid=120,
                  storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                  address="10.77.0.20/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)
    values.update(changes)
    return database_guest_plan(**values)


PASSWORD = "Xq7-generated_by_janus_9f2Kp0aZ"


def create(cluster, the_plan=None, *, reachable=None):
    return cluster.service().database_guest_create(
        the_plan or plan(), password=PASSWORD, reachable=reachable or Reachable(), sleep=lambda _s: None,
        health_wait_seconds=0)


# ---- the plan ----

def test_the_template_volume_is_the_reference_proxmox_normalizes():
    # Live: docker.io/library/postgres:17-alpine listed as local:vztmpl/postgres_17-alpine.tar
    assert oci_template_volume("local", "docker.io/library/postgres:17-alpine") == \
        "local:vztmpl/postgres_17-alpine.tar"
    assert oci_template_volume("local", "docker.io/library/mariadb:11.8") == "local:vztmpl/mariadb_11.8.tar"


def test_only_pinned_official_images_of_known_engines_are_accepted():
    assert set(DATABASE_ENGINES) == {"postgresql", "mariadb"}
    with pytest.raises(DatabaseGuestRefused, match="tag"):
        plan(version="latest")
    with pytest.raises(DatabaseGuestRefused, match="engine"):
        plan(engine="mongodb", version="7")


def test_the_create_request_carries_no_tags_and_no_secret_and_the_tags_follow_in_the_plan():
    the_plan = plan()
    assert the_plan["reference"] == "docker.io/library/postgres:17-alpine"
    assert the_plan["template"] == "local:vztmpl/postgres_17-alpine.tar"
    create_form = the_plan["create"]
    # Live 20 Nov: tags on the create are refused for a pool-scoped token; they are set afterwards.
    assert "tags" not in create_form and the_plan["tags"] == "janus-db,db-postgresql"
    assert create_form["pool"] == "janus"
    assert create_form["net0"] == "name=eth0,bridge=janusnat,ip=10.77.0.20/24,gw=10.77.0.1,firewall=1"
    assert create_form["rootfs"] == "local-lvm:8" and create_form["unprivileged"] == "1"
    # The data directory is a mount point, so a backup and a restore carry the data with the guest.
    assert create_form["mp0"] == "local-lvm:8,mp=/var/lib/postgresql/data,backup=1"
    assert the_plan["port"] == 5432 and the_plan["superuser"] == "postgres"
    # No env and nothing secret in the request or the plan: restore_oci_archive replaces env anyway.
    assert "env" not in create_form
    assert "PASSWORD" not in repr(the_plan)


def test_an_address_outside_the_bridged_subnet_is_refused():
    with pytest.raises(DatabaseGuestRefused, match="10.77.0.0/24"):
        plan(engine="mariadb", version="11.8", address="192.168.1.20/24", gateway="192.168.1.1")


# ---- credentials: only a verifier ever reaches the guest ----

def test_the_postgres_verifier_is_rfc7677_scram_and_never_the_password():
    salt = bytes(range(16))
    verifier = scram_sha256_verifier("pencil", salt=salt, iterations=4096)
    iterations_salt, keys = verifier.removeprefix("SCRAM-SHA-256$").split("$")
    assert iterations_salt == f"4096:{base64.b64encode(salt).decode()}"
    stored, server = keys.split(":")
    salted = hashlib.pbkdf2_hmac("sha256", b"pencil", salt, 4096)
    assert base64.b64decode(stored) == hashlib.sha256(
        hmac.new(salted, b"Client Key", hashlib.sha256).digest()).digest()
    assert base64.b64decode(server) == hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    assert "pencil" not in verifier


def test_the_mariadb_hash_is_mysql_native_password():
    # MariaDB PASSWORD('pencil'): two SHA-1 rounds, upper-case hex, '*' prefix, 41 characters.
    expected = "*" + hashlib.sha1(hashlib.sha1(b"pencil").digest()).hexdigest().upper()
    assert mysql_native_hash("pencil") == expected and len(expected) == 41


def test_first_boot_environment_is_settings_plus_a_verifier_only():
    pg = first_boot_environment("postgresql", PASSWORD)
    assert pg["POSTGRES_PASSWORD"].startswith("SCRAM-SHA-256$") and PASSWORD not in "".join(pg.values())
    # initdb refuses a mount point holding lost+found: the cluster lives one level down.
    assert pg["PGDATA"] == "/var/lib/postgresql/data/pgdata"
    assert first_boot_environment("mariadb", PASSWORD) == {"MARIADB_ROOT_PASSWORD_HASH": mysql_native_hash(PASSWORD)}
    with pytest.raises(DatabaseGuestRefused):
        first_boot_environment("postgresql", "short")
    generated = generate_password()
    assert len(generated) == 43 and generated != generate_password()
    first_boot_environment("mariadb", generated)


def test_the_image_environment_is_kept_and_ours_replaces_only_its_own_names():
    merged = merged_environment(IMAGE_ENV["postgres"],
                                {"PGDATA": "/var/lib/postgresql/data/pgdata", "POSTGRES_PASSWORD": "SCRAM-x"})
    names = environment_names(merged)
    assert names["PATH"] == "/usr/local/bin:/usr/bin:/bin" and names["PG_MAJOR"] == "17"
    assert names["PGDATA"] == "/var/lib/postgresql/data/pgdata" and names["POSTGRES_PASSWORD"] == "SCRAM-x"
    assert merged.count("PGDATA=") == 1
    with pytest.raises(DatabaseGuestRefused):
        merged_environment("", {"BAD": "two\nlines"})


# ---- the approved change: pull, create, configure, start, health ----

def test_pull_create_configure_start_and_health_in_one_approved_change():
    cluster = Cluster()
    reachable = Reachable()
    receipt = create(cluster, reachable=reachable)
    assert (receipt["ok"], receipt["status"], receipt["verification_ok"]) == (True, "executed", True), receipt
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/storage/local/oci-registry-pull",
                                                   f"/nodes/{NODE}/lxc", f"/nodes/{NODE}/lxc/120/status/start"]
    # One configure request, after creation: tags (now a pool member) and the merged environment.
    assert len(cluster.puts) == 1
    put_form = cluster.puts[0][1]
    assert put_form["tags"] == "janus-db,db-postgresql" and put_form["digest"] == "d0"
    env = environment_names(cluster.guests[120]["config"]["env"])
    assert env["PG_MAJOR"] == "17" and env["POSTGRES_PASSWORD"].startswith("SCRAM-SHA-256$")
    assert cluster.guests[120]["status"] == "running"
    assert reachable.asked == [("10.77.0.20", 5432)]
    assert receipt["template_pulled"] is True and receipt["vmid"] == 120 and receipt["name"] == "db-pg-1"
    assert receipt["observed_status"] == "running" and receipt["health"] == "accepting_connections"
    assert receipt["tags"] == "janus-db,db-postgresql" and receipt["type"] == "lxc"
    assert receipt["address"] == "10.77.0.20" and receipt["port"] == 5432
    # The password never appears in the receipt or any request.
    assert PASSWORD not in repr(receipt) and PASSWORD not in repr(cluster.posts + cluster.puts)


def test_a_template_already_on_the_storage_is_not_pulled_again():
    cluster = Cluster(templates={"local:vztmpl/mariadb_11.8.tar"})
    receipt = create(cluster, plan(engine="mariadb", version="11.8", hostname="db-m-1", vmid=121,
                                   address="10.77.0.21/24"))
    assert receipt["ok"] and receipt["template_pulled"] is False, receipt
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/lxc", f"/nodes/{NODE}/lxc/121/status/start"]
    assert environment_names(cluster.guests[121]["config"]["env"])["MARIADB_ROOT_PASSWORD_HASH"].startswith("*")


def test_a_failed_pull_creates_nothing():
    cluster = Cluster(pull_status="command 'skopeo copy' failed: exit code 1")
    receipt = create(cluster)
    assert (receipt["ok"], receipt["status"]) == (False, "failed") and "skopeo" in receipt["error"]
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/storage/local/oci-registry-pull"]
    assert cluster.guests == {} and receipt["guest_may_exist"] is False


def test_a_refused_configure_leaves_the_guest_stopped_and_says_so():
    cluster = Cluster(pool_tags_allowed=False)
    receipt = create(cluster)
    assert (receipt["ok"], receipt["status"]) == (False, "failed"), receipt
    assert receipt["changed"] is True and receipt["retry_safe"] is False and receipt["guest_may_exist"] is True
    assert "120" in receipt["error"] and "not started" in receipt["error"]
    assert cluster.guests[120]["status"] == "stopped"
    assert f"/nodes/{NODE}/lxc/120/status/start" not in [path for path, _ in cluster.posts]


def test_an_environment_that_does_not_read_back_is_never_started():
    # Started without the verifier, the image would refuse to initialise or come up without a password.
    cluster = Cluster(env_readback=False)
    receipt = create(cluster)
    assert (receipt["ok"], receipt["status"]) == (False, "outcome_unknown"), receipt
    assert cluster.guests[120]["status"] == "stopped"


def test_a_guest_that_starts_but_does_not_answer_is_reported_not_retried():
    cluster = Cluster()
    receipt = create(cluster, reachable=Reachable(ok=False))
    assert (receipt["ok"], receipt["status"]) == (False, "failed"), receipt
    assert receipt["observed_status"] == "running" and receipt["health"] == "not_answering"
    assert receipt["retry_safe"] is False
    assert len([path for path, _ in cluster.posts if path.endswith("/status/start")]) == 1


def test_a_failed_start_is_reported_with_the_guest_left_configured():
    cluster = Cluster(start_status="startup for container '120' failed")
    receipt = create(cluster)
    assert (receipt["ok"], receipt["status"]) == (False, "failed") and "120" in receipt["error"], receipt
    assert cluster.guests[120]["config"]["tags"] == "janus-db;db-postgresql"
