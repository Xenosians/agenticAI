"""Database guests from OCI images on Proxmox (owner decision 20 Nov 2026, ~14:10 WIB, item 5).

"database management - PostgreSQL/MariaDB guests from OCI images, health through a read-only login
stored per account, dumps and restores each approved by the owner".

The fake cluster answers like Proxmox VE 9.2 did live on 20 Nov 2026: ``POST
/nodes/{node}/storage/{storage}/oci-registry-pull`` returns a task UPID; the template then lists as
``local:vztmpl/<normalized reference>.tar``; ``POST /nodes/{node}/lxc`` with ``entrypoint`` absent keeps
the image's own entrypoint, and ``env`` is one NUL-separated string; a new guest is listed without a name
for a few seconds (inventory lag), so it is confirmed from its own config.
"""
from __future__ import annotations

import httpx
import pytest

from services.database_guests import (
    DATABASE_ENGINES,
    DatabaseGuestRefused,
    database_guest_plan,
    oci_template_volume,
)
from services.proxmox import ProxmoxConfig, ProxmoxService

NODE = "hummingbird"


class Cluster:
    def __init__(self, *, templates=(), pull_status="OK"):
        self.templates = set(templates)
        self.pull_status = pull_status
        self.guests: dict[int, dict] = {}
        self.posts: list[tuple[str, dict]] = []
        self.tasks: dict[str, str] = {}
        self.next_id = 120

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api2/json")
        method = request.method
        form = dict(httpx.QueryParams(request.content.decode())) if request.content else {}
        if method == "POST":
            self.posts.append((path, form))
        if path == f"/nodes/{NODE}/storage/local/oci-registry-pull" and method == "POST":
            # Live shape: the worker id is the normalized file name (no ':' or '/').
            worker = oci_template_volume("local", form["reference"]).split("/", 1)[1]
            upid = f"UPID:{NODE}:0000A1B2:0001C3D4:6720A000:ociregistrypull:{worker}:janus@pve!itsm:"
            self.tasks[upid] = self.pull_status
            if self.pull_status == "OK":
                self.templates.add(oci_template_volume("local", form["reference"]))
            return httpx.Response(200, json={"data": upid})
        if path.startswith(f"/nodes/{NODE}/tasks/") and path.endswith("/status"):
            upid = path.split("/tasks/", 1)[1].rsplit("/status", 1)[0]
            from urllib.parse import unquote
            upid = unquote(upid)
            return httpx.Response(200, json={"data": {"status": "stopped", "exitstatus": self.tasks.get(upid, "OK")}})
        if path == f"/nodes/{NODE}/storage/local/content":
            return httpx.Response(200, json={"data": [{"volid": volid, "content": "vztmpl"}
                                                      for volid in sorted(self.templates)]})
        if path == "/cluster/nextid":
            return httpx.Response(200, json={"data": str(self.next_id)})
        if path == "/nodes":
            return httpx.Response(200, json={"data": [{"node": NODE, "status": "online"}]})
        if path == "/cluster/resources":
            return httpx.Response(200, json={"data": [
                {"type": "lxc", "vmid": vmid, "node": NODE, "status": "stopped", "name": None,
                 "pool": guest.get("pool"), "tags": guest.get("tags")}
                for vmid, guest in self.guests.items()]})
        if path == f"/nodes/{NODE}/lxc" and method == "POST":
            vmid = int(form["vmid"])
            self.guests[vmid] = {"hostname": form["hostname"], "pool": form.get("pool"), "tags": form.get("tags"),
                                 "form": form}
            upid = f"UPID:{NODE}:0000A1B3:0001C3D5:6720A001:vzcreate:{vmid}:janus@pve!itsm:"
            self.tasks[upid] = "OK"
            return httpx.Response(200, json={"data": upid})
        if path.startswith(f"/nodes/{NODE}/lxc/") and path.endswith("/config"):
            vmid = int(path.split("/")[4])
            guest = self.guests.get(vmid)
            if guest is None:
                return httpx.Response(500, json={"data": None, "message": "Configuration file does not exist"})
            return httpx.Response(200, json={"data": {"hostname": guest["hostname"], "digest": "d" + str(vmid)}})
        if path.startswith(f"/nodes/{NODE}/lxc/") and path.endswith("/status/current"):
            vmid = int(path.split("/")[4])
            if vmid not in self.guests:
                return httpx.Response(500, json={"data": None, "message": "does not exist"})
            return httpx.Response(200, json={"data": {"status": "stopped", "name": self.guests[vmid]["hostname"]}})
        raise AssertionError((method, path))

    def service(self) -> ProxmoxService:
        return ProxmoxService(ProxmoxConfig("https://pve.example:8006", "janus@pve!itsm", "secret"),
                              httpx.Client(transport=httpx.MockTransport(self.handler)))


def test_the_template_volume_is_the_reference_proxmox_normalizes():
    # Live: docker.io/library/postgres:17-alpine listed as local:vztmpl/postgres_17-alpine.tar
    assert oci_template_volume("local", "docker.io/library/postgres:17-alpine") == \
        "local:vztmpl/postgres_17-alpine.tar"
    assert oci_template_volume("local", "docker.io/library/mariadb:11.8") == "local:vztmpl/mariadb_11.8.tar"


def test_only_pinned_official_images_of_known_engines_are_accepted():
    assert set(DATABASE_ENGINES) == {"postgresql", "mariadb"}
    with pytest.raises(DatabaseGuestRefused, match="tag"):
        database_guest_plan(engine="postgresql", version="latest", hostname="db-1", node=NODE, vmid=120,
                            storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                            address="10.77.0.20/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)
    with pytest.raises(DatabaseGuestRefused, match="engine"):
        database_guest_plan(engine="mongodb", version="7", hostname="db-1", node=NODE, vmid=120,
                            storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                            address="10.77.0.20/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)


def test_the_plan_puts_the_guest_in_the_lab_pool_on_the_bridged_subnet_with_its_data_volume():
    plan = database_guest_plan(engine="postgresql", version="17-alpine", hostname="db-pg-1", node=NODE, vmid=120,
                               storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                               address="10.77.0.20/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)
    assert plan["reference"] == "docker.io/library/postgres:17-alpine"
    assert plan["template"] == "local:vztmpl/postgres_17-alpine.tar"
    create = plan["create"]
    assert create["pool"] == "janus" and create["tags"] == "janus-db,db-postgresql"
    assert create["net0"] == "name=eth0,bridge=janusnat,ip=10.77.0.20/24,gw=10.77.0.1,firewall=1"
    assert create["rootfs"] == "local-lvm:8" and create["unprivileged"] == "1"
    # The data directory is a mount point, so a backup and a restore carry the data with the guest.
    assert create["mp0"] == "local-lvm:8,mp=/var/lib/postgresql/data,backup=1"
    assert plan["port"] == 5432
    # No password is ever part of the plan: the image starts with its trust-local setup and Janus
    # sets the passwords afterwards through the guest's own console.
    assert not any("PASSWORD" in value for value in create.values())


def test_an_address_outside_the_bridged_subnet_is_refused():
    with pytest.raises(DatabaseGuestRefused, match="10.77.0.0/24"):
        database_guest_plan(engine="mariadb", version="11.8", hostname="db-m-1", node=NODE, vmid=121,
                            storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                            address="192.168.1.20/24", gateway="192.168.1.1", disk_gib=8, memory_mib=1024, cores=1)


def test_pull_then_create_sends_one_pull_and_one_create_and_confirms_the_guest():
    cluster = Cluster()
    service = cluster.service()
    plan = database_guest_plan(engine="postgresql", version="17-alpine", hostname="db-pg-1", node=NODE, vmid=120,
                               storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                               address="10.77.0.20/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)
    receipt = service.database_guest_create(plan, sleep=lambda _s: None)
    assert (receipt["ok"], receipt["status"], receipt["verification_ok"]) == (True, "executed", True), receipt
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/storage/local/oci-registry-pull",
                                                   f"/nodes/{NODE}/lxc"]
    assert receipt["template_pulled"] is True and receipt["vmid"] == 120 and receipt["name"] == "db-pg-1"


def test_a_template_already_on_the_storage_is_not_pulled_again():
    cluster = Cluster(templates={"local:vztmpl/mariadb_11.8.tar"})
    plan = database_guest_plan(engine="mariadb", version="11.8", hostname="db-m-1", node=NODE, vmid=121,
                               storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                               address="10.77.0.21/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)
    receipt = cluster.service().database_guest_create(plan, sleep=lambda _s: None)
    assert receipt["ok"] and receipt["template_pulled"] is False
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/lxc"]


def test_a_failed_pull_creates_nothing():
    cluster = Cluster(pull_status="command 'skopeo copy' failed: exit code 1")
    plan = database_guest_plan(engine="postgresql", version="17-alpine", hostname="db-pg-1", node=NODE, vmid=120,
                               storage="local-lvm", template_storage="local", pool="janus", vnet="janusnat",
                               address="10.77.0.20/24", gateway="10.77.0.1", disk_gib=8, memory_mib=1024, cores=1)
    receipt = cluster.service().database_guest_create(plan, sleep=lambda _s: None)
    assert (receipt["ok"], receipt["status"]) == (False, "failed") and "skopeo" in receipt["error"]
    assert [path for path, _ in cluster.posts] == [f"/nodes/{NODE}/storage/local/oci-registry-pull"]
    assert cluster.guests == {}
