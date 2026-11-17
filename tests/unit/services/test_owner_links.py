"""Each Janus account connects its own Proxmox and Tailscale (owner decision, 17 November).

The connection belongs to the account, not the computer: it is stored encrypted
for that owner, never listed with its secret, and used for that owner's chat,
tools and dashboard. The machine profile is only the default for an account
that connected nothing.
"""
import asyncio
from types import SimpleNamespace

import pytest

from services.extensions.links import LinkError, OwnerLinks, owner_links
from services.extensions.store import ExtensionStore

PROXMOX = {"url": "https://pve.example.test:8006", "token_id": "janus@pve!dashboard",
           "token_secret": "synthetic-proxmox-secret", "verify_tls": True}
TAILSCALE = {"client_id": "synthetic-client", "client_secret": "synthetic-tailscale-secret"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: store)
    return store


def test_an_account_connects_its_own_proxmox_and_no_other_account_sees_it(store):
    links = OwnerLinks(store)
    saved = links.save("alice", "proxmox", PROXMOX)
    assert saved == {"kind": "proxmox", "host": "pve.example.test", "token_id": "janus@pve!dashboard",
                     "verify_tls": True, "server_name": None, "credential_configured": True}
    assert links.get("alice", "proxmox") == {**PROXMOX, "server_name": None}
    assert links.get("bob", "proxmox") is None
    assert links.list("bob") == []


def test_a_listed_link_never_carries_its_secret_and_the_stored_row_is_encrypted(store):
    links = OwnerLinks(store)
    links.save("alice", "proxmox", PROXMOX)
    links.save("alice", "tailscale", TAILSCALE)
    listed = links.list("alice")
    assert [row["kind"] for row in listed] == ["proxmox", "tailscale"]
    assert "synthetic" not in str(listed).replace("synthetic-client", "")
    assert listed[1] == {"kind": "tailscale", "client_id": "synthetic-client", "credential_configured": True}
    raw = store.path.read_bytes()
    assert b"synthetic-proxmox-secret" not in raw
    assert b"synthetic-tailscale-secret" not in raw


def test_saving_again_replaces_the_link_and_an_empty_secret_keeps_the_stored_one(store):
    links = OwnerLinks(store)
    links.save("alice", "proxmox", PROXMOX)
    links.save("alice", "proxmox", {**PROXMOX, "url": "https://pve2.example.test", "token_secret": ""})
    assert links.get("alice", "proxmox")["url"] == "https://pve2.example.test"
    assert links.get("alice", "proxmox")["token_secret"] == "synthetic-proxmox-secret"
    assert len(links.list("alice")) == 1


def test_a_new_link_needs_its_secret(store):
    with pytest.raises(LinkError) as refused:
        OwnerLinks(store).save("alice", "proxmox", {**PROXMOX, "token_secret": ""})
    assert refused.value.code == "link_secret_required"


@pytest.mark.parametrize("change, code", [
    ({"url": "http://pve.example.test:8006"}, "link_url_invalid"),
    ({"url": "https://user:pass@pve.example.test"}, "link_url_invalid"),
    ({"url": "https://pve.example.test/api2/json"}, "link_url_invalid"),
    ({"url": "https://127.0.0.1:8006"}, "link_url_invalid"),
    ({"url": "https://localhost:8006"}, "link_url_invalid"),
    ({"token_id": "root"}, "link_token_id_invalid"),
    ({"token_id": "janus@pve!x; rm"}, "link_token_id_invalid"),
    ({"verify_tls": "yes"}, "link_verify_tls_invalid"),
    ({"server_name": "bad name!"}, "link_server_name_invalid"),
    ({"token_secret": "x" * 5000}, "link_secret_invalid"),
])
def test_only_a_valid_proxmox_link_is_saved(store, change, code):
    links = OwnerLinks(store)
    with pytest.raises(LinkError) as refused:
        links.save("alice", "proxmox", {**PROXMOX, **change})
    assert refused.value.code == code
    assert links.list("alice") == []


def test_unknown_kinds_and_fields_are_refused(store):
    links = OwnerLinks(store)
    with pytest.raises(LinkError) as refused:
        links.save("alice", "openwrt", PROXMOX)
    assert refused.value.code == "link_kind_invalid"
    with pytest.raises(LinkError) as refused:
        links.save("alice", "proxmox", {**PROXMOX, "extra": 1})
    assert refused.value.code == "link_field_invalid"


def test_removing_a_link_returns_the_account_to_the_machine_default(store):
    links = OwnerLinks(store)
    links.save("alice", "tailscale", TAILSCALE)
    assert links.remove("alice", "tailscale") == {"kind": "tailscale", "removed": True}
    assert links.get("alice", "tailscale") is None
    assert links.remove("alice", "tailscale") == {"kind": "tailscale", "removed": False}


def test_extension_commands_save_list_and_remove_an_accounts_links(store):
    from services.extensions.service import command

    async def run():
        saved = await command("alice", "link_save", {"kind": "proxmox", "link": PROXMOX}, SimpleNamespace())
        listed = await command("alice", "link_list", {}, SimpleNamespace())
        others = await command("bob", "link_list", {}, SimpleNamespace())
        removed = await command("alice", "link_remove", {"kind": "proxmox"}, SimpleNamespace())
        return saved, listed, others, removed

    saved, listed, others, removed = asyncio.run(run())
    assert saved["credential_configured"] is True
    assert "synthetic-proxmox-secret" not in str((saved, listed))
    assert [row["kind"] for row in listed["links"]] == ["proxmox"]
    assert others["links"] == []
    assert removed == {"kind": "proxmox", "removed": True}
    assert owner_links().get("alice", "proxmox") is None


def test_the_connections_catalog_lists_the_accounts_links_without_secrets(store, monkeypatch):
    """The Connections page reads the account's links with the rest of its catalog."""
    from services.extensions import service

    monkeypatch.setattr(service, "specialist_roles", lambda settings: [])
    monkeypatch.setattr(service, "workflow_capabilities", lambda settings: [])
    monkeypatch.setattr(service, "tool_packages", lambda settings: [])
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    settings = SimpleNamespace(model_profiles={})
    alice = asyncio.run(service.command("alice", "list", {}, settings))
    bob = asyncio.run(service.command("bob", "list", {}, settings))
    assert [row["kind"] for row in alice["links"]] == ["proxmox"]
    assert alice["links"][0]["host"] == "pve.example.test"
    assert "synthetic-proxmox-secret" not in str(alice)
    assert bob["links"] == []
