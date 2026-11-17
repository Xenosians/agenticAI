"""Proxmox and Tailscale use the requesting account's own connection (owner decision, 17 November).

A tool, a dashboard read and a pre-approval check all ask one resolver for the
current owner's service. The account's saved link wins; an account without one
falls back to the machine profile's default; neither gives no service, and the
failure then says the account has not connected the provider.
"""
import pytest

from config import Settings
from services.extensions.context import request_owner
from services.extensions.links import OwnerLinks
from services.extensions.store import ExtensionStore
from services.provider_links import (
    owner_proxmox_service,
    owner_tailnet_service,
    proxmox_setup_missing,
)

PROXMOX = {"url": "https://pve.example.test:8006", "token_id": "janus@pve!dashboard",
           "token_secret": "synthetic-proxmox-secret", "verify_tls": True}
TAILSCALE = {"client_id": "synthetic-client", "client_secret": "synthetic-tailscale-secret"}
MACHINE = Settings(_env_file=None, proxmox_base_url="https://machine.example.test",
                   proxmox_api_token_id="machine@pve!default", proxmox_api_token_secret="synthetic-machine-secret",
                   tailscale_client_id="machine-client", tailscale_client_secret="synthetic-machine-tailscale")


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: store)
    return store


def as_owner(owner, call):
    token = request_owner.set(owner)
    try:
        return call()
    finally:
        request_owner.reset(token)


def test_the_accounts_own_proxmox_link_wins_over_the_machine_default(store):
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    service = as_owner("alice", lambda: owner_proxmox_service(MACHINE))
    try:
        assert service.origin == "https://pve.example.test:8006"
        assert service.config.token_id == "janus@pve!dashboard"
    finally:
        service.close()


def test_an_account_without_a_link_uses_the_machine_default(store):
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    service = as_owner("bob", lambda: owner_proxmox_service(MACHINE))
    try:
        assert service.origin == "https://machine.example.test"
        assert service.config.token_id == "machine@pve!default"
    finally:
        service.close()


def test_no_link_and_no_machine_default_gives_no_service(store):
    assert as_owner("bob", lambda: owner_proxmox_service(Settings(_env_file=None))) is None
    assert as_owner(None, lambda: owner_tailnet_service(Settings(_env_file=None))) is None


def test_outside_a_request_only_the_machine_default_applies(store):
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    service = owner_proxmox_service(MACHINE)
    try:
        assert service.origin == "https://machine.example.test"
    finally:
        service.close()


def test_the_accounts_own_tailscale_link_wins_and_never_mixes_with_another_account(store):
    OwnerLinks(store).save("alice", "tailscale", TAILSCALE)
    alice = as_owner("alice", lambda: owner_tailnet_service(MACHINE))
    bob = as_owner("bob", lambda: owner_tailnet_service(MACHINE))
    try:
        assert alice.config.client_id == "synthetic-client"
        assert bob.config.client_id == "machine-client"
    finally:
        alice.close()
        bob.close()


def test_a_missing_setup_names_the_accounts_connection_not_server_settings(store):
    assert proxmox_setup_missing(Settings(_env_file=None)) == ("Proxmox link in Connections",)
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    assert as_owner("alice", lambda: proxmox_setup_missing(Settings(_env_file=None))) == ()
    assert proxmox_setup_missing(MACHINE) == ()


def test_the_pre_approval_check_reads_the_proposing_accounts_proxmox(store, monkeypatch):
    """The guest frozen into an approval comes from the account that proposes it."""
    from tools.proxmox import catalog

    monkeypatch.setattr("config.Settings", lambda: Settings(_env_file=None))
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    service = as_owner("alice", catalog._policy_service)
    try:
        assert service.origin == "https://pve.example.test:8006"
    finally:
        service.close()
    assert as_owner("bob", catalog._policy_service) is None
