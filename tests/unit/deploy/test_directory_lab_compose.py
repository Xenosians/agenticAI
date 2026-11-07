"""The directory lab keeps its own Compose identity and the volumes of its provisioned domain.

Without a project name, Compose derives one from the folder (`itsm-agent`), the
same name another checkout on this machine resolves to. Bringing that stack up
then replaces or orphans the lab. The lab's domain, users and TLS CA live in the
three named volumes, so the file must keep pointing at exactly those.
"""
from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[3] / "docker-compose.yaml"


def compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_the_lab_has_its_own_compose_project_name():
    assert compose()["name"] == "itsm-directory-lab"


def test_the_lab_reuses_the_volumes_of_the_provisioned_domain():
    volumes = compose()["volumes"]
    assert {key: value["name"] for key, value in volumes.items()} == {
        "samba_etc": "itsm-agent_samba_etc",
        "samba_lib": "itsm-agent_samba_lib",
        "samba_cache": "itsm-agent_samba_cache",
    }


def test_the_domain_controller_keeps_the_identity_its_certificate_names():
    service = compose()["services"]["samba-ad"]
    assert service["container_name"] == "itsm-samba-ad"
    assert service["hostname"] == "itsmdc"
    # The NetBIOS host name must differ from the domain name, or provisioning fails.
    environment = service["environment"]
    assert environment["SAMBA_NETBIOS"].endswith(":-ITSMDC}")
    assert environment["SAMBA_DOMAIN"].endswith(":-ITSM}")
    assert service["restart"] == "unless-stopped"
    assert "1636:636" in service["ports"]
