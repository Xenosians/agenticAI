"""Setup readiness for every provider family (SRS 1.9 FR-081, NFR-036).

Each capability's setup state comes from the same configuration its tools
use: an explicitly selected mock is labelled simulated, a missing setting is
named without its value, and a configured provider is never reported as
verified or healthy.
"""
from types import SimpleNamespace

from config import Settings
from services.integration_readiness import integration_readiness, provider_for_capability
from tools.registry import TOOLS


def settings(**values):
    return Settings(_env_file=None, **values)


def by_id(rows):
    return {row["id"]: row for row in rows}


def test_every_registered_capability_belongs_to_exactly_one_provider():
    owners = {}
    for row in integration_readiness(settings()):
        for capability in row["capabilities"]:
            assert capability not in owners, (capability, owners.get(capability), row["id"])
            owners[capability] = row["id"]
    assert set(owners) == set(TOOLS)
    for name in TOOLS:
        assert provider_for_capability(name) == owners[name]


def test_nothing_selected_is_reported_not_configured_with_the_missing_settings_named():
    rows = by_id(integration_readiness(settings()))
    assert rows["directory"]["state"] == "not_configured"
    assert rows["directory"]["missing"] == ["DIRECTORY_BACKEND"]
    assert rows["ticketing"]["missing"] == ["TICKETING_BACKEND"]
    assert rows["assets"]["missing"] == ["ASSET_BACKEND"]
    assert rows["knowledge"]["missing"] == ["KNOWLEDGE_BACKEND"]
    assert rows["jira_admin"]["missing"] == ["JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"]
    assert rows["openwrt"]["missing"] == ["OPENWRT_BASE_URL", "OPENWRT_USERNAME", "OPENWRT_PASSWORD"]
    assert rows["mailbox_lab"]["missing"] == ["MAILBOX_LAB_ENABLED"]
    # Repository work runs inside the configured workspace root; it needs no provider.
    assert rows["workspace"]["state"] == "configured"
    assert rows["workspace"]["missing"] == []


def test_a_selected_provider_without_its_credentials_names_each_missing_setting():
    rows = by_id(integration_readiness(settings(
        directory_backend="ldap", ad_host="ldap.example.test",
        ticketing_backend="jira", jira_base_url="https://example.atlassian.net",
        openwrt_base_url="https://router.example.test", openwrt_username="reader",
    )))
    assert rows["directory"]["state"] == "not_configured"
    assert rows["directory"]["missing"] == ["AD_BIND_USER or AD_BIND_DN", "AD_BIND_PASSWORD", "AD_BASE_DN"]
    assert rows["ticketing"]["missing"] == ["JIRA_EMAIL", "JIRA_API_TOKEN"]
    assert rows["openwrt"]["missing"] == ["OPENWRT_PASSWORD"]


def test_complete_configuration_is_configured_and_never_claims_a_verified_provider():
    rows = by_id(integration_readiness(settings(
        directory_backend="ldap", ad_host="ldap.example.test", ad_bind_user="reader@example.test",
        ad_bind_password="fixture-secret-value", ad_base_dn="DC=example,DC=test",
        ticketing_backend="jira", jira_base_url="https://example.atlassian.net",
        jira_email="agent@example.test", jira_api_token="fixture-token-value",
        atlassian_admin_api_key="fixture-admin-key", atlassian_org_id="fixture-org",
    )))
    for name in ("directory", "ticketing", "service_desk", "jira_admin", "atlassian_admin"):
        assert rows[name]["state"] == "configured", name
        assert rows[name]["missing"] == [], name
    for row in rows.values():
        # Configuration is not a provider observation (FR-081).
        assert row["state"] in {"configured", "simulated", "not_configured"}
        assert "healthy" not in row and "verified" not in row
    serialized = repr(rows)
    for secret in ("fixture-secret-value", "fixture-token-value", "fixture-admin-key", "reader@example.test"):
        assert secret not in serialized


def test_an_explicit_mock_is_labelled_simulated():
    rows = by_id(integration_readiness(settings(
        directory_backend="mock", ticketing_backend="mock", asset_backend="mock", knowledge_backend="mock")))
    for name in ("directory", "ticketing", "assets", "knowledge"):
        assert rows[name]["state"] == "simulated", name
        assert rows[name]["missing"] == [], name
    # Service desk queues exist only on real Jira; a mock ticket backend does not provide them.
    assert rows["service_desk"]["state"] == "not_configured"
    assert rows["service_desk"]["missing"] == ["TICKETING_BACKEND=jira"]


def test_each_provider_names_its_kind_and_mode_without_settings_values():
    for row in integration_readiness(settings(openwrt_base_url="https://router.example.test")):
        assert set(row) == {"id", "name", "kind", "state", "configured", "missing", "capabilities", "mode"}
        assert row["name"] and row["kind"] in {"provider", "local"}
        assert row["mode"] in {"read_only", "read_write"}
        # `configured` keeps the existing integration-status contract: true only
        # when a real provider is fully set up, never for a simulated one.
        assert row["configured"] is (row["state"] == "configured")
        assert all(isinstance(name, str) and name.isupper() or " or " in name or "=" in name
                   for name in row["missing"])
    rows = by_id(integration_readiness(settings()))
    assert rows["openwrt"]["mode"] == "read_only"
    assert rows["directory"]["mode"] == "read_write"


def test_readiness_needs_only_configuration():
    # No provider object is built, so an unreachable host cannot block the catalog.
    rows = integration_readiness(SimpleNamespace(**settings().model_dump()))
    assert rows


def test_operations_setup_follows_the_targets_its_tools_read():
    # Health endpoints default to the backend and AI service; containers need an allowlist.
    derived = by_id(integration_readiness(settings(phoenix_base_url="http://127.0.0.1:4000")))
    assert derived["operations"]["state"] == "configured"
    nothing = by_id(integration_readiness(settings(ai_host="", phoenix_base_url=None)))
    assert nothing["operations"]["state"] == "not_configured"
    assert nothing["operations"]["missing"] == ["OPERATIONS_CONTAINERS or OPERATIONS_HEALTH_URLS"]
