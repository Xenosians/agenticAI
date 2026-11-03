"""What Janus can do right now, from the live registry and setup state (3 November).

Janus once said it had no networking tools while 12 networking reads were set
up. Capability answers come from this summary, built from the registry and the
same readiness rows /v1/integrations shows; nothing is listed from a prompt.
"""
import json

import pytest

from services import capability_context
from services.capability_context import capability_summary, capability_messages
from services.integration_readiness import PROVIDER_FAMILIES
from tools.registry import TOOLS


def row(family, state, missing=()):
    return {"id": family.id, "name": family.name, "kind": family.kind, "state": state,
            "configured": state == "configured", "missing": list(missing),
            "capabilities": sorted(family.capabilities), "mode": "read_only"}


def family(family_id):
    return next(item for item in PROVIDER_FAMILIES if item.id == family_id)


def readiness(**states):
    """Every family configured unless named: state or (state, missing)."""
    rows = []
    for item in PROVIDER_FAMILIES:
        value = states.get(item.id, "configured")
        state, missing = value if isinstance(value, tuple) else (value, ())
        rows.append(row(item, state, missing))
    return rows


def area(summary, family_id):
    return next(entry for entry in summary["areas"] if entry["id"] == family_id)


def test_every_registered_capability_is_counted_once_with_its_approval_rule():
    summary = capability_summary(readiness())
    counted = sum(entry["reads"] + entry["changes"] for entry in summary["areas"])
    assert counted == len(TOOLS)
    gated = sum(entry["changes"] for entry in summary["areas"])
    assert gated == sum(1 for spec in TOOLS.values() if spec.get("requires_approval"))
    openwrt = area(summary, "openwrt")
    assert openwrt["state"] == "available" and openwrt["reads"] == len(family("openwrt").capabilities)


def test_states_come_from_setup_never_from_the_prompt():
    summary = capability_summary(readiness(
        openwrt=("not_configured", ["OPENWRT_BASE_URL"]), ticketing="simulated"))
    assert area(summary, "openwrt")["state"] == "needs_setup"
    assert area(summary, "openwrt")["missing"] == ["OPENWRT_BASE_URL"]
    assert area(summary, "ticketing")["state"] == "simulated"
    assert area(summary, "tailscale")["state"] == "available"


def test_a_family_missing_from_readiness_is_unknown_not_available():
    rows = [item for item in readiness() if item["id"] != "dns_check"]
    assert area(capability_summary(rows), "dns_check")["state"] == "unknown"
    assert area(capability_summary(None), "openwrt")["state"] == "unknown"


def test_examples_are_registered_operation_names_and_missing_values_are_names_only():
    summary = capability_summary(readiness(jira_admin=("not_configured", ["JIRA_API_TOKEN"])))
    for entry in summary["areas"]:
        assert set(entry["examples"]) <= set(TOOLS)
        assert len(entry["examples"]) <= capability_context.MAX_EXAMPLES
    assert area(summary, "jira_admin")["missing"] == ["JIRA_API_TOKEN"]


def test_unsupported_work_is_named_as_unavailable():
    # Nothing registered captures packets, so Janus must not offer it.
    summary = capability_summary(readiness())
    assert not any("capture" in name or "wireshark" in name for name in TOOLS)
    assert "packet capture" in summary["not_available"]


def test_the_model_message_is_bounded_evidence_not_instructions():
    (message,) = capability_messages(readiness())
    header, body = message["content"].split("\n", 1)
    assert message["role"] == "user"
    assert "not the current request" in header
    payload = json.loads(body)
    assert {entry["id"] for entry in payload["areas"]} == {item.id for item in PROVIDER_FAMILIES}
    assert len(message["content"].encode()) <= capability_context.MAX_MESSAGE_BYTES


def test_without_readiness_no_message_claims_anything():
    assert capability_messages(None) == []


@pytest.mark.parametrize("bad", [[{"id": "openwrt"}], [{"id": "openwrt", "state": 7}], ["openwrt"]])
def test_malformed_readiness_rows_count_as_unknown(bad):
    assert area(capability_summary(bad), "openwrt")["state"] == "unknown"
