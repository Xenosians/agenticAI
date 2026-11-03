"""What Janus can do right now: a compact summary of the registry and setup state.

Janus answers "what can you do?" from this summary rather than from its own
description, so a capability is offered only when the registry has it and
its provider is set up (SRS 1.9 FR-081, NFR-036). Built from
``tools.registry.TOOLS`` and the readiness rows ``/v1/integrations`` returns;
setup is configuration only, so "available" never claims a provider is
reachable or that an action will succeed.

Area states:
    available    the provider is set up
    simulated    a mock provider was selected (demo/test data, not real)
    needs_setup  a required setting is missing; ``missing`` names it
    unknown      no readable setup row (nothing is claimed)
"""
from __future__ import annotations

import json
from typing import Any

from services.integration_readiness import PROVIDER_FAMILIES
from tools.registry import TOOLS

MAX_EXAMPLES = 4
MAX_MESSAGE_BYTES = 6_000
_STATES = {"configured": "available", "simulated": "simulated", "not_configured": "needs_setup"}
# Work that no registered operation performs. Named so the model says it is
# unavailable instead of offering it; the registry check in the tests keeps it true.
NOT_AVAILABLE = ("packet capture", "remote desktop", "configuration changes on network devices")
HEADER = ("CURRENT CAPABILITIES (runtime setup, not the current request; describe only "
          "these, and say that a change needs the user's approval before it runs):")


def _rows_by_id(readiness: Any) -> dict[str, dict]:
    if not isinstance(readiness, list):
        return {}
    rows = {}
    for row in readiness:
        if (isinstance(row, dict) and isinstance(row.get("id"), str)
                and isinstance(row.get("state"), str)):
            rows[row["id"]] = row
    return rows


def capability_summary(readiness: Any) -> dict:
    """One entry per provider area: state, read and change counts, a few operation names."""
    rows = _rows_by_id(readiness)
    areas = []
    for family in PROVIDER_FAMILIES:
        row = rows.get(family.id)
        state = _STATES.get(row["state"], "unknown") if row else "unknown"
        names = sorted(name for name in family.capabilities if name in TOOLS)
        changes = [name for name in names if TOOLS[name].get("requires_approval")]
        reads = [name for name in names if not TOOLS[name].get("requires_approval")]
        entry = {"id": family.id, "name": family.name, "state": state,
                 "reads": len(reads), "changes": len(changes),
                 "examples": (reads[:2] + changes[:2])[:MAX_EXAMPLES]}
        missing = row.get("missing") if row else None
        if state == "needs_setup" and isinstance(missing, list):
            entry["missing"] = [name for name in missing if isinstance(name, str)]
        areas.append(entry)
    return {"areas": areas, "not_available": list(NOT_AVAILABLE)}


def capability_messages(readiness: Any) -> list[dict[str, str]]:
    """The summary as one quoted user message, or none when setup cannot be read."""
    if not _rows_by_id(readiness):
        return []
    body = json.dumps(capability_summary(readiness), ensure_ascii=False, separators=(",", ":"))
    content = HEADER + "\n" + body
    if len(content.encode("utf-8")) > MAX_MESSAGE_BYTES:
        return []
    return [{"role": "user", "content": content}]
