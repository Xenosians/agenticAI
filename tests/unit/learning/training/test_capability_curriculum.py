from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import learning.training.capability_curriculum as module


@dataclass
class FakeAgent:
    name: str
    description: str
    model: str
    tools: list[str]
    max_steps: int = 3
    system_prompt: str = ""


def _fake_tools():
    return {
        "read_item": {
            "description":
                "READ the current item state for one exact item identifier.",
            "risk":
                "read",
            "requires_approval":
                False,
            "grounded_arguments": [
                "item_id"
            ],
            "parameters": {
                "item_id": {
                    "type":
                        "str",
                    "description":
                        "Exact item identifier.",
                }
            },
        },
        "delete_item": {
            "description":
                "DELETE exactly one item identified by the exact item identifier.",
            "risk":
                "high",
            "requires_approval":
                True,
            "policy_owns_preconditions":
                True,
            "grounded_arguments": [
                "item_id"
            ],
            "parameters": {
                "item_id": {
                    "type":
                        "str",
                    "description":
                        "Exact item identifier.",
                }
            },
        },
    }


def test_catalog_driven_generation_covers_reads_mutations_and_typos(
    monkeypatch,
    tmp_path: Path,
):
    agent = FakeAgent(
        name="fake-specialist",
        description="fake",
        model="hub-main",
        tools=[
            "read_item",
            "delete_item",
        ],
    )

    tools = _fake_tools()

    monkeypatch.setattr(
        module,
        "load_agent_definition",
        lambda _path: agent,
    )

    monkeypatch.setattr(
        module,
        "get_tool",
        lambda name: tools.get(name),
    )

    monkeypatch.setattr(
        module,
        "build_agent_capability_catalog",
        lambda _agent, include_arguments=True: [
            {
                "name": name,
                "description":
                    tools[name]["description"],
                "argument_schema":
                    tools[name]["parameters"],
            }
            for name in agent.tools
        ],
    )

    records = module.build_agent_capability_records(
        project_root=tmp_path,
        agent_definition=Path("fake.md"),
    )

    assert {
        record.tool_name
        for record in records
    } == {
        "read_item",
        "delete_item",
    }

    assert any(
        record.variant == "typo"
        for record in records
    )

    destructive = [
        record
        for record in records
        if record.tool_name == "delete_item"
    ]

    assert destructive
    assert all(
        record.effect == "mutation"
        for record in destructive
    )
    assert all(
        record.risk == "high"
        for record in destructive
    )
    assert all(
        record.requires_approval is True
        for record in destructive
    )


def test_hub_targets_preserve_existing_semantic_contract():
    record = module.CapabilityLanguageRecord(
        record_id="r1",
        split="train",
        agent_name="jira-specialist",
        tool_name="ticket_create",
        user_request=(
            "create a ticket. "
            "Use these exact values: project_key=OPS; "
            "summary=Login fails."
        ),
        arguments={
            "project_key": "OPS",
            "summary": "Login fails",
        },
        semantic_context={
            "allowed_tools": [
                "ticket_create"
            ],
            "allowed_arguments": {
                "project_key": [
                    "OPS"
                ],
                "summary": [
                    "Login fails"
                ],
            },
            "forbidden_tools": [],
            "forbidden_arguments": {},
        },
        target_response=(
            '[{"arguments":{"project_key":"OPS",'
            '"summary":"Login fails"},'
            '"name":"ticket_create"}]'
        ),
        effect="mutation",
        risk="medium",
        requires_approval=True,
        variant="canonical",
        tags=[],
    )

    payload = json.loads(
        module.build_hub_chosen_response(
            record
        )
    )

    delegation = payload["delegations"][0]

    assert (
        delegation["agent"]
        == "jira-specialist"
    )

    intent = delegation["intent"]

    assert (
        intent["allowed_tools"]
        == [
            "ticket_create"
        ]
    )

    assert (
        intent["effect"]
        == "mutation"
    )

    assert (
        intent["clarification_required"]
        is False
    )


def test_typo_augmentation_does_not_corrupt_exact_argument_values():
    request = dict(
        module._request_variants(
            tool_name="ticket_create",
            description=(
                "CREATE one Jira ticket from explicit user values."
            ),
            arguments={
                "project_key":
                    "OPS",
                "summary":
                    "VPN login fails after reboot",
            },
        )
    )["typo"]

    assert "OPS" in request
    assert (
        "VPN login fails after reboot"
        in request
    )
