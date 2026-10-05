from __future__ import annotations

from subagents.core.definitions.types import SemanticIntent
from subagents.core.orchestration.semantic_binding import (
    bind_authoritative_grounded_arguments,
)
from tools.registry import TOOLS


def _intent(*, tool_name: str, argument_name: str, values: list[str]) -> SemanticIntent:
    tool = TOOLS[tool_name]
    effect = "read" if str(tool.get("risk", "")).strip().lower() == "read" else "mutation"
    return SemanticIntent(
        summary="generic semantic authority regression",
        effect=effect,
        allowed_tools=[tool_name],
        forbidden_tools=[],
        allowed_arguments={argument_name: values},
        forbidden_arguments={},
        max_tool_calls=1,
        clarification_required=False,
    )


def _expects_collection(tool: dict, argument_name: str) -> bool:
    parameter = tool.get("parameters", {}).get(argument_name, {})
    value_type = parameter.get("type", "") if isinstance(parameter, dict) else ""
    normalized = str(value_type).strip().lower().replace(" ", "")
    return (
        normalized in {"list", "array", "seq"}
        or normalized.startswith("list[")
        or normalized.startswith("array[")
        or normalized.startswith("seq[")
    )


def test_every_registered_grounded_argument_can_be_runtime_bound():
    """Registry-generated invariant; contains no domain-specific tool logic."""
    checked = 0

    for tool_name, tool in TOOLS.items():
        grounded_arguments = tool.get("grounded_arguments", [])
        if not isinstance(grounded_arguments, list):
            continue

        for argument_name in grounded_arguments:
            if not isinstance(argument_name, str):
                continue

            trusted_value = "trusted-target"
            result = bind_authoritative_grounded_arguments(
                intent=_intent(
                    tool_name=tool_name,
                    argument_name=argument_name,
                    values=[trusted_value],
                ),
                tool_name=tool_name,
                arguments={argument_name: "worker-drift-target"},
            )

            expected = (
                [trusted_value]
                if _expects_collection(tool, argument_name)
                else trusted_value
            )

            assert result.arguments[argument_name] == expected
            assert len(result.events) == 1
            assert result.events[0].argument_name == argument_name
            assert result.events[0].action == "rebound"
            checked += 1

    assert checked > 0


def test_missing_single_grounded_value_is_inserted():
    fake_tool = {
        "risk": "read",
        "grounded_arguments": ["target"],
        "parameters": {"target": {"type": "str"}},
    }

    def lookup(name: str) -> dict | None:
        return fake_tool if name == "fake_tool" else None

    intent = SemanticIntent(
        summary="read target",
        effect="read",
        allowed_tools=["fake_tool"],
        allowed_arguments={"target": ["alpha"]},
    )

    result = bind_authoritative_grounded_arguments(
        intent=intent,
        tool_name="fake_tool",
        arguments={},
        tool_lookup=lookup,
    )

    assert result.arguments == {"target": "alpha"}
    assert result.events[0].action == "inserted"


def test_unbound_grounded_value_is_not_invented():
    fake_tool = {
        "risk": "read",
        "grounded_arguments": ["target"],
        "parameters": {"target": {"type": "str"}},
    }

    def lookup(name: str) -> dict | None:
        return fake_tool if name == "fake_tool" else None

    intent = SemanticIntent(
        summary="read target",
        effect="read",
        allowed_tools=["fake_tool"],
        allowed_arguments={},
    )

    result = bind_authoritative_grounded_arguments(
        intent=intent,
        tool_name="fake_tool",
        arguments={"target": "worker-invented"},
        tool_lookup=lookup,
    )

    # Binder deliberately leaves it untouched. SemanticGuard must reject it.
    assert result.arguments["target"] == "worker-invented"
    assert result.events == ()


def test_multiple_allowed_values_are_not_auto_selected():
    fake_tool = {
        "risk": "read",
        "grounded_arguments": ["target"],
        "parameters": {"target": {"type": "str"}},
    }

    def lookup(name: str) -> dict | None:
        return fake_tool if name == "fake_tool" else None

    intent = SemanticIntent(
        summary="read one of several targets",
        effect="read",
        allowed_tools=["fake_tool"],
        allowed_arguments={"target": ["alpha", "beta"]},
    )

    result = bind_authoritative_grounded_arguments(
        intent=intent,
        tool_name="fake_tool",
        arguments={"target": "beta"},
        tool_lookup=lookup,
    )

    assert result.arguments["target"] == "beta"
    assert result.events == ()


def test_non_grounded_arguments_remain_model_proposal_content():
    fake_tool = {
        "risk": "mutation",
        "grounded_arguments": ["target"],
        "derived_arguments": ["summary"],
        "parameters": {
            "target": {"type": "str"},
            "summary": {"type": "str"},
        },
    }

    def lookup(name: str) -> dict | None:
        return fake_tool if name == "fake_tool" else None

    intent = SemanticIntent(
        summary="mutate target",
        effect="mutation",
        allowed_tools=["fake_tool"],
        allowed_arguments={"target": ["alpha"]},
    )

    result = bind_authoritative_grounded_arguments(
        intent=intent,
        tool_name="fake_tool",
        arguments={
            "target": "worker-drift",
            "summary": "user-derived proposal content",
        },
        tool_lookup=lookup,
    )

    assert result.arguments["target"] == "alpha"
    assert result.arguments["summary"] == "user-derived proposal content"


def test_explicitly_forbidden_worker_value_is_not_rebound():

    fake_tool = {
        "risk":
            "read",

        "grounded_arguments": [
            "target",
        ],

        "parameters": {
            "target": {
                "type":
                    "str",
            },
        },
    }

    def lookup(
        name: str,
    ) -> dict | None:

        if name == "fake_tool":
            return fake_tool

        return None

    intent = (
        SemanticIntent(
            summary="read trusted target",
            effect="read",

            allowed_tools=[
                "fake_tool",
            ],

            allowed_arguments={
                "target": [
                    "allowed-target",
                ],
            },

            forbidden_arguments={
                "target": [
                    "forbidden-target",
                ],
            },
        )
    )

    result = (
        bind_authoritative_grounded_arguments(
            intent=(
                intent
            ),

            tool_name=(
                "fake_tool"
            ),

            arguments={
                "target":
                    "forbidden-target",
            },

            tool_lookup=(
                lookup
            ),
        )
    )

    # Preserve the forbidden worker proposal so SemanticGuard sees
    # and rejects the exact semantic exclusion.
    assert (
        result.arguments[
            "target"
        ]
        == "forbidden-target"
    )

    assert (
        result.events
        == ()
    )
