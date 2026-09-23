from dataclasses import (
    replace,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.tooling.prompt import (
    build_compact_capability_text,
    build_worker_system_prompt,
)


def _agent() -> AgentDefinition:
    return AgentDefinition(
        name="test-specialist",
        description="test",
        tools=[
            "ticket_get",
        ],
        model="small-model",
        max_steps=1,
        system_prompt="Preserve exact values.",
    )


def test_compact_capability_text_preserves_names_arguments_and_enums():
    text = build_compact_capability_text(
        [
            {
                "name": "ticket_get",
                "description": "very long description omitted",
                "argument_schema": {
                    "ticket_key": {
                        "type": "str",
                    },
                    "mode": {
                        "type": "str",
                        "enum": [
                            "a",
                            "b",
                        ],
                    },
                },
            }
        ]
    )

    assert "ticket_get(" in text
    assert "ticket_key:str" in text
    assert "mode:str=[a|b]" in text
    assert "very long description" not in text


def test_compact_worker_prompt_has_strict_json_contract():
    prompt = build_worker_system_prompt(
        _agent(),
        capability_catalog=[
            {
                "name": "ticket_get",
                "description": "read one ticket",
                "argument_schema": {
                    "ticket_key": {
                        "type": "str",
                    },
                },
            }
        ],
        prompt_profile="compact",
    )

    assert "COMPACT TOOL PROTOCOL" in prompt
    assert "ticket_get(ticket_key:str)" in prompt
    assert "RETURN ONLY THE JSON ARRAY" in prompt
    assert "read one ticket" not in prompt
