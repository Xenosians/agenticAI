from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition

from learning.training.hub_training_contracts import (
    validate_hub_response_contract,
)


def _registry():
    registry = AgentRegistry()

    registry.register(
        AgentDefinition(
            name="demo-specialist",
            description="Demo",
            model="demo-model",
            tools=[],
            max_steps=1,
            system_prompt="Demo",
        )
    )

    return registry


def test_noop_hub_contract_is_valid():
    result = validate_hub_response_contract(
        response='{"delegations":[]}',
        registry=_registry(),
    )

    assert result == []


def test_hub_contract_rejects_non_object():
    try:
        validate_hub_response_contract(
            response="[]",
            registry=_registry(),
        )

    except ValueError as exc:
        assert "JSON object" in str(exc)

    else:
        raise AssertionError(
            "Non-object Hub response was accepted."
        )
