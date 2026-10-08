from __future__ import annotations

import hashlib
import json

from dataclasses import asdict
from pathlib import Path
from typing import Any

from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import SpecialistRequest
from subagents.core.orchestration.condition_contract import parse_result_condition
from subagents.core.orchestration.intent_contract import (
    build_router_semantic_agent_spec,
    parse_semantic_intent,
)
from subagents.prompts.prompt_loader import load_prompt
from subagents.core.orchestration.router_catalog import render_router_catalog


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_hub_registry(*, agent_directory: Path) -> AgentRegistry:
    registry = AgentRegistry()
    registry.register_many(load_agent_directory(agent_directory))
    return registry


def build_hub_training_environment(*, agent_directory: Path) -> dict[str, Any]:
    """
    Reconstruct the current trusted Hub routing environment without
    invoking model inference.
    """
    registry = build_hub_registry(agent_directory=agent_directory)

    specialists = [
        build_router_semantic_agent_spec(agent)
        for agent in registry.list_agents()
    ]

    specialists_json = render_router_catalog(specialists)

    system_prompt = (
        load_prompt("hub_router.txt").replace(
            "{{SPECIALISTS_JSON}}",
            specialists_json,
        )
        + "\n\n"
        + load_prompt("hub_conditional_workflow.txt")
    )

    agent_payload = [
        asdict(
            agent
        )
        for agent in sorted(
            registry.list_agents(),
            key=lambda item: item.name,
        )
    ]

    return {
        "registry": registry,
        "specialists": specialists,
        "system_prompt": system_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "capability_catalog_sha256": sha256_text(
            canonical_json(specialists)
        ),
        "agent_definitions_sha256": sha256_text(
            canonical_json(agent_payload)
        ),
    }


def validate_hub_response_contract(
    *,
    response: str,
    registry: AgentRegistry,
) -> list[SpecialistRequest]:
    """
    Pure trusted validation of one Hub response.

    This mirrors the important LLMRouter contract boundaries while
    executing no capability.
    """
    try:
        parsed = json.loads(response)
    except Exception as exc:
        raise ValueError("Hub response is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Hub response must be a JSON object.")

    delegations = parsed.get("delegations")

    if not isinstance(delegations, list):
        raise ValueError(
            "Hub response must contain a delegations list."
        )

    if not delegations:
        return []

    validated: list[SpecialistRequest] = []
    unconditional_agents: set[str] = set()
    conditional_agents: set[str] = set()

    for delegation in delegations:
        if not isinstance(delegation, dict):
            raise ValueError("Hub delegation must be an object.")

        agent_name = delegation.get("agent")
        instructions = delegation.get("instructions")

        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError(
                "Hub delegation is missing a valid agent."
            )

        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(
                "Hub delegation is missing valid instructions."
            )

        agent_name = agent_name.strip()
        instructions = instructions.strip()

        if not registry.exists(agent_name):
            raise ValueError(
                "Hub response references unknown specialist: "
                f"{agent_name}"
            )

        agent = registry.get(agent_name)

        semantic_intent = parse_semantic_intent(
            delegation.get("intent"),
            agent=agent,
        )

        if semantic_intent is None:
            raise ValueError(
                "Hub response omitted the semantic intent "
                f"for specialist '{agent_name}'."
            )

        condition = parse_result_condition(
            delegation.get("when"),
            prior_delegations=validated,
            target_intent=semantic_intent,
        )

        if condition is None:
            if (
                agent_name in unconditional_agents
                or agent_name in conditional_agents
            ):
                raise ValueError(
                    "Hub response contains multiple delegations "
                    "for one specialist without a valid conditional "
                    f"workflow: {agent_name}"
                )
            unconditional_agents.add(agent_name)
        else:
            if agent_name in conditional_agents:
                raise ValueError(
                    "Hub response contains multiple conditional "
                    f"delegations for specialist: {agent_name}"
                )
            conditional_agents.add(agent_name)

        validated.append(
            SpecialistRequest(
                agent_name=agent_name,
                instructions=instructions,
                semantic_intent=semantic_intent,
                condition=condition,
            )
        )

    return validated
