import asyncio

import pytest

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.orchestration.router import (
    LLMRouter,
    RoutingContractError,
)


from subagents.prompts.prompt_loader import (
    load_prompt,
)


class SequencedInference:

    def __init__(
        self,
        responses,
    ):

        self.responses = list(
            responses
        )

        self.calls = []

    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ):

        self.calls.append(
            {
                "model_key":
                    model_key,

                "messages":
                    messages,

                "max_new_tokens":
                    max_new_tokens,

                "priority":
                    priority,
            }
        )

        if not self.responses:

            raise AssertionError(
                "Unexpected extra router generation."
            )

        return (
            self.responses.pop(
                0
            )
        )


def _registry():

    registry = (
        AgentRegistry()
    )

    registry.register(
        AgentDefinition(
            name=(
                "developer-specialist"
            ),

            description=(
                "Handles governed developer Git operations."
            ),

            model=(
                "hub-main"
            ),

            tools=[
                "workspace_git_branches",
                "workspace_git_switch_branch",
            ],
        )
    )

    return registry


def _router(
    responses,
):

    inference = (
        SequencedInference(
            responses
        )
    )

    router = (
        LLMRouter(
            registry=(
                _registry()
            ),

            inference=(
                inference
            ),

            model_key=(
                "hub-main"
            ),

            strict_contract=True,

            max_new_tokens=512,
        )
    )

    return (
        router,
        inference,
    )


INVALID_SYNTHETIC_PREFLIGHT = """
{
  "delegations": [
    {
      "agent": "developer-specialist",
      "instructions":
        "Check whether the target branch exists.",
      "intent": {
        "summary":
          "Check the local branches.",
        "effect": "read",
        "allowed_tools": [
          "workspace_git_branches"
        ],
        "forbidden_tools": [],
        "allowed_arguments": {
          "repository": [
            "ai"
          ]
        },
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": false
      }
    },
    {
      "agent": "developer-specialist",
      "instructions":
        "Switch to test/agentic-git-smoke.",
      "intent": {
        "summary":
          "Switch branches.",
        "effect": "mutation",
        "allowed_tools": [
          "workspace_git_switch_branch"
        ],
        "forbidden_tools": [],
        "allowed_arguments": {
          "repository": [
            "ai"
          ],
          "branch_name": [
            "test/agentic-git-smoke"
          ]
        },
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": false
      },
      "when": {
        "source_agent":
          "developer-specialist",
        "result_field":
          "branches",
        "equals":
          false
      }
    }
  ]
}
"""


VALID_DIRECT_SWITCH = """
{
  "delegations": [
    {
      "agent": "developer-specialist",
      "instructions":
        "Switch the AI repository to test/agentic-git-smoke.",
      "intent": {
        "summary":
          "Switch the AI repository to test/agentic-git-smoke.",
        "effect": "mutation",
        "allowed_tools": [
          "workspace_git_switch_branch"
        ],
        "forbidden_tools": [],
        "allowed_arguments": {
          "repository": [
            "ai"
          ],
          "branch_name": [
            "test/agentic-git-smoke"
          ]
        },
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": false
      }
    }
  ]
}
"""


def test_router_repairs_invalid_plan_with_fresh_valid_plan():

    (
        router,
        inference,
    ) = (
        _router(
            [
                INVALID_SYNTHETIC_PREFLIGHT,
                VALID_DIRECT_SWITCH,
            ]
        )
    )

    result = (
        asyncio.run(
            router.route_with_repair(
                (
                    "Switch the AI repository to "
                    "test/agentic-git-smoke."
                )
            )
        )
    )

    assert (
        len(
            result
        )
        == 1
    )

    delegation = (
        result[
            0
        ]
    )

    assert (
        delegation.agent_name
        == "developer-specialist"
    )

    assert (
        delegation.condition
        is None
    )

    assert (
        delegation.semantic_intent
        is not None
    )

    assert (
        delegation
        .semantic_intent
        .allowed_tools
        == [
            "workspace_git_switch_branch"
        ]
    )

    assert (
        len(
            inference.calls
        )
        == 2
    )

    normal_prompt = (
        inference.calls[
            0
        ][
            "messages"
        ][
            0
        ][
            "content"
        ]
    )

    repair_prompt = (
        inference.calls[
            1
        ][
            "messages"
        ][
            0
        ][
            "content"
        ]
    )

    assert (
        "ROUTING CONTRACT REPAIR ATTEMPT"
        not in normal_prompt
    )

    assert (
        "ROUTING CONTRACT REPAIR ATTEMPT"
        in repair_prompt
    )

    assert (
        inference.calls[
            0
        ][
            "messages"
        ][
            1
        ][
            "content"
        ]
        == inference.calls[
            1
        ][
            "messages"
        ][
            1
        ][
            "content"
        ]
    )


def test_router_repair_still_fails_closed():

    (
        router,
        inference,
    ) = (
        _router(
            [
                INVALID_SYNTHETIC_PREFLIGHT,
                INVALID_SYNTHETIC_PREFLIGHT,
            ]
        )
    )

    with pytest.raises(
        RoutingContractError
    ):

        asyncio.run(
            router.route_with_repair(
                (
                    "Switch the AI repository to "
                    "test/agentic-git-smoke."
                )
            )
        )

    assert (
        len(
            inference.calls
        )
        == 2
    )


def test_valid_first_route_does_not_use_repair():

    (
        router,
        inference,
    ) = (
        _router(
            [
                VALID_DIRECT_SWITCH,
            ]
        )
    )

    result = (
        asyncio.run(
            router.route_with_repair(
                (
                    "Switch the AI repository to "
                    "test/agentic-git-smoke."
                )
            )
        )
    )

    assert (
        len(
            result
        )
        == 1
    )

    assert (
        len(
            inference.calls
        )
        == 1
    )


def test_direct_repair_omits_conditional_protocol_and_gets_validator_feedback():

    (
        router,
        inference,
    ) = (
        _router(
            [
                INVALID_SYNTHETIC_PREFLIGHT,
                VALID_DIRECT_SWITCH,
            ]
        )
    )

    result = (
        asyncio.run(
            router.route_with_repair(
                (
                    "Switch the AI repository to "
                    "test/agentic-git-smoke."
                )
            )
        )
    )

    assert len(result) == 1

    assert (
        len(
            inference.calls
        )
        == 2
    )

    normal_messages = (
        inference.calls[
            0
        ][
            "messages"
        ]
    )

    repair_messages = (
        inference.calls[
            1
        ][
            "messages"
        ]
    )

    normal_system = (
        normal_messages[
            0
        ][
            "content"
        ]
    )

    repair_system = (
        repair_messages[
            0
        ][
            "content"
        ]
    )

    assert (
        "CONDITIONAL WORKFLOW PROTOCOL"
        in normal_system
    )

    assert (
        "CONDITIONAL WORKFLOW PROTOCOL"
        not in repair_system
    )

    assert (
        "ROUTING CONTRACT REPAIR ATTEMPT"
        in repair_system
    )

    assert (
        repair_messages[
            1
        ][
            "content"
        ]
        == (
            "Switch the AI repository to "
            "test/agentic-git-smoke."
        )
    )

    assert (
        len(
            repair_messages
        )
        == 3
    )

    feedback = (
        repair_messages[
            2
        ][
            "content"
        ]
    )

    assert (
        "TRUSTED VALIDATION FEEDBACK"
        in feedback
    )

    assert (
        "not trusted for branching"
        in feedback
    )

    assert (
        "Do not invent replacement condition fields."
        in feedback
    )


def test_repair_prompt_is_generic_but_preserves_conditional_semantics():

    prompt = (
        load_prompt(
            "hub_router_repair.txt"
        )
    )

    assert (
        "policy_owns_preconditions"
        in prompt
    )

    assert (
        "explicitly makes one operation depend"
        in prompt
    )

    assert (
        "keep the mutation conditional"
        in prompt
    )

    assert (
        "Never repair an invalid conditional workflow "
        "by silently converting"
        in prompt
    )

    assert (
        "workspace_git_"
        not in prompt
    )

    assert (
        "AI repository"
        not in prompt
    )

    assert (
        "Git branch"
        not in prompt
    )
