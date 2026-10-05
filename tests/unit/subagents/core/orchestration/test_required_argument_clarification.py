from __future__ import annotations

import asyncio

from subagents.core.definitions.types import SemanticIntent, SpecialistRequest
from subagents.core.orchestration.orchestrator import Orchestrator


class ClarificationRouter:
    async def route_with_repair(self, user_request, *, context=None):
        return [
            SpecialistRequest(
                agent_name="ticket-specialist",
                instructions="audit-only prose",
                semantic_intent=SemanticIntent(
                    summary="Create ticket.",
                    effect="mutation",
                    allowed_tools=["ticket_create"],
                    forbidden_tools=[],
                    allowed_arguments={},
                    forbidden_arguments={},
                    max_tool_calls=1,
                    clarification_required=True,
                    missing_required_arguments=["project_key"],
                ),
            )
        ]


class NeverRuntime:
    async def run(self, task):
        raise AssertionError("specialist must not run during clarification")


class NeverPrimary:
    async def respond(self, *args, **kwargs):
        raise AssertionError("direct response not expected")

    async def synthesize(self, *args, **kwargs):
        raise AssertionError("model synthesis must not replace deterministic clarification")


def test_missing_required_exact_argument_is_conversational_clarification():
    orchestrator = Orchestrator(
        router=ClarificationRouter(),
        runtime=NeverRuntime(),
        primary_assistant=NeverPrimary(),
        require_semantic_intent=True,
    )

    result = asyncio.run(
        orchestrator.run(
            "create an internal meeting ticket for next sunday"
        )
    )

    assert result.status == "success"
    assert "project key" in (result.answer or "").lower()
    assert len(result.results) == 1
    assert result.results[0].status == "clarification_required"
