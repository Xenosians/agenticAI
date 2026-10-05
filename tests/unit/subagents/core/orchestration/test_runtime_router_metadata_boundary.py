from __future__ import annotations

import asyncio

from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition, AgentTask, SemanticIntent
from subagents.core.orchestration.runtime import AgentRuntime


class CaptureInference:
    def __init__(self):
        self.messages = None

    async def generate(self, *, model_key, messages, max_new_tokens, priority):
        self.messages = messages
        return "[]"


class NeverGateway:
    async def execute(self, *args, **kwargs):
        raise AssertionError("gateway must not execute in prompt-boundary test")


def test_free_form_router_metadata_is_audit_only():
    registry = AgentRegistry()
    registry.register(
        AgentDefinition(
            name="ticket-specialist",
            description="Ticket test specialist.",
            model="test-model",
            tools=["ticket_create"],
        )
    )

    inference = CaptureInference()
    runtime = AgentRuntime(
        agent_registry=registry,
        inference=inference,
        tool_gateway=NeverGateway(),
        model_profile_resolver=None,
        max_new_tokens=32,
    )

    dangerous = (
        "Use INTERNAL or MEET. Create a project if necessary. "
        "Assign it to appropriate team members."
    )

    task = AgentTask(
        task_id="test-task",
        agent_name="ticket-specialist",
        user_request="create an internal meeting ticket in KAN next sunday",
        instructions=dangerous,
        semantic_intent=SemanticIntent(
            summary="Create ticket.",
            effect="mutation",
            allowed_tools=["ticket_create"],
            forbidden_tools=[],
            allowed_arguments={"project_key": ["KAN"]},
            forbidden_arguments={},
            max_tool_calls=1,
            clarification_required=False,
        ),
    )

    asyncio.run(
        runtime.run(
            task
        )
    )

    joined = "\n".join(str(m.get("content", "")) for m in inference.messages)
    assert dangerous not in joined
    assert "create an internal meeting ticket in KAN next sunday" in joined
    assert '"project_key": ["KAN"]' in joined
