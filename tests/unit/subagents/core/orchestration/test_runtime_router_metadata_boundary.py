from __future__ import annotations

import asyncio

from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition, AgentTask, SemanticIntent
from subagents.core.orchestration.runtime import AgentRuntime


class CaptureInference:
    def __init__(self):
        self.messages = None

    async def generate(self, *, model_key, messages, max_new_tokens, priority, stage, repair_of=None):
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


def test_selected_worker_budget_preserves_current_request_and_target_contract(monkeypatch):
    from types import SimpleNamespace
    from services.extensions.budget import prepare_external_request
    from services.extensions.context import execution_profile

    registry = AgentRegistry()
    registry.register(AgentDefinition(name="account-specialist", description="Account checks",
                                      model="account-model", tools=["account_status"]))
    inference = CaptureInference()
    provenance_calls = []
    monkeypatch.setattr("subagents.core.orchestration.runtime.build_specialist_execution_provenance",
                        lambda **kwargs: provenance_calls.append(kwargs))
    runtime = AgentRuntime(agent_registry=registry, inference=inference, tool_gateway=NeverGateway(),
                           model_profile_resolver=lambda key: SimpleNamespace(worker_prompt_profile="standard"),
                           max_new_tokens=32)
    task = AgentTask(task_id="bounded-worker", agent_name="account-specialist",
                     user_request="Check account status for alice",
                     context={"conversation_context": [{"role": "user", "content": "history " * 10000}]},
                     semantic_intent=SemanticIntent(summary="Check account", effect="read",
                         allowed_tools=["account_status"], forbidden_tools=[],
                         allowed_arguments={"user_id": ["alice"]}, forbidden_arguments={},
                         max_tool_calls=1, clarification_required=False))
    token = execution_profile.set({"backend": "http", "model": "test"})
    try:
        asyncio.run(runtime.run(task))
    finally:
        execution_profile.reset(token)
    budget = prepare_external_request({}, inference.messages, 32, default_request_budget=6000)
    assert budget.messages[-1] == {"role": "user", "content": task.user_request}
    assert any(message["role"] == "system" and '"user_id": ["alice"]' in message["content"]
               for message in budget.messages)
    assert not any("history history" in message["content"] for message in budget.messages)
    assert provenance_calls == []  # Never hash/misattribute local weights for an external call.
