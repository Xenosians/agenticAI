from subagents.core.llm_planner import LLMPlanner
from subagents.core.registry import AgentRegistry
from subagents.core.types import AgentDefinition


class FakeBackend:
    def __init__(self, response: str):
        self.response = response

    def generate(
        self,
        messages,
        max_new_tokens=256,
    ):
        return self.response


def build_registry():
    registry = AgentRegistry()

    registry.register(
        AgentDefinition(
            name="account-specialist",
            description="Handles account operations.",
            tools=["account_status", "unlock_user"],
        )
    )

    registry.register(
        AgentDefinition(
            name="access-specialist",
            description="Handles access checks.",
            tools=["check_access"],
        )
    )

    return registry


def test_planner_accepts_independent_tasks():
    backend = FakeBackend(
        """
{
  "tasks": [
    {
      "id": "task-1",
      "agent": "account-specialist",
      "instruction": "Check jdoe lock status.",
      "identifiers": ["jdoe"],
      "depends_on": [],
      "condition": null
    },
    {
      "id": "task-2",
      "agent": "access-specialist",
      "instruction": "Check jdoe VPN access.",
      "identifiers": ["jdoe"],
      "depends_on": [],
      "condition": null
    }
  ]
}
"""
    )

    planner = LLMPlanner(
        registry=build_registry(),
        backend=backend,
    )

    plan = planner.plan(
        "Check whether jdoe is locked and "
        "whether jdoe has VPN access."
    )

    assert len(plan.tasks) == 2

    assert plan.tasks[0].depends_on == []
    assert plan.tasks[1].depends_on == []


def test_planner_accepts_conditional_dependency():
    backend = FakeBackend(
        """
{
  "tasks": [
    {
      "id": "task-1",
      "agent": "account-specialist",
      "instruction": "Check jdoe lock status.",
      "identifiers": ["jdoe"],
      "depends_on": [],
      "condition": null
    },
    {
      "id": "task-2",
      "agent": "account-specialist",
      "instruction": "Unlock jdoe.",
      "identifiers": ["jdoe"],
      "depends_on": ["task-1"],
      "condition": "task-1.locked == true"
    }
  ]
}
"""
    )

    planner = LLMPlanner(
        registry=build_registry(),
        backend=backend,
    )

    plan = planner.plan(
        "If jdoe is locked, unlock it."
    )

    assert len(plan.tasks) == 2

    unlock_task = plan.tasks[1]

    assert unlock_task.depends_on == [
        "task-1"
    ]

    assert unlock_task.condition is not None


def test_planner_rejects_unknown_agent():
    backend = FakeBackend(
        """
{
  "tasks": [
    {
      "id": "task-1",
      "agent": "evil-agent",
      "instruction": "Do something.",
      "identifiers": [],
      "depends_on": [],
      "condition": null
    }
  ]
}
"""
    )

    planner = LLMPlanner(
        registry=build_registry(),
        backend=backend,
    )

    plan = planner.plan("Do something.")

    assert plan.tasks == []


def test_planner_rejects_unknown_dependency():
    backend = FakeBackend(
        """
{
  "tasks": [
    {
      "id": "task-1",
      "agent": "account-specialist",
      "instruction": "Unlock jdoe.",
      "identifiers": ["jdoe"],
      "depends_on": ["task-999"],
      "condition": "task-999.locked == true"
    }
  ]
}
"""
    )

    planner = LLMPlanner(
        registry=build_registry(),
        backend=backend,
    )

    plan = planner.plan(
        "Unlock jdoe if needed."
    )

    assert plan.tasks == []