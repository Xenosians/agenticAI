import json

from subagents.core.registry import AgentRegistry
from subagents.core.types import (
    HubPlan,
    HubTask,
)
from subagents.llm.base import LLMBackend


class LLMPlanner:
    """
    LLM-backed Hub planner.

    The model proposes tasks.

    Python validates:
    - task structure
    - registered agents
    - task IDs
    - dependency references

    The model is never an authorization boundary.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        backend: LLMBackend,
    ) -> None:
        self.registry = registry
        self.backend = backend

    def _build_system_prompt(self) -> str:
        agents = self.registry.list_agents()

        agent_lines = []

        for agent in agents:
            agent_lines.append(
                f"- {agent.name}: {agent.description}"
            )

        agent_text = "\n".join(agent_lines)

        return f"""
You are the Hub planner for an ITSM agent system.

Your job is ONLY to decompose the user's request into
specialist tasks.

Available specialists:

{agent_text}

Rules:

1. Use only specialists listed above.

2. Every distinct operation must be a separate task.

3. Do NOT combine a read operation and a mutation into
   one task.

Example:
"Check whether jdoe is locked. If locked, unlock it."

Must become:
- task to check lock status
- separate task to unlock jdoe

4. Independent tasks MUST NOT depend on each other.

Example:
"Check whether jdoe is locked and whether jdoe has VPN access."

The account check and VPN check are independent.
Both must have an empty depends_on list.

5. Use depends_on only when a task requires the output
   of another task.

6. Conditional actions must:
   - be separate tasks
   - depend on the task producing the required state
   - include a condition

7. Preserve identifiers exactly as provided by the user.
   Never invent usernames, resource names, IDs, or accounts.

8. Do not execute tools.
   Do not claim that anything succeeded.

9. Return ONLY valid JSON.
   No Markdown fences.
   No explanation.

Required schema:

{{
  "tasks": [
    {{
      "id": "task-1",
      "agent": "account-specialist",
      "instruction": "Exact task instruction",
      "identifiers": ["identifier"],
      "depends_on": [],
      "condition": null
    }}
  ]
}}

For an unsupported request return:

{{
  "tasks": []
}}
""".strip()

    def plan(
        self,
        user_request: str,
    ) -> HubPlan:
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            },
            {
                "role": "user",
                "content": user_request,
            },
        ]

        raw = self.backend.generate(
            messages,
            max_new_tokens=384,
        )

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return HubPlan()

        if not isinstance(payload, dict):
            return HubPlan()

        raw_tasks = payload.get("tasks")

        if not isinstance(raw_tasks, list):
            return HubPlan()

        known_agents = {
            agent.name
            for agent in self.registry.list_agents()
        }

        tasks: list[HubTask] = []
        seen_ids: set[str] = set()

        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                return HubPlan()

            task_id = raw_task.get("id")
            agent = raw_task.get("agent")
            instruction = raw_task.get(
                "instruction"
            )
            identifiers = raw_task.get(
                "identifiers",
                [],
            )
            depends_on = raw_task.get(
                "depends_on",
                [],
            )
            condition = raw_task.get(
                "condition"
            )

            if (
                not isinstance(task_id, str)
                or not task_id.strip()
            ):
                return HubPlan()

            if task_id in seen_ids:
                return HubPlan()

            if (
                not isinstance(agent, str)
                or agent not in known_agents
            ):
                return HubPlan()

            if (
                not isinstance(instruction, str)
                or not instruction.strip()
            ):
                return HubPlan()

            if (
                not isinstance(identifiers, list)
                or not all(
                    isinstance(item, str)
                    for item in identifiers
                )
            ):
                return HubPlan()

            if (
                not isinstance(depends_on, list)
                or not all(
                    isinstance(item, str)
                    for item in depends_on
                )
            ):
                return HubPlan()

            if (
                condition is not None
                and not isinstance(condition, str)
            ):
                return HubPlan()

            seen_ids.add(task_id)

            tasks.append(
                HubTask(
                    id=task_id,
                    agent=agent,
                    instruction=instruction,
                    identifiers=identifiers,
                    depends_on=depends_on,
                    condition=condition,
                )
            )

        # Validate dependency graph references only after
        # every task ID has been collected.
        all_ids = {
            task.id
            for task in tasks
        }

        for task in tasks:
            for dependency in task.depends_on:
                if dependency == task.id:
                    return HubPlan()

                if dependency not in all_ids:
                    return HubPlan()

        return HubPlan(
            tasks=tasks
        )