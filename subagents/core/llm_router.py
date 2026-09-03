import json

from subagents.core.registry import AgentRegistry
from subagents.llm.base import LLMBackend
from subagents.prompts.prompt_loader import load_prompt


class LLMRouter:
    """
    Uses the hub model to select specialist workers.

    The hub only decides WHO should handle a request.
    It does not select or execute tools.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        backend: LLMBackend,
    ) -> None:
        self.registry = registry
        self.backend = backend

    def _build_system_prompt(self) -> str:
        specialists = []

        for agent in self.registry.list_agents():
            specialists.append(
                {
                    "name": agent.name,
                    "description": agent.description,
                }
            )

        specialists_json = json.dumps(
            specialists,
            indent=2,
        )

        template = load_prompt(
            "hub_router.txt"
        )

        return template.replace(
            "{{SPECIALISTS_JSON}}",
            specialists_json,
        )

    def route(
        self,
        user_request: str,
    ) -> list[str]:
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

        response = self.backend.generate(
            messages,
            max_new_tokens=128,
        )
        
        print(
            "\n===== HUB ROUTER ====="
            f"\nUSER: {user_request}"
            f"\nRAW: {response}"
            "\n======================"
        )

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, dict):
            return []

        routes = parsed.get("agents")

        if not isinstance(routes, list):
            return []

        validated_routes = []

        for route in routes:
            if not isinstance(route, str):
                continue

            # Hard validation:
            # hub cannot invent specialist names.
            if self.registry.exists(route):
                validated_routes.append(
                    route
                )

        # Remove duplicates while preserving order.
        return list(
            dict.fromkeys(
                validated_routes
            )
        )
        
    