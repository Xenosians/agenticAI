import json

from subagents.core.registry import AgentRegistry
from subagents.llm.base import LLMBackend


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

        return (
            "You are the routing hub for an ITSM "
            "multi-agent system.\n\n"

            "Your ONLY responsibility is deciding which "
            "specialist worker or workers should handle "
            "the user's request.\n\n"

            "Do NOT call tools.\n"
            "Do NOT solve the ITSM request yourself.\n"
            "Do NOT invent specialist names.\n\n"

            "Available specialists:\n"
            f"{specialists_json}\n\n"

            "Respond with ONLY valid JSON using this format:\n"
            '{"agents": ["specialist-name"]}\n\n'

            "For requests involving multiple independent "
            "specialties, include multiple agents:\n"
            '{"agents": ["specialist-a", "specialist-b"]}\n\n'

            "If no specialist matches, return:\n"
            '{"agents": []}'
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