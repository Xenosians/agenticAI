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
            "You are the routing hub for an ITSM multi-agent system.\n\n"

            "Your ONLY task is to select which specialist worker or "
            "workers should handle the user's request.\n\n"

            "DO NOT answer the user's request.\n"
            "DO NOT call tools.\n"
            "DO NOT explain your reasoning.\n"
            "DO NOT invent agent names.\n\n"

            "ROUTING RULES:\n\n"

            "Use account-specialist for requests about:\n"
            "- account status\n"
            "- locked or unlocked accounts\n"
            "- enabled or disabled accounts\n"
            "- unlocking an account\n"
            "- password resets\n"
            "- account lifecycle operations\n\n"

            "Use access-specialist for requests about:\n"
            "- VPN access\n"
            "- application access\n"
            "- permissions\n"
            "- authorization\n"
            "- group membership\n"
            "- resource access\n\n"

            "If the request contains BOTH account-management work "
            "and access-management work, return BOTH specialists.\n\n"

            "If the request does not match any available specialist, "
            "return an empty list.\n\n"

            "AVAILABLE SPECIALISTS:\n"
            f"{specialists_json}\n\n"

            "EXAMPLES:\n\n"

            'User: "Is jdoe locked?"\n'
            'Output: {"agents": ["account-specialist"]}\n\n'

            'User: "Is jdoe enabled?"\n'
            'Output: {"agents": ["account-specialist"]}\n\n'

            'User: "Unlock jdoe"\n'
            'Output: {"agents": ["account-specialist"]}\n\n'

            'User: "Reset the password for jdoe"\n'
            'Output: {"agents": ["account-specialist"]}\n\n'

            'User: "Does jdoe have VPN access?"\n'
            'Output: {"agents": ["access-specialist"]}\n\n'

            'User: "Does jdoe have access to GitLab?"\n'
            'Output: {"agents": ["access-specialist"]}\n\n'

            'User: "Check whether jdoe is locked and whether '
            'jdoe has VPN access."\n'
            'Output: {"agents": ["account-specialist", '
            '"access-specialist"]}\n\n'

            'User: "Tell me a joke."\n'
            'Output: {"agents": []}\n\n'

            "Return ONLY one valid JSON object in this exact form:\n"
            '{"agents": ["agent-name"]}\n\n'

            "No markdown. No explanation. No additional text."
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
        
    