import json

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.llm.inference import (
    InferenceEngine,
)

from subagents.llm.scheduler import (
    InferencePriority,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


class LLMRouter:
    """
    Uses the configured Hub model profile to select specialist
    workers.

    The router does not own model weights or backend objects.

    All model execution passes through the shared inference
    boundary so GPU admission, model residency, and future
    scheduling policy remain outside orchestration code.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        inference: InferenceEngine,
        model_key: str,
    ) -> None:
        self.registry = (
            registry
        )

        self.inference = (
            inference
        )

        self.model_key = (
            model_key
        )

    def _build_system_prompt(
        self,
    ) -> str:
        specialists = []

        for agent in (
            self.registry
            .list_agents()
        ):
            specialists.append(
                {
                    "name":
                        agent.name,

                    "description":
                        agent.description,
                }
            )

        specialists_json = (
            json.dumps(
                specialists,
                indent=2,
            )
        )

        template = (
            load_prompt(
                "hub_router.txt"
            )
        )

        return template.replace(
            "{{SPECIALISTS_JSON}}",
            specialists_json,
        )

    async def route(
        self,
        user_request: str,
    ) -> list[str]:
        messages = [
            {
                "role":
                    "system",

                "content":
                    self._build_system_prompt(),
            },

            {
                "role":
                    "user",

                "content":
                    user_request,
            },
        ]

        response = (
            await self.inference.generate(
                model_key=(
                    self.model_key
                ),
                messages=(
                    messages
                ),
                max_new_tokens=128,
                priority=(
                    InferencePriority
                    .HUB_ROUTING
                ),
            )
        )

        print(
            "\n===== HUB ROUTER ====="
            f"\nUSER: {user_request}"
            f"\nRAW: {response}"
            "\n======================"
        )

        try:
            parsed = (
                json.loads(
                    response
                )
            )

        except json.JSONDecodeError:
            return []

        if not isinstance(
            parsed,
            dict,
        ):
            return []

        routes = (
            parsed.get(
                "agents"
            )
        )

        if not isinstance(
            routes,
            list,
        ):
            return []

        validated_routes = []

        for route in routes:
            if not isinstance(
                route,
                str,
            ):
                continue

            # Hub cannot invent specialist names.
            if self.registry.exists(
                route
            ):
                validated_routes.append(
                    route
                )

        # Remove duplicates while preserving order.
        return list(
            dict.fromkeys(
                validated_routes
            )
        )