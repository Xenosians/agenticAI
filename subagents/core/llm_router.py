from __future__ import annotations

import json

from subagents.core.capabilities import (
    build_router_agent_spec,
)

from subagents.core.registry import (
    AgentRegistry,
)

from subagents.core.types import (
    SpecialistRequest,
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
    Hub routing stage.

    The router reasons over specialists together with their trusted
    runtime capability catalogs.

    Agent descriptions provide high-level domain context.

    Capability metadata comes from the trusted registry and is the
    authoritative description of what each specialist can actually
    do.

    Routing instructions remain advisory task context only.

    Authorization remains inside ToolGateway.
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

        specialists = [
            build_router_agent_spec(
                agent
            )

            for agent
            in (
                self.registry
                .list_agents()
            )
        ]

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

        return (
            template
            .replace(
                "{{SPECIALISTS_JSON}}",
                specialists_json,
            )
        )

    async def route(
        self,
        user_request: str,
    ) -> list[
        SpecialistRequest
    ]:

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

                max_new_tokens=256,

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

        delegations = (
            parsed.get(
                "delegations"
            )
        )

        if not isinstance(
            delegations,
            list,
        ):

            return []

        validated: list[
            SpecialistRequest
        ] = []

        seen_agents: set[
            str
        ] = set()

        for delegation in (
            delegations
        ):

            if not isinstance(
                delegation,
                dict,
            ):

                continue

            agent_name = (
                delegation.get(
                    "agent"
                )
            )

            instructions = (
                delegation.get(
                    "instructions"
                )
            )

            if not isinstance(
                agent_name,
                str,
            ):

                continue

            if not isinstance(
                instructions,
                str,
            ):

                continue

            agent_name = (
                agent_name
                .strip()
            )

            instructions = (
                instructions
                .strip()
            )

            if not agent_name:

                continue

            if not instructions:

                continue

            # ----------------------------------------------------
            # TRUSTED AGENT RESOLUTION
            #
            # Hub output cannot create a new specialist simply by
            # naming one.
            # ----------------------------------------------------

            if not (
                self.registry
                .exists(
                    agent_name
                )
            ):

                continue

            # ----------------------------------------------------
            # CURRENT RUNTIME:
            # one delegation per specialist.
            # ----------------------------------------------------

            if agent_name in seen_agents:

                continue

            seen_agents.add(
                agent_name
            )

            validated.append(
                SpecialistRequest(
                    agent_name=(
                        agent_name
                    ),

                    instructions=(
                        instructions
                    ),
                )
            )

        return validated