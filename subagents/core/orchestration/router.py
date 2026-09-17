from __future__ import annotations

import json

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    SpecialistRequest,
)

from subagents.core.orchestration.intent_contract import (
    build_router_semantic_agent_spec,
    parse_semantic_intent,
)

from subagents.llm.runtime.inference import (
    InferenceEngine,
)

from subagents.llm.runtime.scheduler import (
    InferencePriority,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


class RoutingContractError(
    RuntimeError
):
    """
    Raised when production Hub output cannot be trusted as a valid
    structured routing / semantic-intent contract.

    This is deliberately distinct from:

        no delegation

    A valid empty delegation list means the Primary Assistant may
    handle the request conversationally.

    An invalid routing contract must fail closed instead.
    """


class LLMRouter:
    """
    Hub routing and semantic-intent stage.

    The router reasons over specialists together with trusted
    runtime capability metadata.

    The Hub produces:

        specialist identity
        task instructions
        structured semantic intent

    Semantic intent is descriptive only.

    It is NOT authorization.

    In strict production mode:

        malformed JSON
        malformed delegation
        unknown specialist
        missing semantic intent
        invalid semantic intent
        duplicate same-specialist delegation

    fail closed with RoutingContractError.

    A VALID empty delegation list remains a normal conversational
    path.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        inference: InferenceEngine,
        model_key: str,
        strict_contract: bool = False,
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

        self.strict_contract = (
            strict_contract
        )

    def _build_system_prompt(
        self,
    ) -> str:

        specialists = [
            build_router_semantic_agent_spec(
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

    def _contract_error(
        self,
        message: str,
    ) -> None:

        if (
            self.strict_contract
        ):

            raise (
                RoutingContractError(
                    message
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

                max_new_tokens=512,

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

        except json.JSONDecodeError as exc:

            print(
                "[ROUTER] Invalid JSON "
                f"error={exc}"
            )

            self._contract_error(
                "Hub router returned invalid JSON."
            )

            return []

        if not isinstance(
            parsed,
            dict,
        ):

            self._contract_error(
                "Hub router output must be a JSON object."
            )

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

            self._contract_error(
                "Hub router output is missing a valid "
                "delegations list."
            )

            return []

        # --------------------------------------------------------
        # VALID EMPTY ROUTING RESULT
        #
        # This is deliberately different from malformed routing.
        # --------------------------------------------------------

        if not delegations:

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

                self._contract_error(
                    "Hub router produced a malformed delegation."
                )

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

                self._contract_error(
                    "Hub router delegation is missing a valid "
                    "agent name."
                )

                continue

            if not isinstance(
                instructions,
                str,
            ):

                self._contract_error(
                    "Hub router delegation is missing valid "
                    "instructions."
                )

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

                self._contract_error(
                    "Hub router delegation contains an empty "
                    "agent name."
                )

                continue

            if not instructions:

                self._contract_error(
                    "Hub router delegation contains empty "
                    "instructions."
                )

                continue

            # ----------------------------------------------------
            # TRUSTED AGENT RESOLUTION
            # ----------------------------------------------------

            if not (
                self.registry
                .exists(
                    agent_name
                )
            ):

                self._contract_error(
                    "Hub router referenced an unknown "
                    f"specialist: {agent_name}"
                )

                continue

            # ----------------------------------------------------
            # CURRENT RUNTIME:
            #
            # One delegation per specialist.
            #
            # Silently dropping a second task would lose user
            # intent, so strict production mode rejects it.
            # ----------------------------------------------------

            if (
                agent_name
                in seen_agents
            ):

                self._contract_error(
                    "Hub router produced multiple delegations "
                    "for the same specialist while the current "
                    "runtime supports one delegation per "
                    f"specialist: {agent_name}"
                )

                continue

            agent = (
                self.registry
                .get(
                    agent_name
                )
            )

            # ----------------------------------------------------
            # SEMANTIC INTENT VALIDATION
            # ----------------------------------------------------

            try:

                semantic_intent = (
                    parse_semantic_intent(
                        delegation.get(
                            "intent"
                        ),

                        agent=(
                            agent
                        ),
                    )
                )

            except ValueError as exc:

                print(
                    "[ROUTER] Rejected semantic intent "
                    f"agent={agent_name!r} "
                    f"error={exc}"
                )

                self._contract_error(
                    "Hub router produced an invalid semantic "
                    f"intent for specialist '{agent_name}': "
                    f"{exc}"
                )

                continue

            if (
                semantic_intent
                is None
            ):

                self._contract_error(
                    "Hub router omitted the semantic intent "
                    f"contract for specialist '{agent_name}'."
                )

                # Legacy / isolated non-strict callers remain
                # readable during the transition.

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

                    semantic_intent=(
                        semantic_intent
                    ),
                )
            )

        # --------------------------------------------------------
        # STRICT MODE:
        #
        # A non-empty model delegation list must not collapse into
        # conversational fallback because every delegation was
        # malformed.
        # --------------------------------------------------------

        if (
            self.strict_contract
            and delegations
            and not validated
        ):

            raise (
                RoutingContractError(
                    "Hub router produced no executable valid "
                    "delegation from a non-empty routing result."
                )
            )

        return validated
