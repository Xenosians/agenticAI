from __future__ import annotations

import json

from config import (
    get_settings,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    SpecialistRequest,
)

from subagents.core.orchestration.condition_contract import (
    parse_result_condition,
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
    routing / semantic contract.
    """


class LLMRouter:
    """
    Hub routing and semantic-intent stage.

    Normal execution still permits one unconditional delegation per
    specialist.

    Result-aware workflow execution may additionally contain one
    validated conditional mutation for a specialist.

    Conditions do not grant authorization.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        inference: InferenceEngine,
        model_key: str,
        strict_contract: bool = False,
        max_new_tokens: (
            int | None
        ) = None,
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

        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens
            is not None
            else (
                get_settings()
                .hub_router_max_new_tokens
            )
        )

        if (
            not isinstance(
                self.max_new_tokens,
                int,
            )
            or self.max_new_tokens
            < 1
        ):

            raise ValueError(
                "Router max_new_tokens must "
                "be a positive integer."
            )

    def _build_system_prompt(
        self,
        *,
        repair_mode: bool = False,
        include_workflow_protocol: bool = True,
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

        base_template = (
            load_prompt(
                "hub_router.txt"
            )
        )

        workflow_template = (
            load_prompt(
                "hub_conditional_workflow.txt"
            )
        )

        system_prompt = (
            base_template.replace(
                "{{SPECIALISTS_JSON}}",
                specialists_json,
            )
        )

        if include_workflow_protocol:

            system_prompt = (
                system_prompt
                + "\n\n"
                + workflow_template
            )

        if repair_mode:

            repair_template = (
                load_prompt(
                    "hub_router_repair.txt"
                )
            )

            system_prompt = (
                system_prompt
                + "\n\n"
                + repair_template
            )

        return (
            system_prompt
        )

    def _contract_error(
        self,
        message: str,
    ) -> None:

        if self.strict_contract:

            raise (
                RoutingContractError(
                    message
                )
            )

    async def route_with_repair(
        self,
        user_request: str,
    ) -> list[
        SpecialistRequest
    ]:
        """
        Run one normal routing generation.

        If strict trusted contract validation rejects that plan,
        permit exactly one fresh repair generation.

        The repair generation receives:
            - the original user request
            - the trusted validation error
            - the generic repair protocol

        It does not receive the verbose conditional-workflow
        protocol, avoiding reinforcement of an invalid synthetic
        workflow pattern.

        The repaired plan is validated from scratch through the
        exact same trusted contract.

        No rejected plan is executed.
        """

        try:

            return (
                await self.route(
                    user_request
                )
            )

        except RoutingContractError as exc:

            if not self.strict_contract:

                raise

            print(
                "[ROUTER] Trusted contract rejected "
                "normal plan; attempting one bounded "
                f"repair generation error={exc}"
            )

            print(
                "[ROUTER] Repair prompt "
                "conditional_protocol=False"
            )

            return (
                await self.route(
                    user_request,

                    repair_mode=True,

                    include_workflow_protocol=False,

                    repair_error=(
                        str(
                            exc
                        )
                    ),
                )
            )


    async def route(
        self,
        user_request: str,
        *,
        repair_mode: bool = False,
        include_workflow_protocol: bool = True,
        repair_error: (
            str | None
        ) = None,
    ) -> list[
        SpecialistRequest
    ]:

        messages = [
            {
                "role":
                    "system",

                "content":
                    self._build_system_prompt(
                        repair_mode=(
                            repair_mode
                        ),

                        include_workflow_protocol=(
                            include_workflow_protocol
                        ),
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    user_request,
            },
        ]

        if (
            repair_mode
            and isinstance(
                repair_error,
                str,
            )
            and repair_error.strip()
        ):

            messages.append(
                {
                    "role":
                        "user",

                    "content":
                        (
                            "TRUSTED VALIDATION FEEDBACK:\n"
                            + repair_error.strip()
                            + "\n\n"
                            "Recompute the routing plan from the "
                            "original request. Do not repeat the "
                            "rejected contract structure. Do not "
                            "invent replacement condition fields."
                        ),
                }
            )

        response = (
            await self.inference.generate(
                model_key=(
                    self.model_key
                ),

                messages=(
                    messages
                ),

                max_new_tokens=(
                    self.max_new_tokens
                ),

                priority=(
                    InferencePriority
                    .HUB_ROUTING
                ),
            )
        )

        print(
            "\n===== HUB ROUTER ====="
            f"\nMODE: "
            f"{'repair' if repair_mode else 'normal'}"
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

        if not delegations:

            return []

        validated: list[
            SpecialistRequest
        ] = []

        unconditional_agents: set[
            str
        ] = set()

        conditional_agents: set[
            str
        ] = set()

        for delegation in delegations:

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
                agent_name.strip()
            )

            instructions = (
                instructions.strip()
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

            if not (
                self.registry.exists(
                    agent_name
                )
            ):

                self._contract_error(
                    "Hub router referenced an unknown "
                    f"specialist: {agent_name}"
                )

                continue

            agent = (
                self.registry.get(
                    agent_name
                )
            )

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

            if semantic_intent is None:

                self._contract_error(
                    "Hub router omitted the semantic intent "
                    f"contract for specialist '{agent_name}'."
                )

            try:

                condition = (
                    parse_result_condition(
                        delegation.get(
                            "when"
                        ),

                        prior_delegations=(
                            validated
                        ),

                        target_intent=(
                            semantic_intent
                        ),
                    )
                )

            except ValueError as exc:

                print(
                    "[ROUTER] Rejected workflow condition "
                    f"agent={agent_name!r} "
                    f"error={exc}"
                )

                self._contract_error(
                    "Hub router produced an invalid workflow "
                    f"condition for specialist '{agent_name}': "
                    f"{exc}"
                )

                continue

            if condition is None:

                if (
                    agent_name
                    in unconditional_agents
                    or agent_name
                    in conditional_agents
                ):

                    self._contract_error(
                        "Hub router produced multiple "
                        "delegations for the same specialist "
                        "without a valid conditional workflow: "
                        f"{agent_name}"
                    )

                    continue

                unconditional_agents.add(
                    agent_name
                )

            else:

                if (
                    agent_name
                    in conditional_agents
                ):

                    self._contract_error(
                        "Hub router produced multiple "
                        "conditional delegations for the same "
                        f"specialist: {agent_name}"
                    )

                    continue

                conditional_agents.add(
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

                    condition=(
                        condition
                    ),
                )
            )

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
