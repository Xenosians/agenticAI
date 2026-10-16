from __future__ import annotations

import json
import logging

from services.extensions.budget import estimate_messages_tokens
from services.extensions.context import selected_profile
from subagents.core.orchestration.candidate_selector import (
    CandidateSelector,
    CandidateSet,
    DeterministicCandidateSelector,
    build_candidate_agent_specs,
    validate_candidate_set,
)
from subagents.llm.runtime.stages import InferenceStage, resolve_stage_policy
from subagents.core.orchestration.router_catalog import render_router_catalog

from config import (
    get_settings,
)

from learning.evidence.hub_routing import (
    accept_current_hub_routing_attempt,
    capture_hub_routing_generation_async,
    reject_current_hub_routing_attempt,
    reset_hub_routing_trace,
    set_current_hub_routing_parsed_output,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    SpecialistRequest,
)

from subagents.core.orchestration.conversation_context import (
    conversation_messages,
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

logger = logging.getLogger(__name__)


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
        model_profile_resolver=None,
        candidate_selector: CandidateSelector | None = None,
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

        self.model_profile_resolver = (
            model_profile_resolver
        )
        self.candidate_selector = candidate_selector or DeterministicCandidateSelector()

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
        specialists: (
            list[dict]
            | None
        ) = None,
    ) -> str:

        if specialists is None:

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

        specialists_json = render_router_catalog(specialists)

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

    def _routing_input(self, user_request, *, repair_mode, include_workflow_protocol):
        """Disclose a trusted subset only; broad control routing stays local.

        A candidate miss or oversized packet falls back before any selected
        adapter is called. Candidate identities never replace the full registry
        used by semantic validation, approval policy or tool execution.
        """
        candidates: CandidateSet | None = None
        if selected_profile(stage=InferenceStage.SEMANTIC_ROUTING):
            selection = validate_candidate_set(
                self.candidate_selector.select(user_request, self.registry), self.registry
            )
            fallback = selection.fallback_required or not selection.candidates
            if not fallback:
                specialists = build_candidate_agent_specs(selection, self.registry)
                system = load_prompt("hub_router_compact.txt").replace(
                    "{{SPECIALISTS_JSON}}", render_router_catalog(specialists)
                ).replace(
                    "{{WORKFLOW_PROTOCOL}}",
                    load_prompt("hub_workflow_compact.txt") if include_workflow_protocol
                    else "Conditional workflow generation is disabled for this repair.",
                )
                policy = resolve_stage_policy(InferenceStage.SEMANTIC_ROUTING)
                mandatory = estimate_messages_tokens([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_request},
                ])
                fallback = mandatory + min(self.max_new_tokens, policy.output_token_ceiling) > policy.request_token_ceiling
                if not fallback:
                    candidates = selection
            logger.info(
                "routing_candidates candidates=%d capabilities=%d confidence=%.4f widening=%d fallback=%s",
                selection.candidate_count, selection.capability_count, selection.confidence,
                selection.widening_count, fallback,
            )
            if candidates is not None:
                return specialists, system, InferenceStage.SEMANTIC_ROUTING, candidates

        specialists = [build_router_semantic_agent_spec(agent) for agent in self.registry.list_agents()]
        system = self._build_system_prompt(
            repair_mode=repair_mode, include_workflow_protocol=include_workflow_protocol,
            specialists=specialists,
        )
        return specialists, system, InferenceStage.CONTROL_ROUTING, None

    def _contract_error(
        self,
        message: str,
    ) -> None:

        reject_current_hub_routing_attempt(
            message
        )

        if self.strict_contract:

            raise (
                RoutingContractError(
                    message
                )
            )

    async def route_with_repair(
        self,
        user_request: str,
        *,
        context: list[dict[str, str]] | None = None,
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
                    user_request,
                    context=context,
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

                    context=context,

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
        context: list[dict[str, str]] | None = None,
        repair_mode: bool = False,
        include_workflow_protocol: bool = True,
        repair_error: (
            str | None
        ) = None,
    ) -> list[
        SpecialistRequest
    ]:

        if not repair_mode:

            reset_hub_routing_trace()

        context_messages = (
            conversation_messages(
                context,
                max_turns=24,
            )
        )

        specialists, system_content, origin_stage, candidates = self._routing_input(
            user_request, repair_mode=repair_mode,
            include_workflow_protocol=include_workflow_protocol,
        )

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            *context_messages,
            {
                "role": "user",
                "content": user_request,
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

            # Both the feedback and original request remain mandatory under
            # history trimming. The final user message is always the request.
            messages.insert(
                1,
                {
                    "role":
                        "system",

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
                stage=InferenceStage.REPAIR if repair_mode else origin_stage,
                repair_of=origin_stage if repair_mode else None,
            )
        )

        logger.info(
            "routing_generation stage=%s repair=%s input_estimate=%d response_chars=%d",
            origin_stage.value, repair_mode, estimate_messages_tokens(messages), len(response),
        )

        await capture_hub_routing_generation_async(
            model_key=(
                self.model_key
            ),
            model_profile_resolver=(
                self.model_profile_resolver if candidates is None else None
            ),
            specialists=(
                specialists
            ),
            messages=(
                messages
            ),
            context_messages=(
                context_messages
            ),
            user_request=(
                user_request
            ),
            repair_mode=(
                repair_mode
            ),
            include_workflow_protocol=(
                include_workflow_protocol
            ),
            repair_error=(
                repair_error
            ),
            max_new_tokens=(
                self.max_new_tokens
            ),
            response=(
                response
            ),
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

        if isinstance(
            parsed,
            dict,
        ):

            set_current_hub_routing_parsed_output(
                parsed
            )

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

            if candidates is not None:
                # A retrieved ITSM task that the selected model cannot route
                # must not become a successful conversational completion.
                # The existing caller permits one bounded repair; a repeated
                # miss remains an explicit failure with no tool execution.
                message = (
                    "No supported action was selected from the retrieved capabilities. "
                    "Clarify the requested operation or target."
                )
                reject_current_hub_routing_attempt(message)
                raise RoutingContractError(message)

            accept_current_hub_routing_attempt(
                validated_delegation_count=0,
            )

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

            if candidates is not None:
                disclosed = {item.agent_name: set(item.capability_names) for item in candidates.candidates}
                raw_intent = delegation.get("intent")
                allowed = raw_intent.get("allowed_tools") if isinstance(raw_intent, dict) else None
                if (agent_name not in disclosed or not isinstance(allowed, list)
                        or any(not isinstance(name, str) or name not in disclosed[agent_name] for name in allowed)):
                    self._contract_error("Hub router selected an undisclosed candidate capability.")
                    continue

            try:

                semantic_intent = (
                    parse_semantic_intent(
                        delegation.get(
                            "intent"
                        ),

                        agent=(
                            agent
                        ),

                        user_request=(
                            user_request
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

            self._contract_error(
                "Hub router produced no executable valid "
                "delegation from a non-empty routing result."
            )

        accept_current_hub_routing_attempt(
            validated_delegation_count=(
                len(
                    validated
                )
            ),
        )

        return validated
