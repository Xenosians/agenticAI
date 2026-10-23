from __future__ import annotations

import json
import logging
from agent.requested_workflow import (
    RequestedWorkflowContract, admit_requested_workflow,
    requested_workflow_prompt, validate_requested_plan,
)

from services.extensions.budget import (
    ExternalInferenceBudgetError,
    estimate_messages_tokens,
    prepare_external_request,
)
from services.extensions.context import execution_selection, selected_profile
from subagents.core.orchestration.candidate_selector import (
    Candidate,
    CandidateSelector,
    CandidateSet,
    DeterministicCandidateSelector,
    build_candidate_agent_specs,
    validate_candidate_set,
)
from subagents.llm.runtime.stages import InferenceStage, resolve_stage_policy
from subagents.core.orchestration.router_prompt import (
    build_compact_router_system_prompt,
    build_router_system_prompt,
)

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

logger = logging.getLogger(__name__)


def selected_request_budget(profile: dict | None) -> int:
    """Effective request budget of the selected connection (0 when none)."""
    if not profile:
        return 0
    from services.extensions.budget import default_external_request_budget, prepare_external_request

    default = default_external_request_budget(profile, backend=profile["backend"])
    return prepare_external_request(profile, [], 1, default_request_budget=default).request_token_budget


class RoutingContractError(
    RuntimeError
):
    """
    Raised when production Hub output cannot be trusted as a valid
    routing / semantic contract.
    """


class CandidateMissError(RoutingContractError):
    """The selected model found no matching capability in a retrieved subset.

    Retrieval may have omitted the right capability, so the single repair
    attempt re-routes against the full trusted catalog instead of the subset.
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
        router_prompt_profile: str = "standard",
    ) -> None:

        self.registry = (
            registry
        )

        self._broad_specs: dict[bool, tuple[list[dict], str]] = {}

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
        if not isinstance(router_prompt_profile, str) or router_prompt_profile.strip().lower() not in {
            "standard", "compact",
        }:
            raise ValueError("Router prompt profile must be standard or compact.")
        self.router_prompt_profile = router_prompt_profile.strip().lower()

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

        return build_router_system_prompt(
            specialists, repair_mode=repair_mode,
            include_workflow_protocol=include_workflow_protocol,
        )

    def _compact_system(self, specialists: list[dict], include_workflow_protocol: bool) -> str:
        return build_compact_router_system_prompt(
            specialists, include_workflow_protocol=include_workflow_protocol,
        )

    @staticmethod
    def _compact_policy(*, broad: bool):
        # Share the established disclosure envelopes without changing the local
        # CONTROL_ROUTING stage's adapter or retrieval restrictions.
        return resolve_stage_policy(
            InferenceStage.SEMANTIC_ROUTING_BROAD if broad
            else InferenceStage.SEMANTIC_ROUTING
        )

    def _local_compact_request(self, messages: list[dict], *, broad: bool):
        policy = self._compact_policy(broad=broad)
        return prepare_external_request(
            # Explicit limits prevent external-provider environment settings
            # from changing this local control-stage envelope.
            {"request_token_budget": policy.request_token_ceiling,
             "max_output_tokens": policy.output_token_ceiling},
            messages, self.max_new_tokens,
            default_request_budget=policy.request_token_ceiling,
        )

    def _fits_stage(self, stage: InferenceStage, system: str, user_request: str) -> bool:
        """True when the mandatory prompt plus output fits the stage and provider budgets."""
        policy = resolve_stage_policy(stage)
        limit = policy.request_token_ceiling
        if stage is InferenceStage.SEMANTIC_ROUTING_BROAD:
            limit = min(limit, selected_request_budget(selected_profile(stage=stage)))
        mandatory = estimate_messages_tokens([
            {"role": "system", "content": system},
            {"role": "user", "content": user_request},
        ])
        return mandatory + min(self.max_new_tokens, policy.output_token_ceiling) <= limit

    def _broad_compact_catalog(self, include_workflow_protocol):
        """Rebuild descriptors from the full registry, never selector content."""
        if include_workflow_protocol not in self._broad_specs:
            everything = CandidateSet(
                candidates=tuple(
                    Candidate(agent.name, tuple(agent.tools))
                    for agent in self.registry.list_agents() if agent.tools
                ),
                confidence=1.0,
            )
            specialists = build_candidate_agent_specs(everything, self.registry)
            self._broad_specs[include_workflow_protocol] = (
                specialists, self._compact_system(specialists, include_workflow_protocol)
            )
        return self._broad_specs[include_workflow_protocol]

    def _broad_selected_input(self, user_request, include_workflow_protocol):
        """Full trusted compact catalog on an explicitly selected connection."""
        specialists, system = self._broad_compact_catalog(include_workflow_protocol)
        if not self._fits_stage(InferenceStage.SEMANTIC_ROUTING_BROAD, system, user_request):
            return None
        return specialists, system, InferenceStage.SEMANTIC_ROUTING_BROAD, None

    def _local_compact_input(self, user_request, *, include_workflow_protocol, full_catalog):
        """Keep local routing bounded while retaining every trusted guard."""
        if not full_catalog:
            selection = validate_candidate_set(
                self.candidate_selector.select(user_request, self.registry), self.registry,
            )
            if selection.candidates and not selection.fallback_required:
                specialists = build_candidate_agent_specs(selection, self.registry)
                system = self._compact_system(specialists, include_workflow_protocol)
                try:
                    self._local_compact_request([
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_request},
                    ], broad=False)
                except ExternalInferenceBudgetError:
                    pass  # Widen the disclosure; never truncate mandatory input.
                else:
                    return specialists, system, InferenceStage.CONTROL_ROUTING, selection
        specialists, system = self._broad_compact_catalog(include_workflow_protocol)
        return specialists, system, InferenceStage.CONTROL_ROUTING, None

    def _routing_input(self, user_request, *, repair_mode, include_workflow_protocol, full_catalog=False):
        """Route on the selected connection; the local Hub is the last resort.

        A narrowed candidate packet is preferred. When retrieval cannot narrow
        the request, the full trusted catalog goes to the selected connection
        if its budget admits it; otherwise the local control Hub routes.
        Candidate identities never replace the full registry used by semantic
        validation, approval policy or tool execution.
        """
        candidates: CandidateSet | None = None
        if full_catalog and selected_profile(stage=InferenceStage.SEMANTIC_ROUTING):
            # A small provider budget cannot take the full catalog; the repair
            # then keeps the existing candidate path below.
            broad = self._broad_selected_input(user_request, include_workflow_protocol)
            if broad is not None:
                return broad
        if selected_profile(stage=InferenceStage.SEMANTIC_ROUTING):
            selection = validate_candidate_set(
                self.candidate_selector.select(user_request, self.registry), self.registry
            )
            fallback = selection.fallback_required or not selection.candidates
            if not fallback:
                specialists = build_candidate_agent_specs(selection, self.registry)
                system = self._compact_system(specialists, include_workflow_protocol)
                fallback = not self._fits_stage(InferenceStage.SEMANTIC_ROUTING, system, user_request)
                if not fallback:
                    candidates = selection
            logger.info(
                "routing_candidates candidates=%d capabilities=%d confidence=%.4f widening=%d fallback=%s",
                selection.candidate_count, selection.capability_count, selection.confidence,
                selection.widening_count, fallback,
            )
            if candidates is not None:
                return specialists, system, InferenceStage.SEMANTIC_ROUTING, candidates
            broad = self._broad_selected_input(user_request, include_workflow_protocol)
            if broad is not None:
                return broad

        if self.router_prompt_profile == "compact":
            return self._local_compact_input(
                user_request, include_workflow_protocol=include_workflow_protocol,
                full_catalog=full_catalog,
            )

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
        workflow_contract: RequestedWorkflowContract | None = None,
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
            options = {"workflow_contract": workflow_contract} if workflow_contract is not None else {}
            plan = await self.route(user_request, context=context, **options)
            if workflow_contract is not None:
                try:
                    validate_requested_plan(
                        workflow_contract, plan, admit_requested_workflow(workflow_contract, self.registry),
                    )
                except ValueError as exc:
                    raise RoutingContractError(str(exc)) from exc
            return plan

        except RoutingContractError as exc:
            # A requested-workflow mismatch shares the existing one-repair
            # budget. It never starts a second format-repair loop.
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

            plan = await self.route(
                    user_request,

                    context=context,

                    repair_mode=True,

                    include_workflow_protocol=False,

                    repair_error=(
                        str(
                            exc
                        )
                    ),

                    full_catalog=isinstance(
                        exc,
                        CandidateMissError,
                    ),
                    **options,
                )
            if workflow_contract is not None:
                try:
                    validate_requested_plan(
                        workflow_contract, plan, admit_requested_workflow(workflow_contract, self.registry),
                    )
                except ValueError as final_exc:
                    raise RoutingContractError(str(final_exc)) from final_exc
            return plan


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
        full_catalog: bool = False,
        workflow_contract: RequestedWorkflowContract | None = None,
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

        if workflow_contract is not None:
            owners = admit_requested_workflow(workflow_contract, self.registry)
            # Explicitly selected operations determine disclosure, not an
            # executable route. The model must still generate and validate its
            # complete plan through the same semantic contract and guards.
            candidates = validate_candidate_set(CandidateSet(
                tuple(Candidate(agent, (step.capability,))
                      for step in workflow_contract.steps
                      for agent in owners[step.capability]), confidence=1.0,
                reason="user-reviewed workflow disclosure",
            ), self.registry)
            specialists = build_candidate_agent_specs(candidates, self.registry)
            external = selected_profile(stage=InferenceStage.SEMANTIC_ROUTING) is not None
            origin_stage = InferenceStage.SEMANTIC_ROUTING if external else InferenceStage.CONTROL_ROUTING
            if external or self.router_prompt_profile == "compact":
                system_content = self._compact_system(specialists, include_workflow_protocol)
            else:
                system_content = self._build_system_prompt(
                    repair_mode=repair_mode, include_workflow_protocol=include_workflow_protocol,
                    specialists=specialists,
                )
            system_content += requested_workflow_prompt(workflow_contract)
        else:
            specialists, system_content, origin_stage, candidates = self._routing_input(
                user_request, repair_mode=repair_mode,
                include_workflow_protocol=include_workflow_protocol,
                full_catalog=full_catalog,
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

        local_compact = (
            self.router_prompt_profile == "compact"
            and origin_stage is InferenceStage.CONTROL_ROUTING
        )
        output_tokens = self.max_new_tokens
        if local_compact:
            # Budget rejection is not a RoutingContractError: another model
            # generation cannot fix oversized mandatory input. Keep its count
            # metadata, and do not spend the bounded format repair on it.
            budget = self._local_compact_request(messages, broad=candidates is None)
            messages = budget.messages
            output_tokens = budget.max_output_tokens
            logger.info(
                "local_compact_routing broad=%s input_estimate=%d request_budget=%d output_cap=%d dropped_messages=%d",
                candidates is None, budget.estimated_input_tokens, budget.request_token_budget,
                output_tokens, budget.dropped_messages,
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
                    output_tokens
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
                # Compact local prompts have no admitted training materializer;
                # private model/dataset jobs are excluded too. Do not hash shared
                # multi-GB checkpoints or claim standard-prompt model lineage
                # for these ineligible traces. Actual messages are still hashed.
                self.model_profile_resolver
                if (not local_compact and candidates is None
                    and selected_profile(stage=InferenceStage.SEMANTIC_ROUTING) is None
                    and not (execution_selection.get() or {}).get("dataset_id"))
                else None
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
                output_tokens
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

            # Broad routing discloses the whole catalog, so an empty plan there
            # means "no capability needed" (conversation) exactly as with the
            # local Hub. Only a retrieved candidate packet treats an empty plan
            # as an unroutable task.
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
                raise CandidateMissError(message)

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

            from agent.onboarding import parse_onboarding_dependency
            try:
                after_success_of = parse_onboarding_dependency(
                    delegation.get("after_success_of"), prior_delegations=validated,
                    target_intent=semantic_intent, condition=condition,
                )
            except ValueError as exc:
                self._contract_error(f"Invalid mutation continuation: {exc}")
                continue

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
                    after_success_of=after_success_of,
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

        if workflow_contract is not None:
            try:
                validate_requested_plan(workflow_contract, validated, owners)
            except ValueError as exc:
                reject_current_hub_routing_attempt(str(exc))
                raise RoutingContractError(str(exc)) from exc

        accept_current_hub_routing_attempt(
            validated_delegation_count=(
                len(
                    validated
                )
            ),
        )

        return validated
