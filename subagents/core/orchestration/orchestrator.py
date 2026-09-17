from __future__ import annotations

import uuid

from dataclasses import (
    replace,
)

from subagents.core.orchestration.router import (
    LLMRouter,
    RoutingContractError,
)

from subagents.core.orchestration.primary_assistant import (
    PrimaryAssistant,
)

from subagents.core.orchestration.runtime import (
    AgentRuntime,
)

from subagents.core.definitions.types import (
    AgentResult,
    AgentTask,
    HubResult,
)


class Orchestrator:
    """
    Main -> Specialist -> Main orchestration.

    Direct conversational path:

        valid Hub routing
            ↓
        valid empty delegation list
            ↓
        Primary Assistant

    Governed specialist path:

        Hub routing + semantic intent
            ↓
        structured specialist request
            ↓
        deterministic semantic preconditions
            ↓
        Specialist runtime
            ↓
        SemanticGuard
            ↓
        ToolGateway
            ↓
        Main synthesis

    Invalid routing contracts fail closed.

    A malformed governed-routing response must never silently fall
    through into ordinary Primary Assistant conversation.
    """

    def __init__(
        self,
        router: LLMRouter,
        runtime: AgentRuntime,
        primary_assistant: PrimaryAssistant,
        require_semantic_intent: bool = False,
    ) -> None:

        self.router = (
            router
        )

        self.runtime = (
            runtime
        )

        self.primary_assistant = (
            primary_assistant
        )

        self.require_semantic_intent = (
            require_semantic_intent
        )

    async def run(
        self,
        user_request: str,
    ) -> HubResult:

        # ========================================================
        # HUB ROUTING
        #
        # Important distinction:
        #
        #   valid []
        #       ordinary conversational request
        #
        #   RoutingContractError
        #       malformed / unsafe governed-routing plan
        #
        # The second case must NOT become conversational fallback.
        # ========================================================

        try:

            delegations = (
                await self.router.route(
                    user_request
                )
            )

        except RoutingContractError as exc:

            print(
                "[HUB] Routing contract rejected "
                f"error={exc}"
            )

            message = (
                "The request could not be safely interpreted "
                "into a valid governed execution plan."
            )

            return (
                HubResult(
                    status=(
                        "error"
                    ),

                    user_request=(
                        user_request
                    ),

                    routes=[],

                    results=[],

                    answer=(
                        message
                    ),

                    error=(
                        message
                    ),
                )
            )

        # ========================================================
        # PRIMARY CONVERSATIONAL PATH
        #
        # Only a VALID empty delegation list reaches this branch.
        # ========================================================

        if not delegations:

            answer = (
                await
                self.primary_assistant
                .respond(
                    user_request
                )
            )

            return (
                HubResult(
                    status=(
                        "success"
                    ),

                    user_request=(
                        user_request
                    ),

                    routes=[],

                    results=[],

                    answer=(
                        answer
                    ),
                )
            )

        # ========================================================
        # SPECIALIST EXECUTION
        # ========================================================

        results: list[
            AgentResult
        ] = []

        routes: list[
            str
        ] = []

        for delegation in delegations:

            routes.append(
                delegation.agent_name
            )

            task_id = (
                str(
                    uuid.uuid4()
                )
            )

            # ----------------------------------------------------
            # PRODUCTION SEMANTIC CONTRACT REQUIREMENT
            #
            # Production build_hub enables this.
            #
            # This remains defense-in-depth because strict Router
            # already rejects a missing contract before reaching
            # this point.
            # ----------------------------------------------------

            if (
                self.require_semantic_intent
                and delegation.semantic_intent
                is None
            ):

                results.append(
                    AgentResult(
                        task_id=(
                            task_id
                        ),

                        agent_name=(
                            delegation.agent_name
                        ),

                        status=(
                            "error"
                        ),

                        task_instructions=(
                            delegation.instructions
                        ),

                        outcome_code=(
                            "semantic_intent_missing"
                        ),

                        error=(
                            "A validated semantic intent contract "
                            "is required before specialist "
                            "execution."
                        ),
                    )
                )

                continue

            # ----------------------------------------------------
            # CLARIFICATION FAIL-CLOSED
            #
            # If the Hub itself says the operation/target/scope is
            # ambiguous:
            #
            #   - do not ask the specialist to guess
            #   - do not call ToolGateway
            #   - do not mutate anything
            # ----------------------------------------------------

            if (
                delegation.semantic_intent
                is not None
                and delegation
                .semantic_intent
                .clarification_required
            ):

                results.append(
                    AgentResult(
                        task_id=(
                            task_id
                        ),

                        agent_name=(
                            delegation.agent_name
                        ),

                        status=(
                            "error"
                        ),

                        task_instructions=(
                            delegation.instructions
                        ),

                        outcome_code=(
                            "semantic_clarification_required"
                        ),

                        error=(
                            "The request requires clarification "
                            "before a governed capability can be "
                            "executed."
                        ),
                    )
                )

                continue

            task = (
                AgentTask(
                    task_id=(
                        task_id
                    ),

                    agent_name=(
                        delegation.agent_name
                    ),

                    user_request=(
                        user_request
                    ),

                    instructions=(
                        delegation.instructions
                    ),

                    semantic_intent=(
                        delegation.semantic_intent
                    ),
                )
            )

            runtime_result = (
                await self.runtime.run(
                    task
                )
            )

            # ----------------------------------------------------
            # Preserve exact specialist input evidence.
            #
            # AgentRuntime owns provenance construction.
            #
            # Orchestrator transfers it into the durable result.
            # ----------------------------------------------------

            result = (
                replace(
                    runtime_result,

                    task_instructions=(
                        task.instructions
                    ),

                    execution_provenance=(
                        task.execution_provenance
                    ),
                )
            )

            results.append(
                result
            )

        statuses = {
            result.status

            for result
            in results
        }

        # ========================================================
        # DETERMINISTIC FAILURE / APPROVAL STATES
        #
        # Never let generative synthesis obscure:
        #
        #   semantic denial
        #   approval requirement
        #   runtime failure
        # ========================================================

        if (
            "error"
            in statuses
        ):

            overall_status = (
                "partial_error"
            )

            answer = (
                self
                ._compose_deterministic_answer(
                    results
                )
            )

        elif (
            "approval_required"
            in statuses
        ):

            overall_status = (
                "approval_required"
            )

            answer = (
                self
                ._compose_deterministic_answer(
                    results
                )
            )

        else:

            overall_status = (
                "success"
            )

            # ====================================================
            # MAIN SYNTHESIS
            # ====================================================

            try:

                answer = (
                    await
                    self.primary_assistant
                    .synthesize(
                        user_request,
                        results,
                    )
                )

            except Exception as exc:

                print(
                    "[HUB] Main synthesis failed "
                    f"error={exc!r}"
                )

                answer = (
                    self
                    ._compose_deterministic_answer(
                        results
                    )
                )

            if not answer:

                answer = (
                    self
                    ._compose_deterministic_answer(
                        results
                    )
                )

        return (
            HubResult(
                status=(
                    overall_status
                ),

                user_request=(
                    user_request
                ),

                routes=(
                    routes
                ),

                results=(
                    results
                ),

                answer=(
                    answer
                ),
            )
        )

    def _compose_deterministic_answer(
        self,
        results: list[
            AgentResult
        ],
    ) -> str:

        parts: list[
            str
        ] = []

        for result in results:

            if (
                result.status
                in {
                    "success",
                    "approval_required",
                }
                and result.answer
            ):

                parts.append(
                    result.answer
                )

            elif result.error:

                parts.append(
                    result.error
                )

            else:

                parts.append(
                    "The requested operation "
                    "could not be completed."
                )

        return (
            "\n".join(
                parts
            )
        )
