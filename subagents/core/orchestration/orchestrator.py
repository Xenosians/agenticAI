from __future__ import annotations

import uuid

from dataclasses import (
    replace,
)

from subagents.core.orchestration.router import (
    LLMRouter,
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

    Direct path:

        Main routing
            ↓
        no delegation
            ↓
        Main direct response

    Specialist path:

        Main routing + semantic intent
            ↓
        structured specialist request
            ↓
        deterministic semantic preconditions
            ↓
        Specialist runtime
            ↓
        semantic guard
            ↓
        trusted ToolGateway
            ↓
        Main synthesis

    Approval, semantic-denial, and error states remain
    deterministic and are not rewritten by the Main model.
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

        # -------------------------------------------------
        # Main routing / structured delegation
        # -------------------------------------------------

        delegations = (
            await self.router.route(
                user_request
            )
        )

        # -------------------------------------------------
        # Primary conversational path
        # -------------------------------------------------

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
                    status="success",

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

        # -------------------------------------------------
        # Specialist execution
        # -------------------------------------------------

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

            # ---------------------------------------------
            # PRODUCTION SEMANTIC CONTRACT REQUIREMENT
            #
            # build_hub enables this.
            #
            # Legacy isolated tests can leave it disabled.
            # ---------------------------------------------

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

                        status="error",

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

            # ---------------------------------------------
            # CLARIFICATION FAIL-CLOSED
            #
            # Do not ask a specialist to guess and do not invoke
            # ToolGateway when the Hub itself says the target or
            # operation is ambiguous.
            # ---------------------------------------------

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

                        status="error",

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

            # ---------------------------------------------
            # Preserve exact specialist input evidence.
            #
            # AgentRuntime owns provenance construction.
            #
            # Orchestrator only transfers it from the task
            # execution context into the durable result.
            # ---------------------------------------------

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

        # -------------------------------------------------
        # Fail / approval states remain deterministic.
        #
        # Do not let a generative synthesis step obscure an
        # approval requirement, semantic denial, or runtime
        # failure.
        # -------------------------------------------------

        if "error" in statuses:

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

            # ---------------------------------------------
            # Main synthesis pass
            # ---------------------------------------------

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
