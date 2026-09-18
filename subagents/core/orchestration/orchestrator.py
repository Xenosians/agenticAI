from __future__ import annotations

import math
import uuid

from dataclasses import (
    replace,
)

from learning.evidence.runtime_decisions import (
    attach_hub_runtime_decisions,
    extract_task_runtime_decisions,
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
    ResultCondition,
)


class Orchestrator:
    """
    Main -> Specialist -> Main orchestration.

    Conditional workflow path:

        trusted read
            ↓
        deterministic scalar condition
            ↓
        conditional specialist execution
            ↓
        SemanticGuard
            ↓
        ToolGateway
            ↓
        approval boundary

    Generative models do not evaluate trusted result conditions.

    Exact live SemanticGuard and ToolGateway decision evidence is
    collected from AgentTask runtime context and attached privately
    to the final HubResult.

    That evidence is observability only.

    It is NOT authorization.
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

    @staticmethod
    def _valid_condition_scalar(
        value,
    ) -> bool:

        if value is None:
            return True

        if isinstance(
            value,
            bool,
        ):

            return True

        if isinstance(
            value,
            str,
        ):

            return True

        if (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
        ):

            return True

        if isinstance(
            value,
            float,
        ):

            return (
                math.isfinite(
                    value
                )
            )

        return False

    def _evaluate_condition(
        self,
        condition: ResultCondition,
        *,
        source_results: dict[
            str,
            AgentResult,
        ],
    ) -> tuple[
        bool | None,
        str | None,
        str | None,
    ]:

        source_result = (
            source_results.get(
                condition.source_agent
            )
        )

        if source_result is None:

            return (
                None,
                "conditional_source_missing",
                (
                    "The conditional workflow source result "
                    "is unavailable."
                ),
            )

        if (
            source_result.status
            != "success"
        ):

            return (
                None,
                "conditional_source_failed",
                (
                    "The conditional workflow source did not "
                    "complete successfully."
                ),
            )

        tool_result = (
            source_result.tool_result
        )

        if not isinstance(
            tool_result,
            dict,
        ):

            return (
                None,
                "conditional_source_result_invalid",
                (
                    "The conditional workflow source returned "
                    "no valid trusted structured result."
                ),
            )

        # Provider-level unsuccessful results must never satisfy a
        # workflow condition.
        if (
            tool_result.get(
                "ok"
            )
            is not True
        ):

            return (
                None,
                "conditional_source_result_unsuccessful",
                (
                    "The conditional workflow source did not "
                    "return a successful trusted result."
                ),
            )

        if (
            condition.result_field
            not in tool_result
        ):

            return (
                None,
                "conditional_result_field_missing",
                (
                    "The trusted workflow result does not "
                    "contain the required condition field."
                ),
            )

        actual_value = (
            tool_result[
                condition.result_field
            ]
        )

        expected_value = (
            condition.equals
        )

        if not (
            self._valid_condition_scalar(
                actual_value
            )
        ):

            return (
                None,
                "conditional_result_value_invalid",
                (
                    "The trusted workflow condition field "
                    "does not contain a supported scalar value."
                ),
            )

        # Python considers:
        #
        #     True == 1
        #
        # Workflow equality deliberately does not.
        if (
            type(
                actual_value
            )
            is not type(
                expected_value
            )
        ):

            return (
                None,
                "conditional_result_type_mismatch",
                (
                    "The trusted workflow condition field has "
                    "an unexpected value type."
                ),
            )

        return (
            actual_value
            == expected_value,
            None,
            None,
        )

    async def run(
        self,
        user_request: str,
    ) -> HubResult:

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
                    status="error",

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

        results: list[
            AgentResult
        ] = []

        routes: list[
            str
        ] = []

        # Only unconditional executions may become trusted
        # conditional-workflow sources.
        source_results: dict[
            str,
            AgentResult,
        ] = {}

        # Private task-indexed live learning evidence.
        #
        # This is attached to the final HubResult only after
        # orchestration is complete.
        runtime_decisions_by_task_id: dict[
            str,
            dict,
        ] = {}

        for delegation in delegations:

            task_id = (
                str(
                    uuid.uuid4()
                )
            )

            # ====================================================
            # RESULT-AWARE PRECONDITION
            # ====================================================

            if (
                delegation.condition
                is not None
            ):

                (
                    matched,
                    condition_code,
                    condition_error,
                ) = (
                    self._evaluate_condition(
                        delegation.condition,

                        source_results=(
                            source_results
                        ),
                    )
                )

                if matched is False:

                    print(
                        "[HUB] Conditional delegation skipped "
                        f"agent='{delegation.agent_name}' "
                        f"source="
                        f"'{delegation.condition.source_agent}' "
                        f"field="
                        f"'{delegation.condition.result_field}'"
                    )

                    continue

                if matched is None:

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

                            semantic_intent=(
                                delegation.semantic_intent
                            ),

                            outcome_code=(
                                condition_code
                            ),

                            error=(
                                condition_error
                            ),
                        )
                    )

                    continue

            # Only actually executed specialist paths appear here.
            routes.append(
                delegation.agent_name
            )

            # ====================================================
            # SEMANTIC CONTRACT
            # ====================================================

            if (
                self.require_semantic_intent
                and delegation.semantic_intent
                is None
            ):

                result = (
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
                            "semantic_intent_missing"
                        ),

                        error=(
                            "A validated semantic intent contract "
                            "is required before specialist "
                            "execution."
                        ),
                    )
                )

                results.append(
                    result
                )

                if (
                    delegation.condition
                    is None
                ):

                    source_results[
                        delegation.agent_name
                    ] = (
                        result
                    )

                continue

            if (
                delegation.semantic_intent
                is not None
                and delegation
                .semantic_intent
                .clarification_required
            ):

                result = (
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

                        semantic_intent=(
                            delegation.semantic_intent
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

                results.append(
                    result
                )

                if (
                    delegation.condition
                    is None
                ):

                    source_results[
                        delegation.agent_name
                    ] = (
                        result
                    )

                continue

            # ====================================================
            # SPECIALIST EXECUTION
            # ====================================================

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
            # LIVE DECISION EVIDENCE
            #
            # Copy it BEFORE moving on to another task.
            #
            # This does not change runtime_result.
            # ----------------------------------------------------

            task_runtime_decisions = (
                extract_task_runtime_decisions(
                    task
                )
            )

            if task_runtime_decisions:

                runtime_decisions_by_task_id[
                    task_id
                ] = (
                    task_runtime_decisions
                )

            # ----------------------------------------------------
            # DURABLE SPECIALIST RESULT
            # ----------------------------------------------------

            result = (
                replace(
                    runtime_result,

                    task_instructions=(
                        task.instructions
                    ),

                    semantic_intent=(
                        task.semantic_intent
                    ),

                    execution_provenance=(
                        task.execution_provenance
                    ),
                )
            )

            results.append(
                result
            )

            if (
                delegation.condition
                is None
            ):

                source_results[
                    delegation.agent_name
                ] = (
                    result
                )

        # ========================================================
        # OVERALL RESULT
        # ========================================================

        statuses = {
            result.status

            for result
            in results
        }

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

        hub_result = (
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

        # ========================================================
        # PRIVATE LEARNING-EVIDENCE HANDOFF
        #
        # Dynamic attribute:
        #     ignored by dataclasses.asdict()
        #
        # Therefore it does not leak into Phoenix/public result
        # serialization or Main synthesis.
        # ========================================================

        hub_result = (
            attach_hub_runtime_decisions(
                hub_result=(
                    hub_result
                ),

                decisions_by_task_id=(
                    runtime_decisions_by_task_id
                ),
            )
        )

        return (
            hub_result
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