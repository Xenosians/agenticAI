import uuid

from dataclasses import (
    replace,
)

from subagents.core.llm_router import (
    LLMRouter,
)

from subagents.core.primary_assistant import (
    PrimaryAssistant,
)

from subagents.core.runtime import (
    AgentRuntime,
)

from subagents.core.types import (
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

        Main routing
            ↓
        structured specialist request
            ↓
        Specialist runtime
            ↓
        trusted structured result
            ↓
        Main synthesis

    Approval and error states remain deterministic and are not
    rewritten by the Main model.
    """

    def __init__(
        self,
        router: LLMRouter,
        runtime: AgentRuntime,
        primary_assistant: PrimaryAssistant,
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

            task = (
                AgentTask(
                    task_id=(
                        str(
                            uuid.uuid4()
                        )
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
                )
            )

            runtime_result = (
                await self.runtime.run(
                    task
                )
            )

            # ---------------------------------------------
            # Preserve the exact advisory routing context
            # that was supplied to the specialist.
            #
            # Do this here rather than requiring every
            # AgentRuntime return branch to repeat it.
            # ---------------------------------------------

            result = (
                replace(
                    runtime_result,

                    task_instructions=(
                        task.instructions
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
        # approval requirement or runtime failure.
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