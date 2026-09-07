import uuid

from subagents.core.primary_assistant import (
    PrimaryAssistant,
)
from subagents.core.router import Router
from subagents.core.runtime import AgentRuntime
from subagents.core.types import (
    AgentResult,
    AgentTask,
    HubResult,
)


class Orchestrator:
    """
    Primary assistant and specialist orchestrator.

    Ordinary requests are answered directly by the primary
    assistant.

    Requests that match specialist capabilities are delegated
    through the specialist runtime.

    Specialist implementation details are not exposed in the
    final user-facing response.
    """

    def __init__(
        self,
        router: Router,
        runtime: AgentRuntime,
        primary_assistant: PrimaryAssistant,
    ) -> None:
        self.router = router
        self.runtime = runtime
        self.primary_assistant = primary_assistant

    async def run(
        self,
        user_request: str,
    ) -> HubResult:
        # -----------------------------------------
        # Determine whether specialist delegation
        # is useful.
        # -----------------------------------------

        routes = self.router.route(
            user_request
        )

        # -----------------------------------------
        # Primary conversational path
        # -----------------------------------------

        if not routes:
            answer = (
                self.primary_assistant.respond(
                    user_request
                )
            )

            return HubResult(
                status="success",
                user_request=user_request,
                routes=[],
                results=[],
                answer=answer,
            )

        # -----------------------------------------
        # Delegate to specialists
        # -----------------------------------------

        results: list[AgentResult] = []

        for agent_name in routes:
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_name=agent_name,
                user_request=user_request,
            )

            result = await self.runtime.run(
                task
            )

            results.append(result)

        # -----------------------------------------
        # Determine overall status
        # -----------------------------------------

        statuses = {
            result.status
            for result in results
        }

        if "error" in statuses:
            overall_status = "partial_error"

        elif "approval_required" in statuses:
            overall_status = "approval_required"

        else:
            overall_status = "success"

        # -----------------------------------------
        # Compose user-facing answer.
        #
        # Do not expose internal specialist names or
        # raw tool structures.
        # -----------------------------------------

        answer = self._compose_answer(
            results
        )

        return HubResult(
            status=overall_status,
            user_request=user_request,
            routes=routes,
            results=results,
            answer=answer,
        )

    def _compose_answer(
        self,
        results: list[AgentResult],
    ) -> str:
        parts: list[str] = []

        for result in results:
            if (
                result.status == "success"
                and result.answer
            ):
                parts.append(
                    result.answer
                )

            elif (
                result.status
                == "approval_required"
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
                    "The requested operation could not "
                    "be completed."
                )

        return "\n".join(parts)