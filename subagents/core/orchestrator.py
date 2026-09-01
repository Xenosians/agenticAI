import uuid

from subagents.core.router import Router
from subagents.core.runtime import AgentRuntime
from subagents.core.types import (
    AgentResult,
    AgentTask,
    HubResult,
)


class Orchestrator:
    """
    Hub responsible for routing user requests
    to specialist worker agents.
    """

    def __init__(
        self,
        router: Router,
        runtime: AgentRuntime,
    ) -> None:
        self.router = router
        self.runtime = runtime

    async def run(
        self,
        user_request: str,
    ) -> HubResult:
        # -----------------------------------------
        # Route request
        # -----------------------------------------

        routes = self.router.route(
            user_request
        )

        if not routes:
            return HubResult(
                status="no_route",
                user_request=user_request,
                routes=[],
                answer=(
                    "No specialist agent matched "
                    "the request."
                ),
            )

        # -----------------------------------------
        # Delegate to workers
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
        # V1 response composition
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
        """
        Temporary deterministic result composition.

        Later the Qwen hub can turn these structured
        results into a natural-language response.
        """

        parts = []

        for result in results:
            if result.status == "success":
                parts.append(
                    f"[{result.agent_name}] "
                    f"{result.answer}"
                )

            elif (
                result.status
                == "approval_required"
            ):
                parts.append(
                    f"[{result.agent_name}] "
                    f"{result.answer}"
                )

            else:
                parts.append(
                    f"[{result.agent_name}] "
                    f"ERROR: {result.error}"
                )

        return "\n".join(parts)