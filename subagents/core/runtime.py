from subagents.core.registry import (
    AgentRegistry,
)

from subagents.core.tool_gateway import (
    ToolGateway,
)

from subagents.core.tool_parser import (
    parse_tool_calls,
)

from subagents.core.types import (
    AgentResult,
    AgentTask,
)

from subagents.core.tool_prompt import (
    build_worker_system_prompt,
)

from subagents.llm.inference import (
    InferenceEngine,
)

from subagents.llm.scheduler import (
    InferencePriority,
)

from tools.registry import (
    format_approval_required,
    format_tool_result,
)

from tools.result_presentation_registry import (
    build_tool_presentation,
)


class AgentRuntime:
    """
    Executes specialist tasks.

    The runtime is intentionally generic.

    It does not know:
    - backend implementations
    - GPU ownership details
    - tool-specific risk rules
    - tool-specific result presentation
    - tool-specific approval wording

    Those concerns live behind their respective trusted
    boundaries.

    Flow:

        AgentTask
            ↓
        AgentDefinition
            ↓
        InferenceCoordinator
            ↓
        ModelManager / GPU Scheduler
            ↓
        Tool parser
            ↓
        ToolGateway
            ↓
        MCP / Approval
            ↓
        trusted structured tool result
            ↓
        human formatter + optional presentation builder
            ↓
        AgentResult
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        inference: InferenceEngine,
        tool_gateway: ToolGateway,
    ) -> None:
        self.agent_registry = (
            agent_registry
        )

        self.inference = (
            inference
        )

        self.tool_gateway = (
            tool_gateway
        )

    async def run(
        self,
        task: AgentTask,
    ) -> AgentResult:
        try:
            agent = (
                self.agent_registry
                .get(
                    task.agent_name
                )
            )

        except KeyError as exc:
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                error=str(
                    exc
                ),
            )

        messages = [
            {
                "role":
                    "system",

                "content":
                    build_worker_system_prompt(
                        agent
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    task.user_request,
            },
        ]

        if task.instructions:
            messages.append(
                {
                    "role":
                        "user",

                    "content": (
                        "Additional instructions:\n"
                        f"{task.instructions}"
                    ),
                }
            )

        try:
            response = (
                await
                self.inference
                .generate(
                    model_key=(
                        agent.model
                    ),
                    messages=(
                        messages
                    ),
                    max_new_tokens=256,
                    priority=(
                        InferencePriority
                        .SPECIALIST
                    ),
                )
            )

        except Exception as exc:
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                error=(
                    "Worker model failed: "
                    f"{exc}"
                ),
            )

        try:
            tool_calls = (
                parse_tool_calls(
                    response
                )
            )

        except ValueError as exc:
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                error=str(
                    exc
                ),
            )

        if (
            len(
                tool_calls
            )
            != 1
        ):
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                error=(
                    "Worker must return exactly "
                    "one tool call for this "
                    "runtime version."
                ),
            )

        tool_call = (
            tool_calls[0]
        )

        tool_name = (
            tool_call[
                "name"
            ]
        )

        arguments = (
            tool_call[
                "arguments"
            ]
        )

        try:
            gateway_result = (
                await
                self.tool_gateway
                .execute(
                    agent=(
                        agent
                    ),
                    user_input=(
                        task.user_request
                    ),
                    tool_name=(
                        tool_name
                    ),
                    arguments=(
                        arguments
                    ),
                )
            )

        except Exception as exc:
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                proposed_tool=(
                    tool_name
                ),
                proposed_arguments=(
                    arguments
                ),
                error=(
                    "Tool gateway failed: "
                    f"{exc}"
                ),
            )

        if (
            gateway_result.get(
                "status"
            )
            == "approval_required"
        ):
            approval_id = (
                gateway_result.get(
                    "approval_id"
                )
            )

            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status=(
                    "approval_required"
                ),
                proposed_tool=(
                    tool_name
                ),
                proposed_arguments=(
                    arguments
                ),
                approval_id=(
                    approval_id
                ),
                answer=(
                    format_approval_required(
                        tool_name,
                        arguments,
                        approval_id,
                    )
                ),
            )

        if not (
            gateway_result.get(
                "ok",
                False,
            )
        ):
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                proposed_tool=(
                    tool_name
                ),
                proposed_arguments=(
                    arguments
                ),
                error=(
                    gateway_result.get(
                        "error",
                        "Tool execution failed.",
                    )
                ),
            )

        tool_result = (
            gateway_result.get(
                "result"
            )
        )

        if not isinstance(
            tool_result,
            dict,
        ):
            return AgentResult(
                task_id=(
                    task.task_id
                ),
                agent_name=(
                    task.agent_name
                ),
                status="error",
                proposed_tool=(
                    tool_name
                ),
                proposed_arguments=(
                    arguments
                ),
                error=(
                    "Tool returned an invalid "
                    "structured result."
                ),
            )

        # --------------------------------------------------------
        # Preserve trusted machine data.
        #
        # Human-facing prose and UI presentation are both derived
        # views. Neither replaces the authoritative tool result.
        # --------------------------------------------------------

        presentation = (
            build_tool_presentation(
                tool_name,
                tool_result,
            )
        )

        return AgentResult(
            task_id=(
                task.task_id
            ),
            agent_name=(
                task.agent_name
            ),
            status="success",
            proposed_tool=(
                tool_name
            ),
            proposed_arguments=(
                arguments
            ),
            answer=(
                format_tool_result(
                    tool_name,
                    tool_result,
                )
            ),
            tool_result=(
                tool_result
            ),
            presentation=(
                presentation
            ),
        )