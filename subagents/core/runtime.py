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

    The Hub chooses WHICH specialist should receive a task.

    The specialist reasons about HOW to satisfy the original
    user request using its available capabilities.

    Hub-generated delegation instructions are retained as
    orchestration metadata, but are deliberately not injected
    into specialist reasoning.

    This prevents one generative model from accidentally
    introducing identifiers, interpretations, scopes, or other
    concrete values that were not present in the original
    user request.

    Trusted execution policy remains inside ToolGateway.
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

        # ========================================================
        # SPECIALIST CONTEXT
        #
        # IMPORTANT:
        #
        # The original user request is the specialist's semantic
        # source of truth.
        #
        # Hub delegation instructions are NOT inserted into the
        # model conversation because they are themselves generated
        # text and may contain invented interpretations or values.
        #
        # The Hub owns routing.
        # The specialist owns capability reasoning.
        # ========================================================

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
            print(
                "[WORKER] Model generation failed "
                f"agent='{agent.name}' "
                f"error={exc!r}"
            )

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

        print(
            "\n===== SPECIALIST WORKER ====="
            f"\nAGENT: {agent.name}"
            f"\nUSER: {task.user_request}"
            f"\nROUTER_METADATA: {task.instructions}"
            f"\nRAW: {response}"
            "\n============================="
        )

        try:
            tool_calls = (
                parse_tool_calls(
                    response
                )
            )

        except ValueError as exc:
            print(
                "[WORKER] Tool-call parse failed "
                f"agent='{agent.name}' "
                f"error={exc}"
            )

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
            print(
                "[WORKER] Invalid tool-call count "
                f"agent='{agent.name}' "
                f"count={len(tool_calls)}"
            )

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
            tool_calls[
                0
            ]
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

        print(
            "[WORKER] Proposed tool "
            f"agent='{agent.name}' "
            f"tool='{tool_name}' "
            f"arguments={arguments}"
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
            print(
                "[WORKER] Tool gateway failed "
                f"agent='{agent.name}' "
                f"tool='{tool_name}' "
                f"error={exc!r}"
            )

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

        print(
            "[WORKER] Gateway result "
            f"agent='{agent.name}' "
            f"tool='{tool_name}' "
            f"status={gateway_result.get('status')} "
            f"ok={gateway_result.get('ok')}"
        )

        # ========================================================
        # APPROVAL REQUIRED
        # ========================================================

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

        # ========================================================
        # DENIED / FAILED
        # ========================================================

        if not (
            gateway_result.get(
                "ok",
                False,
            )
        ):
            error = (
                gateway_result.get(
                    "error",
                    "Tool execution failed.",
                )
            )

            print(
                "[WORKER] Tool execution denied/failed "
                f"agent='{agent.name}' "
                f"tool='{tool_name}' "
                f"error={error}"
            )

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
                    error
                ),
            )

        # ========================================================
        # TRUSTED STRUCTURED RESULT
        # ========================================================

        tool_result = (
            gateway_result.get(
                "result"
            )
        )

        if not isinstance(
            tool_result,
            dict,
        ):
            print(
                "[WORKER] Invalid structured result "
                f"agent='{agent.name}' "
                f"tool='{tool_name}'"
            )

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

        # ========================================================
        # DERIVED PRESENTATION
        #
        # Authoritative machine data remains tool_result.
        #
        # Human prose and UI presentation are derived views.
        # ========================================================

        presentation = (
            build_tool_presentation(
                tool_name,
                tool_result,
            )
        )

        print(
            "[WORKER] Completed "
            f"agent='{agent.name}' "
            f"tool='{tool_name}'"
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