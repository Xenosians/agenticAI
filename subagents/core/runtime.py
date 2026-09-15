from __future__ import annotations

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

    The specialist receives:

        1. trusted capability-aware system prompt
        2. original user request
        3. optional Hub routing/task context

    Hub instructions are advisory task context only.

    They never grant authorization and never override ToolGateway
    policy.

    Stable outcome codes are propagated into AgentResult so
    learning/evaluation infrastructure can classify runtime
    behavior without parsing human-readable errors.
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

        # ========================================================
        # AGENT RESOLUTION
        # ========================================================

        try:

            agent = (
                self.agent_registry
                .get(
                    task.agent_name
                )
            )

        except KeyError as exc:

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="error",

                    outcome_code=(
                        "agent_definition_error"
                    ),

                    error=(
                        str(
                            exc
                        )
                    ),
                )
            )

        # ========================================================
        # MODEL MESSAGE CONSTRUCTION
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

        # --------------------------------------------------------
        # HUB TASK CONTEXT
        #
        # This is intentionally separate from the original user
        # request.
        #
        # The specialist may use it as semantic task-scoping
        # context, but trusted policy still evaluates execution
        # against the original user request.
        # --------------------------------------------------------

        if (
            task.instructions
            is not None
        ):

            normalized_instructions = (
                task.instructions
                .strip()
            )

            if normalized_instructions:

                messages.append(
                    {
                        "role":
                            "user",

                        "content":
                            (
                                "Additional task context "
                                "from the routing stage:\n"
                                f"{normalized_instructions}"
                            ),
                    }
                )

        # ========================================================
        # MODEL GENERATION
        # ========================================================

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

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="error",

                    outcome_code=(
                        "model_generation_error"
                    ),

                    error=(
                        "Worker model failed: "
                        f"{exc}"
                    ),
                )
            )

        print(
            "\n===== SPECIALIST WORKER ====="
            f"\nAGENT: {agent.name}"
            f"\nUSER: {task.user_request}"
            f"\nROUTER_METADATA: {task.instructions}"
            f"\nRAW: {response}"
            "\n============================="
        )

        # ========================================================
        # TOOL-CALL PARSING
        # ========================================================

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

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="error",

                    outcome_code=(
                        "tool_parse_error"
                    ),

                    error=(
                        str(
                            exc
                        )
                    ),
                )
            )

        # ========================================================
        # CURRENT RUNTIME CONTRACT
        #
        # Exactly one tool call per specialist execution.
        # ========================================================

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

            return (
                AgentResult(
                    task_id=(
                        task.task_id
                    ),

                    agent_name=(
                        task.agent_name
                    ),

                    status="error",

                    outcome_code=(
                        "invalid_tool_call_count"
                    ),

                    error=(
                        "Worker must return exactly "
                        "one tool call for this "
                        "runtime version."
                    ),
                )
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

        # ========================================================
        # TRUSTED TOOL GATEWAY
        # ========================================================

        try:

            gateway_result = (
                await
                self.tool_gateway
                .execute(
                    agent=(
                        agent
                    ),

                    # IMPORTANT:
                    #
                    # Trusted grounding remains bound to the
                    # original user request.
                    #
                    # Router instructions are not authorization.
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

            return (
                AgentResult(
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

                    outcome_code=(
                        "tool_gateway_exception"
                    ),

                    error=(
                        "Tool gateway failed: "
                        f"{exc}"
                    ),
                )
            )

        decision_code = (
            gateway_result.get(
                "decision_code"
            )
        )

        print(
            "[WORKER] Gateway result "
            f"agent='{agent.name}' "
            f"tool='{tool_name}' "
            f"status="
            f"{gateway_result.get('status')} "
            f"decision="
            f"{decision_code} "
            f"ok="
            f"{gateway_result.get('ok')}"
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

            return (
                AgentResult(
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

                    outcome_code=(
                        decision_code
                        or "approval_required"
                    ),

                    answer=(
                        format_approval_required(
                            tool_name,
                            arguments,
                            approval_id,
                        )
                    ),
                )
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
                "[WORKER] Tool execution "
                "denied/failed "
                f"agent='{agent.name}' "
                f"tool='{tool_name}' "
                f"decision={decision_code} "
                f"error={error}"
            )

            return (
                AgentResult(
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

                    outcome_code=(
                        decision_code
                        or "tool_execution_error"
                    ),

                    error=(
                        error
                    ),
                )
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

            return (
                AgentResult(
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

                    outcome_code=(
                        "invalid_structured_result"
                    ),

                    error=(
                        "Tool returned an invalid "
                        "structured result."
                    ),
                )
            )

        # ========================================================
        # PRESENTATION
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

        # ========================================================
        # SUCCESS
        # ========================================================

        return (
            AgentResult(
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

                outcome_code=(
                    decision_code
                    or "success"
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
        )