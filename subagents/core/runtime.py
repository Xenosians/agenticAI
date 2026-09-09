from typing import Any

from subagents.core.registry import AgentRegistry
from subagents.core.tool_gateway import ToolGateway
from subagents.core.tool_parser import parse_tool_calls
from subagents.core.types import (
    AgentResult,
    AgentTask,
)
from subagents.llm.registry import ModelRegistry
from subagents.core.tool_prompt import (
    build_worker_system_prompt,
)


def format_tool_result(
    tool_name: str,
    result: dict[str, Any],
) -> str:
    """
    Convert trusted structured tool output into a
    user-facing sentence.

    Structured tool data stays internal. The frontend receives
    a normal assistant answer instead of a Python dict.
    """

    if tool_name == "account_status":
        user_id = result.get(
            "user_id",
            "The account",
        )

        enabled = result.get(
            "enabled"
        )

        locked = result.get(
            "locked"
        )

        if (
            enabled is True
            and locked is True
        ):
            return (
                f"{user_id} is enabled, "
                "but the account is currently locked."
            )

        if (
            enabled is True
            and locked is False
        ):
            return (
                f"{user_id} is enabled "
                "and is not locked."
            )

        if (
            enabled is False
            and locked is True
        ):
            return (
                f"{user_id} is disabled "
                "and is currently locked."
            )

        if (
            enabled is False
            and locked is False
        ):
            return (
                f"{user_id} is disabled "
                "and is not locked."
            )

        return (
            "I retrieved the account status for "
            f"{user_id}, but the directory did not "
            "return a complete enabled/locked state."
        )

    if tool_name == "check_access":
        user_id = result.get(
            "user_id",
            "The user",
        )

        resource = result.get(
            "resource",
            "the requested resource",
        )

        has_access = result.get(
            "has_access"
        )

        if has_access is True:
            return (
                f"{user_id} has access to "
                f"{resource}."
            )

        if has_access is False:
            return (
                f"{user_id} does not have access to "
                f"{resource}."
            )

        return (
            f"I checked {user_id}'s access to "
            f"{resource}, but the directory did not "
            "return a definitive access state."
        )

    if tool_name == "process_exec":
        executable = result.get(
            "executable"
        )

        stdout = result.get(
            "stdout"
        )

        if (
            executable == "pwd"
            and isinstance(
                stdout,
                str,
            )
        ):
            working_directory = (
                stdout.strip()
            )

            if working_directory:
                return (
                    "The current working directory is "
                    f"{working_directory}."
                )

        return (
            "The approved local process "
            "completed successfully."
        )

    #
    # New tools should get an explicit formatter rather
    # than leaking their raw internal result to the user.
    #

    return (
        f"The {tool_name} operation "
        "completed successfully."
    )


def format_approval_required(
    tool_name: str,
    arguments: dict[str, Any],
    approval_id: str | None,
) -> str:
    """
    Produce a user-facing approval message without exposing
    internal structures.
    """

    user_id = arguments.get(
        "user_id"
    )

    if tool_name == "unlock_user":
        if user_id:
            message = (
                f"Unlocking {user_id} "
                "requires approval."
            )
        else:
            message = (
                "The account unlock "
                "requires approval."
            )

    elif tool_name == "reset_password":
        if user_id:
            message = (
                f"Resetting {user_id}'s password "
                "requires approval."
            )
        else:
            message = (
                "The password reset "
                "requires approval."
            )

    else:
        message = (
            "This action requires approval."
        )

    if approval_id:
        return (
            f"{message} "
            f"Approval ID: {approval_id}."
        )

    return message


class AgentRuntime:
    """
    Executes tasks using registered specialist agents
    and their configured model backends.

    Flow:
        AgentTask
            ↓
        AgentDefinition
            ↓
        ModelRegistry
            ↓
        Worker model
            ↓
        Tool-call parser
            ↓
        ToolGateway
            ↓
        MCP / Approval
            ↓
        AgentResult
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        model_registry: ModelRegistry,
        tool_gateway: ToolGateway,
    ) -> None:
        self.agent_registry = (
            agent_registry
        )

        self.model_registry = (
            model_registry
        )

        self.tool_gateway = (
            tool_gateway
        )

    async def run(
        self,
        task: AgentTask,
    ) -> AgentResult:
        # -------------------------------------------------
        # Resolve agent
        # -------------------------------------------------

        try:
            agent = (
                self.agent_registry.get(
                    task.agent_name
                )
            )

        except KeyError as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=str(exc),
            )

        # -------------------------------------------------
        # Resolve model backend
        # -------------------------------------------------

        try:
            backend = (
                self.model_registry.get(
                    agent.model
                )
            )

        except KeyError as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=str(exc),
            )

        # -------------------------------------------------
        # Build worker prompt
        # -------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    build_worker_system_prompt(
                        agent
                    )
                ),
            },
            {
                "role": "user",
                "content": (
                    task.user_request
                ),
            },
        ]

        if task.instructions:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Additional instructions:\n"
                        f"{task.instructions}"
                    ),
                }
            )

        # -------------------------------------------------
        # Worker model inference
        # -------------------------------------------------

        try:
            response = backend.generate(
                messages
            )

        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=(
                    "Worker model failed: "
                    f"{exc}"
                ),
            )

        # -------------------------------------------------
        # Parse worker tool proposal
        # -------------------------------------------------

        try:
            tool_calls = (
                parse_tool_calls(
                    response
                )
            )

        except ValueError as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=str(exc),
            )

        # -------------------------------------------------
        # V1 restriction:
        # exactly one tool call per worker task
        # -------------------------------------------------

        if len(tool_calls) != 1:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                error=(
                    "Worker must return exactly one "
                    "tool call for this runtime version."
                ),
            )

        tool_call = tool_calls[0]

        tool_name = (
            tool_call["name"]
        )

        arguments = (
            tool_call["arguments"]
        )

        # -------------------------------------------------
        # Security / execution gateway
        # -------------------------------------------------

        try:
            gateway_result = (
                await self.tool_gateway.execute(
                    agent=agent,
                    user_input=(
                        task.user_request
                    ),
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )

        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                proposed_tool=tool_name,
                proposed_arguments=arguments,
                error=(
                    "Tool gateway failed: "
                    f"{exc}"
                ),
            )

        # -------------------------------------------------
        # Approval required
        # -------------------------------------------------

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
                task_id=task.task_id,
                agent_name=task.agent_name,
                status=(
                    "approval_required"
                ),
                proposed_tool=tool_name,
                proposed_arguments=arguments,
                approval_id=approval_id,
                answer=(
                    format_approval_required(
                        tool_name,
                        arguments,
                        approval_id,
                    )
                ),
            )

        # -------------------------------------------------
        # Gateway denied / execution failed
        # -------------------------------------------------

        if not gateway_result.get(
            "ok",
            False,
        ):
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                proposed_tool=tool_name,
                proposed_arguments=arguments,
                error=(
                    gateway_result.get(
                        "error",
                        "Tool execution failed.",
                    )
                ),
            )

        # -------------------------------------------------
        # Successful tool execution
        # -------------------------------------------------

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
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="error",
                proposed_tool=tool_name,
                proposed_arguments=arguments,
                error=(
                    "Tool returned an invalid "
                    "structured result."
                ),
            )

        return AgentResult(
            task_id=task.task_id,
            agent_name=task.agent_name,
            status="success",
            proposed_tool=tool_name,
            proposed_arguments=arguments,
            answer=format_tool_result(
                tool_name,
                tool_result,
            ),
        )