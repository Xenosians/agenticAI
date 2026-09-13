from typing import Any

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


def format_tool_result(
    tool_name: str,
    result: dict[str, Any],
) -> str:
    """
    Convert trusted structured tool output into a
    user-facing sentence.

    Structured tool data stays internal.
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

        if (
            executable == "ls"
            and isinstance(
                stdout,
                str,
            )
        ):
            entries = [
                line.strip()

                for line
                in stdout.splitlines()

                if line.strip()
            ]

            if not entries:
                return (
                    "The current workspace is empty."
                )

            formatted_entries = (
                "\n".join(
                    f"- {entry}"

                    for entry
                    in entries
                )
            )

            return (
                "The current workspace contains:\n"
                f"{formatted_entries}"
            )

        return (
            "The approved local process "
            "completed successfully."
        )

    if tool_name == "workspace_read_text":
        path = result.get(
            "path"
        )

        content = result.get(
            "content"
        )

        truncated = result.get(
            "truncated",
            False,
        )

        if (
            isinstance(
                path,
                str,
            )
            and isinstance(
                content,
                str,
            )
        ):
            if truncated:
                return (
                    f"Contents of {path} "
                    "(truncated to the allowed "
                    "read limit):\n\n"
                    f"{content}"
                )

            return (
                f"Contents of {path}:\n\n"
                f"{content}"
            )

        return (
            "The workspace file was read successfully, "
            "but the result did not contain valid text."
        )

    if tool_name == "workspace_git_status":
        stdout = result.get(
            "stdout"
        )

        if not isinstance(
            stdout,
            str,
        ):
            return (
                "Git status completed successfully, "
                "but no readable status output was returned."
            )

        lines = [
            line

            for line
            in stdout.splitlines()

            if line.strip()
        ]

        if not lines:
            return (
                "Git status completed successfully, "
                "but no branch information was returned."
            )

        branch_line = (
            lines[0]
        )

        if branch_line.startswith(
            "## "
        ):
            branch_status = (
                branch_line[3:]
                .strip()
            )

        else:
            branch_status = (
                branch_line
                .strip()
            )

        changes = (
            lines[1:]
        )

        if not changes:
            return (
                "Git status:\n"
                f"- Branch: {branch_status}\n"
                "- Working tree: clean"
            )

        formatted_changes = (
            "\n".join(
                f"- {change}"

                for change
                in changes
            )
        )

        return (
            "Git status:\n"
            f"- Branch: {branch_status}\n"
            "- Working tree changes:\n"
            f"{formatted_changes}"
        )

    return (
        f"The {tool_name} operation "
        "completed successfully."
    )


def format_approval_required(
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
    approval_id: str | None,
) -> str:
    user_id = (
        arguments.get(
            "user_id"
        )
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
                f"Resetting {user_id}'s "
                "password requires approval."
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
    Executes specialist tasks.

    Model ownership is deliberately outside this class.

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
                await self.inference.generate(
                    model_key=(
                        agent.model
                    ),
                    messages=messages,
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

        if len(
            tool_calls
        ) != 1:
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
                    agent=agent,
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

        if not gateway_result.get(
            "ok",
            False,
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
        )