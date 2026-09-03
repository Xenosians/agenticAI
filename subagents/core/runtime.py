from subagents.core.registry import AgentRegistry
from subagents.core.tool_gateway import ToolGateway
from subagents.core.tool_parser import parse_tool_calls
from subagents.core.types import AgentResult, AgentTask
from subagents.llm.registry import ModelRegistry
from subagents.core.tool_prompt import build_worker_system_prompt

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
        self.agent_registry = agent_registry
        self.model_registry = model_registry
        self.tool_gateway = tool_gateway

    async def run(
        self,
        task: AgentTask,
    ) -> AgentResult:
        # -------------------------------------------------
        # Resolve agent
        # -------------------------------------------------

        try:
            agent = self.agent_registry.get(
                task.agent_name
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
            backend = self.model_registry.get(
                agent.model
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
                "content": build_worker_system_prompt(agent),
            },
            {
                "role": "user",
                "content": task.user_request,
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
                    f"Worker model failed: {exc}"
                ),
            )

        # -------------------------------------------------
        # Parse worker tool proposal
        # -------------------------------------------------

        try:
            tool_calls = parse_tool_calls(
                response
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

        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        # -------------------------------------------------
        # Security / execution gateway
        # -------------------------------------------------

        try:
            gateway_result = (
                await self.tool_gateway.execute(
                    agent=agent,
                    user_input=task.user_request,
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
                    f"Tool gateway failed: {exc}"
                ),
            )

        # -------------------------------------------------
        # Approval required
        # -------------------------------------------------

        if (
            gateway_result.get("status")
            == "approval_required"
        ):
            approval_id = gateway_result.get(
                "approval_id"
            )

            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="approval_required",
                proposed_tool=tool_name,
                proposed_arguments=arguments,
                approval_id=approval_id,
                answer=(
                    "Approval required"
                    + (
                        f": {approval_id}"
                        if approval_id
                        else "."
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
                error=gateway_result.get(
                    "error",
                    "Tool execution failed.",
                ),
            )

        # -------------------------------------------------
        # Successful tool execution
        # -------------------------------------------------

        return AgentResult(
            task_id=task.task_id,
            agent_name=task.agent_name,
            status="success",
            proposed_tool=tool_name,
            proposed_arguments=arguments,
            answer=str(
                gateway_result.get(
                    "result"
                )
            ),
        )