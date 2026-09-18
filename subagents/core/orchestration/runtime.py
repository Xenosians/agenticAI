from __future__ import annotations

from typing import (
    Callable,
)

from config import (
    ModelProfileSettings,
    get_settings,
)

from learning.evidence.execution_provenance import (
    build_specialist_execution_provenance,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
)

from subagents.core.tooling.parser import (
    parse_tool_calls,
)

from subagents.core.definitions.types import (
    AgentResult,
    AgentTask,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)

from subagents.llm.runtime.inference import (
    InferenceEngine,
)

from subagents.llm.runtime.scheduler import (
    InferencePriority,
)

from tools.registry import (
    format_approval_required,
    format_tool_result,
)

from tools.presentation.registry import (
    build_tool_presentation,
)


ModelProfileResolver = Callable[
    [
        str,
    ],
    ModelProfileSettings,
]


class AgentRuntime:
    """
    Executes specialist tasks.

    The specialist receives:

        1. trusted capability-aware system prompt
        2. original user request
        3. optional Hub routing/task context

    Hub instructions are advisory task context only.

    SemanticIntent is NOT fed to the specialist as authorization.

    After the specialist proposes a tool call:

        specialist output
            ↓
        SemanticGuard
            ↓
        ToolGateway

    SemanticGuard verifies semantic consistency.

    ToolGateway remains authoritative for:
        authorization
        grounding
        policy
        approval
        execution

    Runtime provenance is captured before generation whenever a
    model-profile resolver is available.

    Provenance capture remains deliberately non-fatal.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        inference: InferenceEngine,
        tool_gateway: ToolGateway,
        model_profile_resolver: (
            ModelProfileResolver
            | None
        ) = None,
        semantic_guard: (
            SemanticGuard
            | None
        ) = None,

        max_new_tokens: (
            int | None
        ) = None,
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

        self.model_profile_resolver = (
            model_profile_resolver
        )

        self.semantic_guard = (
            semantic_guard
            if semantic_guard is not None
            else SemanticGuard()
        )

        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens
            is not None
            else (
                get_settings()
                .specialist_max_new_tokens
            )
        )

        if (
            not isinstance(
                self.max_new_tokens,
                int,
            )
            or self.max_new_tokens
            < 1
        ):

            raise ValueError(
                "Specialist max_new_tokens must "
                "be a positive integer."
            )

    async def run(
        self,
        task: AgentTask,
    ) -> AgentResult:

        # A task object should never accidentally carry provenance
        # from a previous execution attempt.
        task.execution_provenance = (
            None
        )

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
        # CAPABILITY CATALOG
        # ========================================================

        capability_catalog = (
            build_agent_capability_catalog(
                agent,
                include_arguments=True,
            )
        )

        system_prompt = (
            build_worker_system_prompt(
                agent,
                capability_catalog=(
                    capability_catalog
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
                    system_prompt,
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
        # This remains separate from the original user request.
        #
        # The specialist may use it as semantic task-scoping
        # context, but trusted policy still evaluates execution
        # against the original user request.
        # --------------------------------------------------------

        normalized_instructions: (
            str
            | None
        ) = None

        if (
            task.instructions
            is not None
        ):

            candidate = (
                task.instructions
                .strip()
            )

            if candidate:

                normalized_instructions = (
                    candidate
                )

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
        # EXECUTION PROVENANCE
        # ========================================================

        if (
            self.model_profile_resolver
            is not None
        ):

            try:

                model_profile = (
                    self.model_profile_resolver(
                        agent.model
                    )
                )

                provenance = (
                    build_specialist_execution_provenance(
                        agent=(
                            agent
                        ),

                        model_profile=(
                            model_profile
                        ),

                        capability_catalog=(
                            capability_catalog
                        ),

                        messages=(
                            messages
                        ),

                        user_request=(
                            task.user_request
                        ),

                        task_instructions=(
                            normalized_instructions
                        ),

                        max_new_tokens=(
                            self.max_new_tokens
                        ),
                    )
                )

                task.execution_provenance = (
                    provenance.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )

            except Exception as exc:

                print(
                    "[LEARNING] Specialist provenance "
                    "capture failed "
                    f"agent='{agent.name}' "
                    f"model='{agent.model}' "
                    f"error={exc!r}"
                )

                task.execution_provenance = (
                    None
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

                    max_new_tokens=(
                        self.max_new_tokens
                    ),

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

                    raw_model_output=(
                        response
                    ),

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

                    raw_model_output=(
                        response
                    ),

                    proposed_tool_calls=(
                        tool_calls
                    ),

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
        # SEMANTIC GUARD
        #
        # Production Orchestrator requires SemanticIntent.
        #
        # Runtime keeps None compatible for isolated low-level
        # tests and explicit non-Hub callers.
        #
        # When intent is present, it MUST pass before ToolGateway.
        # ========================================================

        if (
            task.semantic_intent
            is not None
        ):

            try:

                semantic_decision = (
                    self.semantic_guard.evaluate(
                        agent=(
                            agent
                        ),

                        intent=(
                            task.semantic_intent
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
                    "[SEMANTIC_GUARD] Internal failure "
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
                            "semantic_guard_exception"
                        ),

                        error=(
                            "Semantic validation failed closed."
                        ),
                    )
                )

            print(
                "[SEMANTIC_GUARD] "
                f"agent='{agent.name}' "
                f"tool='{tool_name}' "
                f"decision="
                f"{semantic_decision.decision_code} "
                f"allowed={semantic_decision.allowed}"
            )

            if not (
                semantic_decision.allowed
            ):

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
                            semantic_decision
                            .decision_code
                        ),

                        error=(
                            semantic_decision.error
                            or (
                                "The proposed capability "
                                "failed semantic validation."
                            )
                        ),
                    )
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
                    # Router instructions and SemanticIntent are
                    # not authorization.
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
