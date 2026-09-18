from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
)


@dataclass
class AgentDefinition:
    name: str

    description: str

    model: str

    tools: list[str] = field(
        default_factory=list
    )

    max_steps: int = 3

    system_prompt: str = ""


@dataclass
class SemanticIntent:
    """
    Hub-produced semantic interpretation of one specialist task.

    This describes what the Hub believes the user requested.

    It is NOT authorization.

    Trusted application code independently validates this contract
    against:
        - trusted capability metadata
        - specialist output
        - ToolGateway policy
        - grounding
        - approvals
    """

    summary: str

    effect: str

    allowed_tools: list[str] = field(
        default_factory=list
    )

    forbidden_tools: list[str] = field(
        default_factory=list
    )

    allowed_arguments: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    forbidden_arguments: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    max_tool_calls: int = 1

    clarification_required: bool = False


@dataclass(
    frozen=True
)
class ResultCondition:
    """
    Deterministic condition over one trusted specialist result.

    source_agent:
        Earlier unconditional read specialist whose trusted
        structured result is inspected.

    result_field:
        Trusted top-level result field explicitly declared by that
        capability as safe for workflow branching.

    equals:
        Exact scalar value required before the conditional
        specialist execution may proceed.

    The condition is routing/execution control only.

    It is NOT authorization.
    """

    source_agent: str

    result_field: str

    equals: (
        str
        | int
        | float
        | bool
        | None
    )


@dataclass(
    frozen=True
)
class ResultCondition:
    """
    Deterministic condition over one earlier trusted specialist
    result.

    The condition controls workflow progression only.

    It is NOT authorization.
    """

    source_agent: str

    result_field: str

    equals: (
        str
        | int
        | float
        | bool
        | None
    )


@dataclass
class SpecialistRequest:
    """
    Structured Hub -> specialist delegation.

    instructions:
        advisory specialist task context

    semantic_intent:
        machine-checkable description of what the Hub believes the
        user requested

    condition:
        optional deterministic predicate over an earlier trusted
        read result

    None of these fields independently grant authorization.

    Trusted execution remains owned by deterministic application
    code.
    """

    agent_name: str

    instructions: str

    semantic_intent: (
        SemanticIntent
        | None
    ) = None

    condition: (
        ResultCondition
        | None
    ) = None

@dataclass
class AgentTask:
    task_id: str

    agent_name: str

    user_request: str

    instructions: (
        str | None
    ) = None

    semantic_intent: (
        SemanticIntent
        | None
    ) = None

    context: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    # Runtime-produced immutable identity evidence describing the
    # exact specialist model-input environment.
    #
    # This begins as None and is populated by AgentRuntime before
    # model generation.
    #
    # The core layer stores a plain dictionary so it does not
    # depend on learning-layer Pydantic types.
    execution_provenance: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None


@dataclass
class AgentResult:
    task_id: str

    agent_name: str

    status: str

    # Exact Hub routing/task context supplied to this specialist.
    #
    # This is evidence about the model input, NOT authorization.
    #
    # It is preserved so future learning artifacts can reproduce
    # the actual specialist prompt shape instead of training only
    # against the original user request.
    task_instructions: (
        str | None
    ) = None

    # Immutable model/prompt identity evidence captured by the
    # specialist runtime.
    #
    # Optional for backward compatibility with historical results
    # and isolated tests.
    execution_provenance: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    # Supplemental worker-output evidence.
    #
    # These fields are intentionally populated only when the
    # canonical exactly-one-tool representation would otherwise
    # lose information:
    #
    #     tool_parse_error
    #         raw_model_output is preserved
    #
    #     invalid_tool_call_count
    #         raw_model_output and the complete validated call set
    #         are preserved
    #
    # Normal exactly-one-tool executions continue to use
    # proposed_tool / proposed_arguments as the canonical
    # representation and leave these supplemental fields empty.
    #
    # These values are evidence only. They are never authorization
    # and are never sent to ToolGateway for execution as a batch.
    raw_model_output: (
        str | None
    ) = None

    proposed_tool_calls: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None

    answer: (
        str | None
    ) = None

    proposed_tool: (
        str | None
    ) = None

    proposed_arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    error: (
        str | None
    ) = None

    approval_id: (
        str | None
    ) = None

    # Stable machine-readable runtime outcome.
    #
    # This is intentionally separate from human-readable error
    # messages so learning/evaluation infrastructure never needs
    # to parse prose to understand what happened.
    outcome_code: (
        str | None
    ) = None

    # Authoritative structured result returned by the trusted tool.
    tool_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    # Optional generic UI-oriented representation.
    presentation: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None


@dataclass
class HubResult:
    status: str

    user_request: str

    routes: list[str] = field(
        default_factory=list
    )

    results: list[
        AgentResult
    ] = field(
        default_factory=list
    )

    answer: (
        str | None
    ) = None

    error: (
        str | None
    ) = None
