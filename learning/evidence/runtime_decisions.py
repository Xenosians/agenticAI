from __future__ import annotations

from copy import (
    deepcopy,
)

from typing import (
    Any,
)


# ============================================================
# PRIVATE LEARNING-EVIDENCE KEYS
# ============================================================


_TASK_CONTEXT_KEY = (
    "_learning_runtime_decisions"
)

_HUB_ATTRIBUTE = (
    "_learning_runtime_decisions"
)


# ============================================================
# TASK CONTEXT
# ============================================================


def _task_context(
    task: Any,
) -> dict[
    str,
    Any,
] | None:
    """
    Return the private runtime-decision evidence mapping for one
    AgentTask.

    This data is observability only.

    It does NOT:
        - authorize a tool;
        - alter SemanticIntent;
        - alter SemanticGuard;
        - alter ToolGateway;
        - grant approval;
        - execute a provider.
    """

    context = (
        getattr(
            task,
            "context",
            None,
        )
    )

    if not isinstance(
        context,
        dict,
    ):
        return None

    existing = (
        context.get(
            _TASK_CONTEXT_KEY
        )
    )

    if isinstance(
        existing,
        dict,
    ):
        return existing

    evidence: dict[
        str,
        Any,
    ] = {}

    context[
        _TASK_CONTEXT_KEY
    ] = (
        evidence
    )

    return evidence


def extract_task_runtime_decisions(
    task: Any,
) -> dict[
    str,
    Any,
]:
    """
    Read private live decision evidence accumulated on one
    AgentTask.

    A copy is returned so downstream learning code cannot mutate
    the task's execution state.
    """

    try:

        context = (
            getattr(
                task,
                "context",
                None,
            )
        )

        if not isinstance(
            context,
            dict,
        ):
            return {}

        evidence = (
            context.get(
                _TASK_CONTEXT_KEY
            )
        )

        if not isinstance(
            evidence,
            dict,
        ):
            return {}

        return (
            deepcopy(
                evidence
            )
        )

    except Exception:

        return {}


# ============================================================
# SEMANTIC GUARD
# ============================================================


def capture_semantic_guard_decision(
    task: Any,
    decision: Any,
) -> bool:
    """
    Preserve the exact live SemanticGuard decision.

    Capture is deliberately best-effort.

    Failure to capture learning evidence must never affect trusted
    execution.
    """

    try:

        evidence = (
            _task_context(
                task
            )
        )

        if evidence is None:

            return False

        allowed = (
            getattr(
                decision,
                "allowed",
                None,
            )
        )

        decision_code = (
            getattr(
                decision,
                "decision_code",
                None,
            )
        )

        error = (
            getattr(
                decision,
                "error",
                None,
            )
        )

        if not isinstance(
            allowed,
            bool,
        ):

            return False

        if not isinstance(
            decision_code,
            str,
        ):

            return False

        decision_code = (
            decision_code.strip()
        )

        if not decision_code:

            return False

        if (
            error is not None
            and not isinstance(
                error,
                str,
            )
        ):

            error = (
                str(
                    error
                )
            )

        evidence[
            "semantic_guard_decision"
        ] = {
            "allowed":
                allowed,

            "decision_code":
                decision_code,

            "error":
                error,
        }

        return True

    except Exception:

        return False


# ============================================================
# TOOL GATEWAY
# ============================================================


def capture_gateway_decision(
    task: Any,
    gateway_result: Any,
) -> bool:
    """
    Preserve the exact machine-readable decision metadata returned
    by ToolGateway.

    The trusted provider result itself is NOT duplicated here.

    It already has the canonical trajectory location:

        TrajectoryStep.tool_result
    """

    try:

        if not isinstance(
            gateway_result,
            dict,
        ):

            return False

        evidence = (
            _task_context(
                task
            )
        )

        if evidence is None:

            return False

        decision: dict[
            str,
            Any,
        ] = {}

        # Explicit allowlist.
        #
        # Do not blindly persist arbitrary gateway payload fields.
        decision_fields = (
            "ok",
            "status",
            "decision_code",
            "tool",
            "risk",
            "approval_id",
            "error",
        )

        for field_name in (
            decision_fields
        ):

            if (
                field_name
                not in gateway_result
            ):

                continue

            decision[
                field_name
            ] = (
                deepcopy(
                    gateway_result[
                        field_name
                    ]
                )
            )

        if not decision:

            return False

        evidence[
            "gateway_decision"
        ] = (
            decision
        )

        return True

    except Exception:

        return False


# ============================================================
# HUB-LEVEL PRIVATE EVIDENCE
# ============================================================


def attach_hub_runtime_decisions(
    *,
    hub_result: Any,
    decisions_by_task_id: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> Any:
    """
    Attach task-indexed private runtime decision evidence to the
    final HubResult.

    This attribute is intentionally NOT declared on the HubResult
    dataclass.

    Therefore:

        dataclasses.asdict(hub_result)

    does not serialize it.

    This keeps the evidence out of:
        - Phoenix completion payloads;
        - public API results;
        - Main-model synthesis input.

    TrajectoryRecorder explicitly reads it before persistence.
    """

    try:

        normalized: dict[
            str,
            dict[
                str,
                Any,
            ],
        ] = {}

        for (
            task_id,
            evidence,
        ) in decisions_by_task_id.items():

            if not isinstance(
                task_id,
                str,
            ):

                continue

            normalized_task_id = (
                task_id.strip()
            )

            if not normalized_task_id:

                continue

            if not isinstance(
                evidence,
                dict,
            ):

                continue

            if not evidence:

                continue

            normalized[
                normalized_task_id
            ] = (
                deepcopy(
                    evidence
                )
            )

        if not normalized:

            return hub_result

        setattr(
            hub_result,
            _HUB_ATTRIBUTE,
            normalized,
        )

    except Exception:

        # Learning observability must never alter the authoritative
        # Hub result.
        pass

    return hub_result


def extract_hub_runtime_decisions(
    hub_result: Any,
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """
    Retrieve the private task-indexed runtime decision map from a
    HubResult.

    Missing evidence is expected for:
        - historical execution;
        - direct primary responses;
        - pre-tool failures;
        - tests that do not use live runtime decision capture.
    """

    try:

        evidence = (
            getattr(
                hub_result,
                _HUB_ATTRIBUTE,
                None,
            )
        )

        if not isinstance(
            evidence,
            dict,
        ):

            return {}

        normalized: dict[
            str,
            dict[
                str,
                Any,
            ],
        ] = {}

        for (
            task_id,
            task_evidence,
        ) in evidence.items():

            if not isinstance(
                task_id,
                str,
            ):

                continue

            if not isinstance(
                task_evidence,
                dict,
            ):

                continue

            normalized[
                task_id
            ] = (
                deepcopy(
                    task_evidence
                )
            )

        return (
            normalized
        )

    except Exception:

        return {}