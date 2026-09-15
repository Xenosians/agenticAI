from __future__ import annotations

from learning.types import (
    CorrectionEvent,
    TrajectoryQuality,
    TrajectoryStep,
)


FAILURE_CATEGORY_BY_CODE = {
    "agent_definition_error":
        "agent_configuration_error",

    "model_generation_error":
        "model_generation_error",

    "tool_parse_error":
        "model_output_error",

    "invalid_tool_call_count":
        "model_output_error",

    "agent_tool_not_allowed":
        "authorization_denial",

    "unknown_tool":
        "capability_resolution_error",

    "grounding_policy_invalid":
        "policy_configuration_error",

    "grounding_field_invalid":
        "policy_configuration_error",

    "grounding_failed":
        "grounding_failure",

    "policy_arguments_invalid":
        "policy_input_error",

    "policy_evaluation_error":
        "policy_evaluation_error",

    "policy_result_invalid":
        "policy_configuration_error",

    "policy_denied":
        "policy_denial",

    "risk_invalid":
        "policy_configuration_error",

    "approval_decision_invalid":
        "policy_configuration_error",

    "tool_gateway_exception":
        "gateway_error",

    "tool_execution_denied":
        "execution_denial",

    "tool_execution_error":
        "execution_error",

    "invalid_structured_result":
        "execution_contract_error",
}


GROUNDING_PASSED_CODES = {
    "policy_arguments_invalid",
    "policy_evaluation_error",
    "policy_result_invalid",
    "policy_denied",
    "risk_invalid",
    "approval_decision_invalid",
    "approval_required",
    "tool_execution_denied",
    "tool_execution_error",
    "invalid_structured_result",
    "success",
}


POLICY_PASSED_CODES = {
    "approval_required",
    "tool_execution_denied",
    "tool_execution_error",
    "invalid_structured_result",
    "success",
}


POLICY_DENIED_CODES = {
    "agent_tool_not_allowed",
    "policy_denied",
}


EXECUTION_FAILED_CODES = {
    "tool_execution_denied",
    "tool_execution_error",
    "invalid_structured_result",
}


ARGUMENT_CORRECTION_TYPES = {
    "argument",
    "repository_scope",
    "ticket_scope",
    "workspace_scope",
    "identifier",
}


CORRECTION_FAILURE_TYPE = {
    "route":
        "routing_error",

    "tool_selection":
        "tool_selection_error",

    "argument":
        "argument_error",

    "repository_scope":
        "argument_error",

    "ticket_scope":
        "argument_error",

    "workspace_scope":
        "argument_error",

    "identifier":
        "argument_error",

    "answer":
        "answer_error",

    "other":
        "semantic_correction",
}


def _unique(
    values: list[
        str
    ],
) -> list[
    str
]:

    result: list[str] = []

    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue

        seen.add(
            value
        )

        result.append(
            value
        )

    return result


def derive_trajectory_quality(
    steps: list[
        TrajectoryStep
    ],
) -> TrajectoryQuality:
    """
    Derive only facts supported by deterministic runtime evidence.

    Semantic correctness remains unknown until another evidence
    source establishes it.
    """

    codes = [
        step.outcome_code

        for step
        in steps

        if (
            step.outcome_code
            is not None
        )
    ]

    failure_types = (
        _unique(
            [
                FAILURE_CATEGORY_BY_CODE[
                    code
                ]

                for code
                in codes

                if (
                    code
                    in FAILURE_CATEGORY_BY_CODE
                )
            ]
        )
    )

    # ============================================================
    # GROUNDING
    # ============================================================

    if (
        "grounding_failed"
        in codes
    ):
        grounding_valid = (
            False
        )

    elif any(
        code
        in GROUNDING_PASSED_CODES

        for code
        in codes
    ):
        grounding_valid = (
            True
        )

    else:
        grounding_valid = (
            None
        )

    # ============================================================
    # TRUSTED POLICY
    # ============================================================

    if any(
        code
        in POLICY_DENIED_CODES

        for code
        in codes
    ):
        gateway_policy_passed = (
            False
        )

    elif any(
        code
        in POLICY_PASSED_CODES

        for code
        in codes
    ):
        gateway_policy_passed = (
            True
        )

    else:
        gateway_policy_passed = (
            None
        )

    # ============================================================
    # EXECUTION CONTRACT
    # ============================================================

    if (
        "success"
        in codes
    ):
        tool_execution_valid = (
            True
        )

    elif any(
        code
        in EXECUTION_FAILED_CODES

        for code
        in codes
    ):
        tool_execution_valid = (
            False
        )

    else:
        tool_execution_valid = (
            None
        )

    return (
        TrajectoryQuality(
            route_correct=None,
            tool_correct=None,
            arguments_correct=None,
            answer_correct=None,
            answer_grounded=None,

            grounding_valid=(
                grounding_valid
            ),

            gateway_policy_passed=(
                gateway_policy_passed
            ),

            tool_execution_valid=(
                tool_execution_valid
            ),

            failure_types=(
                failure_types
            ),

            user_corrected=False,

            correction_type=None,

            review_state=(
                "unreviewed"
            ),

            quality_eligible=False,
        )
    )


def apply_correction_to_quality(
    quality: TrajectoryQuality,
    correction: CorrectionEvent,
) -> TrajectoryQuality:
    """
    Apply correction evidence to a derived quality state.

    The original trajectory itself is never mutated.
    """

    failure_types = list(
        quality.failure_types
    )

    correction_failure = (
        CORRECTION_FAILURE_TYPE.get(
            correction.correction_type
        )
    )

    if correction_failure:
        failure_types.append(
            correction_failure
        )

    failure_types = (
        _unique(
            failure_types
        )
    )

    route_correct = (
        quality.route_correct
    )

    tool_correct = (
        quality.tool_correct
    )

    arguments_correct = (
        quality.arguments_correct
    )

    answer_correct = (
        quality.answer_correct
    )

    if (
        correction.correction_type
        == "route"
    ):
        route_correct = False

    elif (
        correction.correction_type
        == "tool_selection"
    ):
        tool_correct = False

    elif (
        correction.correction_type
        in ARGUMENT_CORRECTION_TYPES
    ):
        arguments_correct = False

    elif (
        correction.correction_type
        == "answer"
    ):
        answer_correct = False

    review_state_by_source = {
        "explicit_user":
            "corrected",

        "trusted_review":
            "reviewed",

        "evaluation":
            "evaluated",
    }

    return (
        quality.model_copy(
            deep=True,

            update={
                "review_state":
                    review_state_by_source
                    .get(
                        correction.source,
                        "corrected",
                    ),

                "route_correct":
                    route_correct,

                "tool_correct":
                    tool_correct,

                "arguments_correct":
                    arguments_correct,

                "answer_correct":
                    answer_correct,

                "user_corrected": (
                    quality.user_corrected
                    or correction.source
                    == "explicit_user"
                ),

                "correction_type":
                    correction
                    .correction_type,

                "failure_types":
                    failure_types,

                # Promotion remains separate.
                "quality_eligible":
                    False,
            },
        )
    )