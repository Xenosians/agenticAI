from __future__ import annotations

from learning.eval_types import (
    EvaluationCase,
    EvaluationCheck,
    EvaluationResult,
    EvaluationSemanticLabels,
)

from learning.types import (
    LearningTrajectory,
    TrajectoryStep,
)


def _matching_step(
    trajectory: LearningTrajectory,
    case: EvaluationCase,
) -> (
    TrajectoryStep
    | None
):
    """
    Resolve the specialist step relevant to one eval case.

    For future multi-specialist trajectories, expected.agent
    disambiguates the target step.
    """

    expected_agent = (
        case.expected.agent
    )

    if expected_agent is not None:
        matches = [
            step

            for step
            in trajectory.steps

            if (
                step.agent
                == expected_agent
            )
        ]

        if len(matches) == 1:
            return (
                matches[
                    0
                ]
            )

        return None

    if len(
        trajectory.steps
    ) == 1:
        return (
            trajectory.steps[
                0
            ]
        )

    return None


def _check(
    *,
    name: str,
    expected,
    observed,
) -> EvaluationCheck:

    return (
        EvaluationCheck(
            name=(
                name
            ),

            passed=(
                observed
                == expected
            ),

            expected=(
                expected
            ),

            observed=(
                observed
            ),
        )
    )


def evaluate_trajectory(
    *,
    case: EvaluationCase,
    trajectory: LearningTrajectory,
) -> EvaluationResult:
    """
    Score one observed trajectory against deterministic expected
    behavior.

    No generative judge is used.

    Only explicitly declared expectations contribute to score.
    """

    checks: list[
        EvaluationCheck
    ] = []

    expected = (
        case.expected
    )

    step = (
        _matching_step(
            trajectory,
            case,
        )
    )

    # ============================================================
    # HUB STATUS
    # ============================================================

    if (
        expected.hub_status
        is not None
    ):
        checks.append(
            _check(
                name=(
                    "hub_status"
                ),

                expected=(
                    expected.hub_status
                ),

                observed=(
                    trajectory.hub_status
                ),
            )
        )

    # ============================================================
    # ROUTING
    # ============================================================

    route_correct = None

    if (
        expected.routes
        is not None
    ):
        route_check = (
            _check(
                name="routes",

                expected=(
                    expected.routes
                ),

                observed=(
                    trajectory.routes
                ),
            )
        )

        checks.append(
            route_check
        )

        route_correct = (
            route_check.passed
        )

    # ============================================================
    # AGENT
    # ============================================================

    if (
        expected.agent
        is not None
    ):
        observed_agent = (
            step.agent
            if step is not None
            else None
        )

        checks.append(
            _check(
                name="agent",

                expected=(
                    expected.agent
                ),

                observed=(
                    observed_agent
                ),
            )
        )

    # ============================================================
    # TOOL
    # ============================================================

    tool_correct = None

    if (
        expected.tool
        is not None
    ):
        observed_tool = (
            step.proposed_tool
            if step is not None
            else None
        )

        tool_check = (
            _check(
                name="tool",

                expected=(
                    expected.tool
                ),

                observed=(
                    observed_tool
                ),
            )
        )

        checks.append(
            tool_check
        )

        tool_correct = (
            tool_check.passed
        )

    # ============================================================
    # ARGUMENTS
    #
    # Exact equality is intentional.
    #
    # Unexpected model-generated arguments may alter execution
    # scope and therefore should not silently pass evaluation.
    # ============================================================

    arguments_correct = None

    if (
        expected.arguments
        is not None
    ):
        observed_arguments = (
            step.proposed_arguments
            if step is not None
            else None
        )

        argument_check = (
            _check(
                name="arguments",

                expected=(
                    expected.arguments
                ),

                observed=(
                    observed_arguments
                ),
            )
        )

        checks.append(
            argument_check
        )

        arguments_correct = (
            argument_check.passed
        )

    # ============================================================
    # OUTCOME
    # ============================================================

    outcome_correct = None

    if (
        expected.outcome_code
        is not None
    ):
        observed_outcome = (
            step.outcome_code
            if step is not None
            else None
        )

        outcome_check = (
            _check(
                name="outcome_code",

                expected=(
                    expected.outcome_code
                ),

                observed=(
                    observed_outcome
                ),
            )
        )

        checks.append(
            outcome_check
        )

        outcome_correct = (
            outcome_check.passed
        )

    # ============================================================
    # SCORE
    # ============================================================

    total_checks = (
        len(
            checks
        )
    )

    passed_checks = sum(
        1

        for check
        in checks

        if (
            check.passed
        )
    )

    if total_checks:
        score = (
            passed_checks
            / total_checks
        )

    else:
        score = 0.0

    return (
        EvaluationResult(
            case_id=(
                case.case_id
            ),

            suite=(
                case.suite
            ),

            target=(
                case.target
            ),

            passed=(
                total_checks > 0
                and passed_checks
                == total_checks
            ),

            score=(
                score
            ),

            passed_checks=(
                passed_checks
            ),

            total_checks=(
                total_checks
            ),

            checks=(
                checks
            ),

            semantic_labels=(
                EvaluationSemanticLabels(
                    route_correct=(
                        route_correct
                    ),

                    tool_correct=(
                        tool_correct
                    ),

                    arguments_correct=(
                        arguments_correct
                    ),

                    outcome_correct=(
                        outcome_correct
                    ),
                )
            ),
        )
    )