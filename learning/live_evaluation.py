from __future__ import annotations

import time

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from learning.eval_types import (
    EvaluationCase,
    EvaluationMetric,
    EvaluationResult,
    EvaluationSuiteReport,
)

from learning.evaluation import (
    evaluate_trajectory,
)

from learning.recorder import (
    TrajectoryRecorder,
)


# ============================================================
# TIME HELPERS
# ============================================================


def _utc_now(
) -> str:
    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


# ============================================================
# METRIC AGGREGATION
# ============================================================


def _metric(
    results: list[
        EvaluationResult
    ],
    field_name: str,
) -> EvaluationMetric:
    """
    Aggregate one semantic evaluation field across a suite.

    Fields with value None were not evaluated by that case and
    therefore do not contribute to the denominator.
    """

    values: list[
        bool
    ] = []

    for result in results:
        value = getattr(
            result.semantic_labels,
            field_name,
        )

        if value is None:
            continue

        values.append(
            value
        )

    checked = (
        len(
            values
        )
    )

    passed = sum(
        1

        for value
        in values

        if value
    )

    accuracy: (
        float
        | None
    ) = None

    if checked:
        accuracy = (
            passed
            / checked
        )

    return (
        EvaluationMetric(
            checked=(
                checked
            ),

            passed=(
                passed
            ),

            accuracy=(
                accuracy
            ),
        )
    )


# ============================================================
# LIVE ORCHESTRATOR EVALUATION
# ============================================================


class LiveOrchestratorEvaluationRunner:
    """
    Execute evaluation cases through the real orchestration graph:

        Main Router
            ->
        Specialist
            ->
        ToolGateway
            ->
        trusted capability

    The runner does NOT persist evaluation executions into the
    normal trajectory JSONL stream.

    Instead it uses TrajectoryRecorder.build() only to transform
    the observed HubResult into the canonical LearningTrajectory
    shape expected by the deterministic evaluator.

    Evaluation therefore remains:

        real model behavior
            +
        real trusted execution
            +
        deterministic scoring
    """

    def __init__(
        self,
        *,
        hub,
        hub_model: str,
    ) -> None:
        self.hub = (
            hub
        )

        self.hub_model = (
            hub_model
        )

        # --------------------------------------------------------
        # TEMPORARY TRAJECTORY BUILDER
        #
        # build() never writes to disk.
        #
        # enabled=False additionally guarantees record() would be
        # inert if accidentally called.
        # --------------------------------------------------------

        self.trajectory_builder = (
            TrajectoryRecorder(
                path=(
                    Path(
                        ".runtime/"
                        "learning/"
                        "eval-unused.jsonl"
                    )
                ),

                enabled=False,

                hub_model=(
                    hub_model
                ),
            )
        )

    # ============================================================
    # ONE CASE
    # ============================================================

    async def run_case(
        self,
        case: EvaluationCase,
    ) -> EvaluationResult:
        """
        Execute and evaluate one orchestrator-targeted case.
        """

        if (
            case.target
            != "orchestrator"
        ):
            raise ValueError(
                "LiveOrchestratorEvaluationRunner "
                "only accepts "
                "target='orchestrator'."
            )

        started = (
            time.perf_counter()
        )

        # --------------------------------------------------------
        # REAL MAIN -> SPECIALIST -> TOOLGATEWAY EXECUTION
        # --------------------------------------------------------

        hub_result = (
            await self.hub.run(
                case.user_request
            )
        )

        duration_seconds = (
            time.perf_counter()
            - started
        )

        # --------------------------------------------------------
        # NORMALIZE INTO LEARNING TRAJECTORY
        #
        # This is intentionally build(), not record().
        #
        # Evaluations should not contaminate production learning
        # evidence.
        # --------------------------------------------------------

        trajectory = (
            self.trajectory_builder
            .build(
                job_id=(
                    "eval-"
                    f"{case.case_id}"
                ),

                attempt=1,

                result=(
                    hub_result
                ),
            )
        )

        # --------------------------------------------------------
        # DETERMINISTIC SCORING
        # --------------------------------------------------------

        result = (
            evaluate_trajectory(
                case=(
                    case
                ),

                trajectory=(
                    trajectory
                ),
            )
        )

        return (
            result.model_copy(
                update={
                    "duration_seconds":
                        duration_seconds,
                }
            )
        )

    # ============================================================
    # SUITE
    # ============================================================

    async def run(
        self,
        cases: list[
            EvaluationCase
        ],
    ) -> EvaluationSuiteReport:
        """
        Execute one homogeneous orchestrator evaluation suite and
        aggregate deterministic semantic metrics.
        """

        if not cases:
            raise ValueError(
                "At least one orchestrator "
                "evaluation case is required."
            )

        # --------------------------------------------------------
        # SUITE CONSISTENCY
        # --------------------------------------------------------

        suites = {
            case.suite

            for case
            in cases
        }

        if (
            len(
                suites
            )
            != 1
        ):
            raise ValueError(
                "All evaluation cases in one run "
                "must belong to the same suite."
            )

        for case in cases:
            if (
                case.target
                != "orchestrator"
            ):
                raise ValueError(
                    "Mixed evaluation targets are "
                    "not allowed in one live "
                    "orchestrator run."
                )

        suite = (
            next(
                iter(
                    suites
                )
            )
        )

        # --------------------------------------------------------
        # RUN START
        # --------------------------------------------------------

        started_at = (
            _utc_now()
        )

        total_started = (
            time.perf_counter()
        )

        results: list[
            EvaluationResult
        ] = []

        # --------------------------------------------------------
        # EXECUTE CASES SEQUENTIALLY
        #
        # Sequential execution is deliberate for the first
        # baseline:
        #
        # - predictable GPU pressure
        # - easier diagnostics
        # - clearer logs
        # - no cross-case scheduler contention
        # --------------------------------------------------------

        for (
            index,
            case,
        ) in enumerate(
            cases,
            start=1,
        ):
            print(
                "\n"
                "[EVAL] "
                f"{index}/"
                f"{len(cases)} "
                f"{case.case_id}"
            )

            print(
                "[EVAL] Request: "
                f"{case.user_request}"
            )

            result = (
                await self.run_case(
                    case
                )
            )

            results.append(
                result
            )

            state = (
                "PASS"
                if result.passed
                else "FAIL"
            )

            duration = (
                result.duration_seconds
                or 0.0
            )

            print(
                "[EVAL] "
                f"{state} "
                f"score="
                f"{result.score:.3f} "
                f"time="
                f"{duration:.2f}s"
            )

            # ----------------------------------------------------
            # PRINT ONLY FAILED CHECKS
            # ----------------------------------------------------

            if not result.passed:
                for check in (
                    result.checks
                ):
                    if check.passed:
                        continue

                    print(
                        "[EVAL]   "
                        f"{check.name}: "
                        f"expected="
                        f"{check.expected!r} "
                        f"observed="
                        f"{check.observed!r}"
                    )

        # --------------------------------------------------------
        # SUITE FINISH
        # --------------------------------------------------------

        total_duration_seconds = (
            time.perf_counter()
            - total_started
        )

        completed_at = (
            _utc_now()
        )

        case_count = (
            len(
                results
            )
        )

        passed_cases = sum(
            1

            for result
            in results

            if result.passed
        )

        failed_cases = (
            case_count
            - passed_cases
        )

        pass_rate = (
            passed_cases
            / case_count
        )

        mean_score = (
            sum(
                result.score

                for result
                in results
            )
            / case_count
        )

        # --------------------------------------------------------
        # SEMANTIC METRICS
        # --------------------------------------------------------

        route_metric = (
            _metric(
                results,
                "route_correct",
            )
        )

        tool_metric = (
            _metric(
                results,
                "tool_correct",
            )
        )

        arguments_metric = (
            _metric(
                results,
                "arguments_correct",
            )
        )

        outcome_metric = (
            _metric(
                results,
                "outcome_correct",
            )
        )

        # --------------------------------------------------------
        # REPORT
        # --------------------------------------------------------

        return (
            EvaluationSuiteReport(
                suite=(
                    suite
                ),

                target=(
                    "orchestrator"
                ),

                model_key=(
                    self.hub_model
                ),

                started_at=(
                    started_at
                ),

                completed_at=(
                    completed_at
                ),

                case_count=(
                    case_count
                ),

                passed_cases=(
                    passed_cases
                ),

                failed_cases=(
                    failed_cases
                ),

                pass_rate=(
                    pass_rate
                ),

                mean_score=(
                    mean_score
                ),

                total_duration_seconds=(
                    total_duration_seconds
                ),

                route_metric=(
                    route_metric
                ),

                tool_metric=(
                    tool_metric
                ),

                arguments_metric=(
                    arguments_metric
                ),

                outcome_metric=(
                    outcome_metric
                ),

                results=(
                    results
                ),
            )
        )