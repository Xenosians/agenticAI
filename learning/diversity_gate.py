from __future__ import annotations

import hashlib
import json

from collections import (
    Counter,
)

from pathlib import (
    Path,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.corpus_analysis import (
    load_trajectories,
    normalize_request,
)

from learning.curation import (
    CurationReport,
)

from learning.types import (
    LearningTrajectory,
)


# ============================================================
# POLICY
# ============================================================


class DiversityGatePolicy(
    BaseModel
):
    """
    Deterministic policy controlling whether a curated corpus has
    enough diversity and balance to proceed toward dataset export.

    These values are engineering policy, not learned values.
    """

    min_eligible_trajectories: int = 20

    min_unique_requests: int = 15

    min_unique_behavior_patterns: int = 10

    min_unique_domains: int = 2

    min_unique_capabilities: int = 2

    max_duplicate_request_rate: float = 0.25

    max_dominant_domain_share: float = 0.70

    max_dominant_capability_share: float = 0.70


# ============================================================
# METRICS
# ============================================================


class DiversityGateMetrics(
    BaseModel
):
    eligible_trajectory_count: int

    unique_request_count: int

    duplicate_request_count: int

    duplicate_request_rate: float

    unique_behavior_pattern_count: int

    unique_domain_count: int

    unique_capability_count: int

    dominant_domain: (
        str | None
    ) = None

    dominant_domain_share: float = 0.0

    dominant_capability: (
        str | None
    ) = None

    dominant_capability_share: float = 0.0

    domain_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    capability_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )


# ============================================================
# CHECK RESULT
# ============================================================


class DiversityGateCheck(
    BaseModel
):
    name: str

    passed: bool

    actual: (
        int
        | float
    )

    threshold: (
        int
        | float
    )

    comparator: str


class DiversityGateReport(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "diversity-gate-report.v1"
        ),
        alias="schema",
    )

    policy: DiversityGatePolicy

    metrics: DiversityGateMetrics

    checks: list[
        DiversityGateCheck
    ] = Field(
        default_factory=list
    )

    failed_checks: list[
        str
    ] = Field(
        default_factory=list
    )

    promotion_eligible: bool = False


# ============================================================
# STRUCTURAL HELPERS
# ============================================================


def _domain_from_agent(
    agent_name: str,
) -> str:

    normalized = (
        agent_name
        .strip()
        .lower()
    )

    suffix = (
        "-specialist"
    )

    if normalized.endswith(
        suffix
    ):

        normalized = (
            normalized[
                :-len(
                    suffix
                )
            ]
        )

    return (
        normalized
        or "unknown"
    )


def _behavior_fingerprint(
    trajectory: LearningTrajectory,
) -> str:
    """
    Structural behavioral fingerprint.

    Concrete argument values are intentionally excluded.

    The purpose here is behavioral diversity rather than evidence
    deduplication.
    """

    payload = {
        "routes":
            list(
                trajectory.routes
            ),

        "steps": [
            {
                "agent":
                    step.agent,

                "tool":
                    step.proposed_tool,

                "argument_keys":
                    sorted(
                        (
                            step.proposed_arguments
                            or {}
                        )
                        .keys()
                    ),

                "outcome":
                    step.outcome_code,
            }

            for step
            in trajectory.steps
        ],
    }

    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )

    return (
        hashlib
        .sha256(
            canonical.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _dominant_share(
    counter: Counter,
) -> tuple[
    str | None,
    float,
]:

    total = (
        sum(
            counter.values()
        )
    )

    if total <= 0:

        return (
            None,
            0.0,
        )

    (
        name,
        count,
    ) = (
        counter
        .most_common(
            1
        )[
            0
        ]
    )

    return (
        name,
        count / total,
    )


def _sorted_counts(
    counter: Counter,
) -> dict[
    str,
    int,
]:

    return {
        name:
            count

        for (
            name,
            count,
        ) in sorted(
            counter.items(),

            key=lambda item: (
                -item[
                    1
                ],
                item[
                    0
                ],
            ),
        )
    }


# ============================================================
# METRIC BUILDING
# ============================================================


def _eligible_trajectories(
    *,
    trajectory_path: Path,
    curation_report: CurationReport,
) -> list[
    LearningTrajectory
]:

    trajectories = (
        load_trajectories(
            trajectory_path
        )
    )

    by_id = {
        trajectory.trajectory_id:
            trajectory

        for trajectory
        in trajectories
    }

    eligible_ids = [
        item.trajectory_id

        for item
        in curation_report.eligible
    ]

    missing = [
        trajectory_id

        for trajectory_id
        in eligible_ids

        if trajectory_id
        not in by_id
    ]

    if missing:

        raise ValueError(
            "Curation report references "
            "trajectory IDs absent from the "
            "source corpus: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if (
        len(
            eligible_ids
        )
        != len(
            set(
                eligible_ids
            )
        )
    ):

        raise ValueError(
            "Curation report contains duplicate "
            "eligible trajectory IDs."
        )

    return [
        by_id[
            trajectory_id
        ]

        for trajectory_id
        in eligible_ids
    ]


def build_diversity_metrics(
    *,
    trajectory_path: Path,
    curation_report: CurationReport,
) -> DiversityGateMetrics:

    trajectories = (
        _eligible_trajectories(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                curation_report
            ),
        )
    )

    # ========================================================
    # REQUEST DIVERSITY
    # ========================================================

    request_counts = (
        Counter(
            normalize_request(
                trajectory.user_request
            )

            for trajectory
            in trajectories
        )
    )

    unique_request_count = (
        len(
            request_counts
        )
    )

    duplicate_request_count = (
        sum(
            count - 1

            for count
            in request_counts.values()

            if count > 1
        )
    )

    trajectory_count = (
        len(
            trajectories
        )
    )

    duplicate_request_rate = (
        duplicate_request_count
        / trajectory_count

        if trajectory_count
        else 0.0
    )

    # ========================================================
    # BEHAVIOR / DOMAIN / CAPABILITY DIVERSITY
    # ========================================================

    behavior_fingerprints: set[
        str
    ] = set()

    domain_counts = (
        Counter()
    )

    capability_counts = (
        Counter()
    )

    for trajectory in trajectories:

        behavior_fingerprints.add(
            _behavior_fingerprint(
                trajectory
            )
        )

        for step in trajectory.steps:

            domain_counts[
                _domain_from_agent(
                    step.agent
                )
            ] += 1

            if (
                step.proposed_tool
                is not None
            ):

                capability_counts[
                    step.proposed_tool
                ] += 1

    (
        dominant_domain,
        dominant_domain_share,
    ) = (
        _dominant_share(
            domain_counts
        )
    )

    (
        dominant_capability,
        dominant_capability_share,
    ) = (
        _dominant_share(
            capability_counts
        )
    )

    return (
        DiversityGateMetrics(
            eligible_trajectory_count=(
                trajectory_count
            ),

            unique_request_count=(
                unique_request_count
            ),

            duplicate_request_count=(
                duplicate_request_count
            ),

            duplicate_request_rate=(
                duplicate_request_rate
            ),

            unique_behavior_pattern_count=(
                len(
                    behavior_fingerprints
                )
            ),

            unique_domain_count=(
                len(
                    domain_counts
                )
            ),

            unique_capability_count=(
                len(
                    capability_counts
                )
            ),

            dominant_domain=(
                dominant_domain
            ),

            dominant_domain_share=(
                dominant_domain_share
            ),

            dominant_capability=(
                dominant_capability
            ),

            dominant_capability_share=(
                dominant_capability_share
            ),

            domain_counts=(
                _sorted_counts(
                    domain_counts
                )
            ),

            capability_counts=(
                _sorted_counts(
                    capability_counts
                )
            ),
        )
    )


# ============================================================
# CHECK HELPERS
# ============================================================


def _minimum_check(
    *,
    name: str,
    actual: int,
    threshold: int,
) -> DiversityGateCheck:

    return (
        DiversityGateCheck(
            name=(
                name
            ),

            passed=(
                actual
                >= threshold
            ),

            actual=(
                actual
            ),

            threshold=(
                threshold
            ),

            comparator=(
                ">="
            ),
        )
    )


def _maximum_check(
    *,
    name: str,
    actual: float,
    threshold: float,
) -> DiversityGateCheck:

    return (
        DiversityGateCheck(
            name=(
                name
            ),

            passed=(
                actual
                <= threshold
            ),

            actual=(
                actual
            ),

            threshold=(
                threshold
            ),

            comparator=(
                "<="
            ),
        )
    )


# ============================================================
# GATE
# ============================================================


def evaluate_diversity_gate(
    *,
    trajectory_path: Path,
    curation_report: CurationReport,
    policy: (
        DiversityGatePolicy
        | None
    ) = None,
) -> DiversityGateReport:

    if policy is None:

        policy = (
            DiversityGatePolicy()
        )

    metrics = (
        build_diversity_metrics(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                curation_report
            ),
        )
    )

    checks = [
        _minimum_check(
            name=(
                "minimum_eligible_trajectories"
            ),

            actual=(
                metrics
                .eligible_trajectory_count
            ),

            threshold=(
                policy
                .min_eligible_trajectories
            ),
        ),

        _minimum_check(
            name=(
                "minimum_unique_requests"
            ),

            actual=(
                metrics
                .unique_request_count
            ),

            threshold=(
                policy
                .min_unique_requests
            ),
        ),

        _minimum_check(
            name=(
                "minimum_unique_behavior_patterns"
            ),

            actual=(
                metrics
                .unique_behavior_pattern_count
            ),

            threshold=(
                policy
                .min_unique_behavior_patterns
            ),
        ),

        _minimum_check(
            name=(
                "minimum_unique_domains"
            ),

            actual=(
                metrics
                .unique_domain_count
            ),

            threshold=(
                policy
                .min_unique_domains
            ),
        ),

        _minimum_check(
            name=(
                "minimum_unique_capabilities"
            ),

            actual=(
                metrics
                .unique_capability_count
            ),

            threshold=(
                policy
                .min_unique_capabilities
            ),
        ),

        _maximum_check(
            name=(
                "maximum_duplicate_request_rate"
            ),

            actual=(
                metrics
                .duplicate_request_rate
            ),

            threshold=(
                policy
                .max_duplicate_request_rate
            ),
        ),

        _maximum_check(
            name=(
                "maximum_dominant_domain_share"
            ),

            actual=(
                metrics
                .dominant_domain_share
            ),

            threshold=(
                policy
                .max_dominant_domain_share
            ),
        ),

        _maximum_check(
            name=(
                "maximum_dominant_capability_share"
            ),

            actual=(
                metrics
                .dominant_capability_share
            ),

            threshold=(
                policy
                .max_dominant_capability_share
            ),
        ),
    ]

    failed_checks = [
        check.name

        for check
        in checks

        if not check.passed
    ]

    return (
        DiversityGateReport(
            policy=(
                policy
            ),

            metrics=(
                metrics
            ),

            checks=(
                checks
            ),

            failed_checks=(
                failed_checks
            ),

            promotion_eligible=(
                len(
                    failed_checks
                )
                == 0
            ),
        )
    )