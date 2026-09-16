from __future__ import annotations

import hashlib
import json
import re

from collections import (
    Counter,
)

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Iterable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.evidence.types import (
    CorrectionEvent,
    LearningTrajectory,
)


# ============================================================
# REPORT TYPES
# ============================================================


class CorpusWarning(
    BaseModel
):
    code: str

    severity: str

    message: str


class ContaminationMatch(
    BaseModel
):
    trajectory_id: str

    eval_cases: list[
        str
    ] = Field(
        default_factory=list
    )


class CorpusAnalysisReport(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "corpus-analysis-report.v1"
        ),
        alias="schema",
    )

    generated_at: str

    trajectory_count: int

    correction_count: int

    corrected_trajectory_count: int

    orphan_correction_count: int

    unique_request_count: int

    duplicate_request_count: int

    duplicate_request_rate: float

    unique_behavior_pattern_count: int

    held_out_eval_case_count: int

    held_out_contamination_count: int

    contamination_matches: list[
        ContaminationMatch
    ] = Field(
        default_factory=list
    )

    domain_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    route_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    agent_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    tool_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    outcome_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    correction_type_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    correction_source_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    quality_review_state_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    failure_type_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    warnings: list[
        CorpusWarning
    ] = Field(
        default_factory=list
    )


# ============================================================
# REQUEST NORMALIZATION
# ============================================================


_WHITESPACE = re.compile(
    r"\s+"
)


def normalize_request(
    value: str,
) -> str:
    """
    Conservative request normalization.

    Used for:

        duplicate detection
        exact held-out contamination detection

    This deliberately does NOT perform semantic similarity.

    We only collapse trivial formatting/case differences so a
    paraphrase is not falsely classified as contamination.
    """

    return (
        _WHITESPACE
        .sub(
            " ",
            value.strip(),
        )
        .casefold()
    )


# ============================================================
# JSONL LOADERS
# ============================================================


def _load_jsonl_models(
    *,
    path: Path,
    model_type,
) -> list:

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.exists():

        return []

    if not resolved.is_file():

        raise ValueError(
            "Learning corpus path is not a file: "
            f"{resolved}"
        )

    result = []

    for (
        line_number,
        line,
    ) in enumerate(
        resolved
        .read_text(
            encoding="utf-8"
        )
        .splitlines(),
        start=1,
    ):

        if not line.strip():

            continue

        try:

            raw = (
                json.loads(
                    line
                )
            )

        except Exception as exc:

            raise ValueError(
                "Invalid JSON at "
                f"{resolved}:{line_number}: "
                f"{exc}"
            ) from exc

        try:

            item = (
                model_type
                .model_validate(
                    raw
                )
            )

        except Exception as exc:

            raise ValueError(
                "Invalid learning record at "
                f"{resolved}:{line_number}: "
                f"{exc}"
            ) from exc

        result.append(
            item
        )

    return result


def load_trajectories(
    path: Path,
) -> list[
    LearningTrajectory
]:

    trajectories = (
        _load_jsonl_models(
            path=(
                path
            ),

            model_type=(
                LearningTrajectory
            ),
        )
    )

    seen: set[
        str
    ] = set()

    for trajectory in trajectories:

        if (
            trajectory.trajectory_id
            in seen
        ):

            raise ValueError(
                "Duplicate trajectory_id "
                "in corpus: "
                f"{trajectory.trajectory_id}"
            )

        seen.add(
            trajectory.trajectory_id
        )

    return trajectories


def load_corrections(
    path: Path,
) -> list[
    CorrectionEvent
]:

    corrections = (
        _load_jsonl_models(
            path=(
                path
            ),

            model_type=(
                CorrectionEvent
            ),
        )
    )

    seen: set[
        str
    ] = set()

    for correction in corrections:

        if (
            correction.correction_id
            in seen
        ):

            raise ValueError(
                "Duplicate correction_id "
                "in corpus: "
                f"{correction.correction_id}"
            )

        seen.add(
            correction.correction_id
        )

    return corrections


# ============================================================
# HELD-OUT EVAL INDEX
# ============================================================


def load_eval_request_index(
    paths: Iterable[
        Path
    ],
) -> dict[
    str,
    list[str],
]:
    """
    Build:

        normalized request
            ->
        [suite:case_id, ...]

    Only request text and provenance are retained.

    Evaluation answers/results are irrelevant here.
    """

    index: dict[
        str,
        list[str],
    ] = {}

    for path in paths:

        resolved = (
            path
            .expanduser()
            .resolve()
        )

        if not resolved.is_file():

            raise ValueError(
                "Evaluation suite does not exist: "
                f"{resolved}"
            )

        for (
            line_number,
            line,
        ) in enumerate(
            resolved
            .read_text(
                encoding="utf-8"
            )
            .splitlines(),
            start=1,
        ):

            if not line.strip():

                continue

            try:

                raw = (
                    json.loads(
                        line
                    )
                )

            except Exception as exc:

                raise ValueError(
                    "Invalid evaluation JSON at "
                    f"{resolved}:{line_number}: "
                    f"{exc}"
                ) from exc

            user_request = (
                raw.get(
                    "user_request"
                )
            )

            case_id = (
                raw.get(
                    "case_id"
                )
            )

            suite = (
                raw.get(
                    "suite"
                )
            )

            if not isinstance(
                user_request,
                str,
            ):

                raise ValueError(
                    "Evaluation case missing "
                    "user_request at "
                    f"{resolved}:{line_number}"
                )

            if not isinstance(
                case_id,
                str,
            ):

                raise ValueError(
                    "Evaluation case missing "
                    "case_id at "
                    f"{resolved}:{line_number}"
                )

            if not isinstance(
                suite,
                str,
            ):

                raise ValueError(
                    "Evaluation case missing "
                    "suite at "
                    f"{resolved}:{line_number}"
                )

            key = (
                normalize_request(
                    user_request
                )
            )

            reference = (
                f"{suite}:{case_id}"
            )

            values = (
                index.setdefault(
                    key,
                    [],
                )
            )

            if (
                reference
                not in values
            ):

                values.append(
                    reference
                )

    return index


# ============================================================
# BEHAVIOR FINGERPRINT
# ============================================================


def _behavior_fingerprint(
    trajectory: LearningTrajectory,
) -> str:
    """
    Structural behavioral fingerprint.

    Argument VALUES are intentionally excluded.

    We care about diversity of:

        routes
        agents
        tools
        argument shapes
        outcomes

    without retaining potentially sensitive concrete values.
    """

    payload: dict[
        str,
        Any,
    ] = {
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


# ============================================================
# DOMAIN DERIVATION
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


# ============================================================
# HELPERS
# ============================================================


def _sorted_counts(
    counter: Counter,
) -> dict[
    str,
    int,
]:

    return {
        key:
            value

        for (
            key,
            value,
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


# ============================================================
# ANALYSIS
# ============================================================


def analyze_corpus(
    *,
    trajectory_path: Path,
    correction_path: Path,
    eval_paths: Iterable[
        Path
    ] = (),
) -> CorpusAnalysisReport:

    trajectories = (
        load_trajectories(
            trajectory_path
        )
    )

    corrections = (
        load_corrections(
            correction_path
        )
    )

    eval_paths = list(
        eval_paths
    )

    eval_index = (
        load_eval_request_index(
            eval_paths
        )
        if eval_paths
        else {}
    )

    # ========================================================
    # REQUEST DIVERSITY
    # ========================================================

    request_counts = Counter(
        normalize_request(
            trajectory.user_request
        )

        for trajectory
        in trajectories
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
    # STRUCTURAL DISTRIBUTIONS
    # ========================================================

    route_counts = Counter()

    agent_counts = Counter()

    tool_counts = Counter()

    outcome_counts = Counter()

    domain_counts = Counter()

    review_state_counts = Counter()

    failure_type_counts = Counter()

    behavior_fingerprints: set[
        str
    ] = set()

    for trajectory in trajectories:

        behavior_fingerprints.add(
            _behavior_fingerprint(
                trajectory
            )
        )

        for route in trajectory.routes:

            route_counts[
                route
            ] += 1

        for step in trajectory.steps:

            agent_counts[
                step.agent
            ] += 1

            domain_counts[
                _domain_from_agent(
                    step.agent
                )
            ] += 1

            if (
                step.proposed_tool
                is not None
            ):

                tool_counts[
                    step.proposed_tool
                ] += 1

            outcome_counts[
                step.outcome_code
                or "unknown"
            ] += 1

        if (
            trajectory.quality
            is None
        ):

            review_state_counts[
                "legacy_or_unknown"
            ] += 1

        else:

            review_state_counts[
                trajectory
                .quality
                .review_state
            ] += 1

            for failure_type in (
                trajectory
                .quality
                .failure_types
            ):

                failure_type_counts[
                    failure_type
                ] += 1

    # ========================================================
    # CORRECTION EVIDENCE
    # ========================================================

    trajectory_ids = {
        trajectory.trajectory_id

        for trajectory
        in trajectories
    }

    corrected_ids: set[
        str
    ] = set()

    orphan_correction_count = 0

    correction_type_counts = Counter()

    correction_source_counts = Counter()

    for correction in corrections:

        correction_type_counts[
            correction.correction_type
        ] += 1

        correction_source_counts[
            correction.source
        ] += 1

        if (
            correction.trajectory_id
            in trajectory_ids
        ):

            corrected_ids.add(
                correction.trajectory_id
            )

        else:

            orphan_correction_count += 1

    # ========================================================
    # HELD-OUT CONTAMINATION
    # ========================================================

    contamination_matches: list[
        ContaminationMatch
    ] = []

    for trajectory in trajectories:

        normalized = (
            normalize_request(
                trajectory.user_request
            )
        )

        matched_cases = (
            eval_index.get(
                normalized
            )
        )

        if not matched_cases:

            continue

        contamination_matches.append(
            ContaminationMatch(
                trajectory_id=(
                    trajectory.trajectory_id
                ),

                eval_cases=(
                    list(
                        matched_cases
                    )
                ),
            )
        )

    # ========================================================
    # WARNINGS
    # ========================================================

    warnings: list[
        CorpusWarning
    ] = []

    if not trajectories:

        warnings.append(
            CorpusWarning(
                code=(
                    "no_trajectories"
                ),

                severity=(
                    "info"
                ),

                message=(
                    "No runtime trajectories "
                    "have been captured yet."
                ),
            )
        )

    if (
        contamination_matches
    ):

        warnings.append(
            CorpusWarning(
                code=(
                    "held_out_contamination"
                ),

                severity=(
                    "critical"
                ),

                message=(
                    "Runtime corpus contains "
                    "requests matching held-out "
                    "evaluation cases. These "
                    "records must not enter a "
                    "training dataset."
                ),
            )
        )

    if (
        orphan_correction_count
        > 0
    ):

        warnings.append(
            CorpusWarning(
                code=(
                    "orphan_corrections"
                ),

                severity=(
                    "warning"
                ),

                message=(
                    f"{orphan_correction_count} "
                    "correction event(s) reference "
                    "trajectories absent from the "
                    "current corpus."
                ),
            )
        )

    if (
        trajectory_count
        >= 10
        and duplicate_request_rate
        > 0.25
    ):

        warnings.append(
            CorpusWarning(
                code=(
                    "high_request_duplication"
                ),

                severity=(
                    "warning"
                ),

                message=(
                    "More than 25% of captured "
                    "requests are normalized "
                    "duplicates."
                ),
            )
        )

    (
        dominant_domain,
        dominant_domain_share,
    ) = (
        _dominant_share(
            domain_counts
        )
    )

    if (
        sum(
            domain_counts.values()
        )
        >= 10
        and dominant_domain is not None
        and dominant_domain_share
        > 0.70
    ):

        warnings.append(
            CorpusWarning(
                code=(
                    "domain_imbalance"
                ),

                severity=(
                    "warning"
                ),

                message=(
                    f"Domain '{dominant_domain}' "
                    "represents "
                    f"{dominant_domain_share * 100:.1f}% "
                    "of specialist evidence."
                ),
            )
        )

    (
        dominant_tool,
        dominant_tool_share,
    ) = (
        _dominant_share(
            tool_counts
        )
    )

    if (
        sum(
            tool_counts.values()
        )
        >= 10
        and dominant_tool is not None
        and dominant_tool_share
        > 0.50
    ):

        warnings.append(
            CorpusWarning(
                code=(
                    "tool_imbalance"
                ),

                severity=(
                    "warning"
                ),

                message=(
                    f"Capability '{dominant_tool}' "
                    "represents "
                    f"{dominant_tool_share * 100:.1f}% "
                    "of tool-selection evidence."
                ),
            )
        )

    if (
        trajectories
        and not corrections
    ):

        warnings.append(
            CorpusWarning(
                code=(
                    "no_correction_evidence"
                ),

                severity=(
                    "info"
                ),

                message=(
                    "Trajectories exist, but no "
                    "correction evidence has been "
                    "recorded."
                ),
            )
        )

    # ========================================================
    # REPORT
    # ========================================================

    return (
        CorpusAnalysisReport(
            generated_at=(
                datetime
                .now(
                    timezone.utc
                )
                .isoformat()
            ),

            trajectory_count=(
                trajectory_count
            ),

            correction_count=(
                len(
                    corrections
                )
            ),

            corrected_trajectory_count=(
                len(
                    corrected_ids
                )
            ),

            orphan_correction_count=(
                orphan_correction_count
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

            held_out_eval_case_count=(
                sum(
                    len(
                        values
                    )

                    for values
                    in eval_index.values()
                )
            ),

            held_out_contamination_count=(
                len(
                    contamination_matches
                )
            ),

            contamination_matches=(
                contamination_matches
            ),

            domain_counts=(
                _sorted_counts(
                    domain_counts
                )
            ),

            route_counts=(
                _sorted_counts(
                    route_counts
                )
            ),

            agent_counts=(
                _sorted_counts(
                    agent_counts
                )
            ),

            tool_counts=(
                _sorted_counts(
                    tool_counts
                )
            ),

            outcome_counts=(
                _sorted_counts(
                    outcome_counts
                )
            ),

            correction_type_counts=(
                _sorted_counts(
                    correction_type_counts
                )
            ),

            correction_source_counts=(
                _sorted_counts(
                    correction_source_counts
                )
            ),

            quality_review_state_counts=(
                _sorted_counts(
                    review_state_counts
                )
            ),

            failure_type_counts=(
                _sorted_counts(
                    failure_type_counts
                )
            ),

            warnings=(
                warnings
            ),
        )
    )