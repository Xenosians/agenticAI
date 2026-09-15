from __future__ import annotations

import hashlib
import json
import uuid

from collections import (
    Counter,
    defaultdict,
)

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Iterable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.corpus_analysis import (
    ContaminationMatch,
    load_corrections,
    load_eval_request_index,
    load_trajectories,
    normalize_request,
)

from learning.types import (
    CorrectionEvent,
    LearningTrajectory,
)


# ============================================================
# TRUSTED CURATION CONSTANTS
# ============================================================


TRUSTED_CORRECTION_SOURCES = {
    "trusted_review",
    "evaluation",
}


KNOWN_OUTCOME_CODES = {
    "agent_definition_error",
    "model_generation_error",
    "tool_parse_error",
    "invalid_tool_call_count",

    "agent_tool_not_allowed",
    "unknown_tool",

    "grounding_policy_invalid",
    "grounding_field_invalid",
    "grounding_failed",

    "policy_arguments_invalid",
    "policy_evaluation_error",
    "policy_result_invalid",
    "policy_denied",

    "risk_invalid",
    "approval_decision_invalid",
    "approval_required",

    "tool_gateway_exception",
    "tool_execution_denied",
    "tool_execution_error",
    "invalid_structured_result",

    "success",
}


# ============================================================
# MACHINE-READABLE EXCLUSION REASONS
# ============================================================


HELD_OUT_CONTAMINATION = (
    "held_out_contamination"
)

TRAJECTORY_DATASET_INELIGIBLE = (
    "trajectory_dataset_ineligible"
)

MISSING_TRUSTED_OUTCOME = (
    "missing_trusted_outcome"
)

UNKNOWN_TRUSTED_OUTCOME = (
    "unknown_trusted_outcome"
)

DUPLICATE_EVIDENCE = (
    "duplicate_evidence"
)

UNTRUSTED_CORRECTION_PROVENANCE = (
    "untrusted_correction_provenance"
)

CORRECTION_DATASET_INELIGIBLE = (
    "correction_dataset_ineligible"
)


# ============================================================
# CURATION TYPES
# ============================================================


class CuratedTrajectoryReference(
    BaseModel
):
    trajectory_id: str

    evidence_fingerprint: str

    usable_correction_ids: list[
        str
    ] = Field(
        default_factory=list
    )


class ExcludedTrajectoryReference(
    BaseModel
):
    trajectory_id: str

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )

    linked_correction_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    contamination_references: list[
        str
    ] = Field(
        default_factory=list
    )

    duplicate_of_trajectory_id: (
        str | None
    ) = None


class CurationReport(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "curation-report.v1"
        ),
        alias="schema",
    )

    curation_id: str

    generated_at: str

    candidate_trajectory_count: int

    eligible_trajectory_count: int

    excluded_trajectory_count: int

    correction_count: int

    usable_correction_count: int

    orphan_correction_count: int

    deduplicated_trajectory_count: int

    held_out_contamination_count: int

    exclusion_reason_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    eligible: list[
        CuratedTrajectoryReference
    ] = Field(
        default_factory=list
    )

    excluded: list[
        ExcludedTrajectoryReference
    ] = Field(
        default_factory=list
    )

    orphan_correction_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    contamination_matches: list[
        ContaminationMatch
    ] = Field(
        default_factory=list
    )


# ============================================================
# HELPERS
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


def _canonical_json(
    value: dict,
) -> str:

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def _unique(
    values: list[
        str
    ],
) -> list[
    str
]:

    result: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

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
# EVIDENCE FINGERPRINT
# ============================================================


def evidence_fingerprint(
    trajectory: LearningTrajectory,
) -> str:
    """
    Produce a deterministic hash for behavioral evidence.

    Raw values are not exposed by the curation artifact.

    Unlike the Phase 3A diversity fingerprint, argument values are
    included here because two otherwise identical executions with
    different concrete behavior must not be silently deduplicated.
    """

    payload = {
        "user_request":
            normalize_request(
                trajectory.user_request
            ),

        "hub_status":
            trajectory.hub_status,

        "routes":
            list(
                trajectory.routes
            ),

        "steps": [
            {
                "agent":
                    step.agent,

                "status":
                    step.status,

                "outcome_code":
                    step.outcome_code,

                "proposed_tool":
                    step.proposed_tool,

                "proposed_arguments":
                    step.proposed_arguments,
            }

            for step
            in trajectory.steps
        ],

        "final_answer":
            trajectory.final_answer,
    }

    serialized = (
        _canonical_json(
            payload
        )
    )

    return (
        hashlib
        .sha256(
            serialized.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# ============================================================
# OUTCOME VALIDATION
# ============================================================


def _outcome_exclusion_reasons(
    trajectory: LearningTrajectory,
) -> list[
    str
]:

    if not trajectory.steps:

        return [
            MISSING_TRUSTED_OUTCOME
        ]

    reasons: list[
        str
    ] = []

    for step in trajectory.steps:

        outcome = (
            step.outcome_code
        )

        if (
            outcome is None
            or not outcome.strip()
        ):

            reasons.append(
                MISSING_TRUSTED_OUTCOME
            )

            continue

        if (
            outcome
            not in KNOWN_OUTCOME_CODES
        ):

            reasons.append(
                UNKNOWN_TRUSTED_OUTCOME
            )

    return (
        _unique(
            reasons
        )
    )


# ============================================================
# CORRECTION VALIDATION
# ============================================================


def _usable_corrections(
    corrections: list[
        CorrectionEvent
    ],
) -> list[
    CorrectionEvent
]:

    return [
        correction

        for correction
        in corrections

        if (
            correction.dataset_eligible
            and correction.source
            in TRUSTED_CORRECTION_SOURCES
        )
    ]


def _correction_exclusion_reasons(
    corrections: list[
        CorrectionEvent
    ],
) -> list[
    str
]:
    """
    A trajectory with correction evidence must not be treated as
    clean evidence when none of its corrections has crossed the
    trusted-review boundary.

    Once at least one trusted eligible correction exists, unusable
    raw correction evidence remains in the audit corpus but is not
    used for dataset promotion.
    """

    if not corrections:

        return []

    usable = (
        _usable_corrections(
            corrections
        )
    )

    if usable:

        return []

    reasons: list[
        str
    ] = []

    if any(
        correction.source
        not in TRUSTED_CORRECTION_SOURCES

        for correction
        in corrections
    ):

        reasons.append(
            UNTRUSTED_CORRECTION_PROVENANCE
        )

    if any(
        not correction.dataset_eligible

        for correction
        in corrections
    ):

        reasons.append(
            CORRECTION_DATASET_INELIGIBLE
        )

    return reasons


# ============================================================
# CURATION
# ============================================================


def curate_corpus(
    *,
    trajectory_path: Path,
    correction_path: Path,
    eval_paths: Iterable[
        Path
    ] = (),
) -> CurationReport:
    """
    Deterministically curate raw runtime evidence.

    Raw trajectories and corrections are never modified.

    Curation only emits references to source IDs plus deterministic
    exclusion metadata.
    """

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
    # SOURCE INDEXES
    # ========================================================

    trajectory_ids = {
        trajectory.trajectory_id

        for trajectory
        in trajectories
    }

    corrections_by_trajectory: dict[
        str,
        list[
            CorrectionEvent
        ],
    ] = (
        defaultdict(
            list
        )
    )

    orphan_correction_ids: list[
        str
    ] = []

    for correction in corrections:

        if (
            correction.trajectory_id
            not in trajectory_ids
        ):

            orphan_correction_ids.append(
                correction.correction_id
            )

            continue

        corrections_by_trajectory[
            correction.trajectory_id
        ].append(
            correction
        )

    # ========================================================
    # OUTPUT COLLECTIONS
    # ========================================================

    eligible: list[
        CuratedTrajectoryReference
    ] = []

    excluded: list[
        ExcludedTrajectoryReference
    ] = []

    contamination_matches: list[
        ContaminationMatch
    ] = []

    exclusion_counts = (
        Counter()
    )

    accepted_fingerprints: dict[
        str,
        str,
    ] = {}

    usable_correction_ids: set[
        str
    ] = set()

    # ========================================================
    # TRAJECTORY CURATION
    # ========================================================

    for trajectory in trajectories:

        reasons: list[
            str
        ] = []

        linked_corrections = (
            corrections_by_trajectory
            .get(
                trajectory.trajectory_id,
                [],
            )
        )

        contamination_references = (
            list(
                eval_index.get(
                    normalize_request(
                        trajectory.user_request
                    ),
                    [],
                )
            )
        )

        if contamination_references:

            reasons.append(
                HELD_OUT_CONTAMINATION
            )

            contamination_matches.append(
                ContaminationMatch(
                    trajectory_id=(
                        trajectory.trajectory_id
                    ),

                    eval_cases=(
                        contamination_references
                    ),
                )
            )

        # ----------------------------------------------------
        # Raw runtime evidence must be explicitly approved by
        # a trusted process before it can cross this gate.
        # ----------------------------------------------------

        if not trajectory.dataset_eligible:

            reasons.append(
                TRAJECTORY_DATASET_INELIGIBLE
            )

        # ----------------------------------------------------
        # Stable machine-readable runtime outcome required.
        # ----------------------------------------------------

        reasons.extend(
            _outcome_exclusion_reasons(
                trajectory
            )
        )

        # ----------------------------------------------------
        # Corrections are evidence, not automatic truth.
        # ----------------------------------------------------

        reasons.extend(
            _correction_exclusion_reasons(
                linked_corrections
            )
        )

        reasons = (
            _unique(
                reasons
            )
        )

        fingerprint = (
            evidence_fingerprint(
                trajectory
            )
        )

        duplicate_of = None

        # ----------------------------------------------------
        # Only otherwise-eligible records participate in the
        # accepted fingerprint set.
        #
        # A contaminated or otherwise-invalid first observation
        # must not prevent a later independently reviewed record
        # from becoming eligible.
        # ----------------------------------------------------

        if not reasons:

            duplicate_of = (
                accepted_fingerprints
                .get(
                    fingerprint
                )
            )

            if duplicate_of is not None:

                reasons.append(
                    DUPLICATE_EVIDENCE
                )

            else:

                accepted_fingerprints[
                    fingerprint
                ] = (
                    trajectory
                    .trajectory_id
                )

        # ----------------------------------------------------
        # Excluded
        # ----------------------------------------------------

        if reasons:

            for reason in reasons:

                exclusion_counts[
                    reason
                ] += 1

            excluded.append(
                ExcludedTrajectoryReference(
                    trajectory_id=(
                        trajectory
                        .trajectory_id
                    ),

                    reasons=(
                        reasons
                    ),

                    linked_correction_ids=[
                        correction
                        .correction_id

                        for correction
                        in linked_corrections
                    ],

                    contamination_references=(
                        contamination_references
                    ),

                    duplicate_of_trajectory_id=(
                        duplicate_of
                    ),
                )
            )

            continue

        # ----------------------------------------------------
        # Eligible
        # ----------------------------------------------------

        usable = (
            _usable_corrections(
                linked_corrections
            )
        )

        correction_ids = [
            correction.correction_id

            for correction
            in usable
        ]

        usable_correction_ids.update(
            correction_ids
        )

        eligible.append(
            CuratedTrajectoryReference(
                trajectory_id=(
                    trajectory.trajectory_id
                ),

                evidence_fingerprint=(
                    fingerprint
                ),

                usable_correction_ids=(
                    correction_ids
                ),
            )
        )

    # ========================================================
    # REPORT
    # ========================================================

    return (
        CurationReport(
            curation_id=(
                "curation-"
                f"{uuid.uuid4().hex}"
            ),

            generated_at=(
                _utc_now()
            ),

            candidate_trajectory_count=(
                len(
                    trajectories
                )
            ),

            eligible_trajectory_count=(
                len(
                    eligible
                )
            ),

            excluded_trajectory_count=(
                len(
                    excluded
                )
            ),

            correction_count=(
                len(
                    corrections
                )
            ),

            usable_correction_count=(
                len(
                    usable_correction_ids
                )
            ),

            orphan_correction_count=(
                len(
                    orphan_correction_ids
                )
            ),

            deduplicated_trajectory_count=(
                exclusion_counts[
                    DUPLICATE_EVIDENCE
                ]
            ),

            held_out_contamination_count=(
                len(
                    contamination_matches
                )
            ),

            exclusion_reason_counts=(
                _sorted_counts(
                    exclusion_counts
                )
            ),

            eligible=(
                eligible
            ),

            excluded=(
                excluded
            ),

            orphan_correction_ids=(
                orphan_correction_ids
            ),

            contamination_matches=(
                contamination_matches
            ),
        )
    )