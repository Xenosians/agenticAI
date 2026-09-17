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

from learning.curation.corpus_analysis import (
    ContaminationMatch,
    load_corrections,
    load_eval_request_index,
    load_trajectories,
    normalize_request,
)

from learning.curation.reviews import (
    ReviewDecision,
    latest_review_index,
    load_reviews,
    subject_is_approved,
)

from learning.evidence.types import (
    CorrectionEvent,
    LearningTrajectory,
    TrajectoryStep,
)


# ============================================================
# KNOWN RUNTIME OUTCOMES
# ============================================================


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

TRAJECTORY_REVIEW_NOT_APPROVED = (
    "trajectory_review_not_approved"
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

CORRECTION_REVIEW_NOT_APPROVED = (
    "correction_review_not_approved"
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

    review_count: int = 0

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


def _step_fingerprint_payload(
    step: TrajectoryStep,
) -> dict:
    """
    Build the deterministic behavior identity for one specialist
    step.

    Compatibility rule:

        historical and normal singular-tool steps retain the exact
        previous fingerprint payload shape.

    Supplemental abnormal-output evidence participates only when it
    actually exists.

    For invalid multi-call generations, the validated structured
    call set is authoritative for deduplication. Raw formatting is
    deliberately ignored in that case so whitespace differences in
    otherwise identical JSON do not create distinct evidence.

    For parse failures there is no validated structured call set,
    so the raw generation is the only faithful representation of
    the observed worker behavior.
    """

    payload = {
        "task_instructions":
            (
                None
                if (
                    step.task_instructions
                    is None
                )
                else (
                    normalize_request(
                        step.task_instructions
                    )
                )
            ),

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

    if (
        step.proposed_tool_calls
        is not None
    ):

        payload[
            "proposed_tool_calls"
        ] = (
            step.proposed_tool_calls
        )

    elif (
        step.raw_model_output
        is not None
    ):

        payload[
            "raw_model_output"
        ] = (
            step.raw_model_output
        )

    return payload


def evidence_fingerprint(
    trajectory: LearningTrajectory,
) -> str:
    """
    Deterministic hash for one observed behavior.

    Concrete argument values are included because this fingerprint
    is used for evidence deduplication rather than diversity
    analysis.

    Specialist task context is also part of the observed model
    input and therefore participates in evidence identity.

    Trivial case / whitespace differences in task context are
    normalized so formatting noise does not defeat deduplication.

    Abnormal specialist output is represented losslessly enough for
    deduplication:

        validated multi-call output
            hashes the complete structured call set

        unparsable output
            hashes the raw worker generation

    Normal singular-tool trajectory fingerprints remain compatible
    with the historical payload shape.
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
            _step_fingerprint_payload(
                step
            )

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
# REVIEW AUTHORITY
# ============================================================


def _load_review_state(
    review_path: (
        Path
        | None
    ),
) -> tuple[
    list[
        ReviewDecision
    ],
    dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ],
]:

    if review_path is None:

        return (
            [],
            {},
        )

    reviews = (
        load_reviews(
            review_path
        )
    )

    return (
        reviews,
        latest_review_index(
            reviews
        ),
    )


def _trajectory_is_approved(
    *,
    trajectory: LearningTrajectory,
    review_path: (
        Path
        | None
    ),
    review_index: dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ],
) -> bool:
    """
    When a review ledger is configured, it is authoritative.

    dataset_eligible remains only as a compatibility fallback for
    older callers/tests that do not yet provide review_path.
    """

    if review_path is not None:

        return (
            subject_is_approved(
                review_index=(
                    review_index
                ),

                subject_type=(
                    "trajectory"
                ),

                subject_id=(
                    trajectory.trajectory_id
                ),
            )
        )

    return (
        trajectory.dataset_eligible
    )


def _correction_is_approved(
    *,
    correction: CorrectionEvent,
    review_path: (
        Path
        | None
    ),
    review_index: dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ],
) -> bool:

    if review_path is not None:

        return (
            subject_is_approved(
                review_index=(
                    review_index
                ),

                subject_type=(
                    "correction"
                ),

                subject_id=(
                    correction.correction_id
                ),
            )
        )

    # Compatibility fallback only.
    return (
        correction.dataset_eligible
        and correction.source
        in {
            "trusted_review",
            "evaluation",
        }
    )


# ============================================================
# CORRECTION VALIDATION
# ============================================================


def _usable_corrections(
    *,
    corrections: list[
        CorrectionEvent
    ],
    review_path: (
        Path
        | None
    ),
    review_index: dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ],
) -> list[
    CorrectionEvent
]:

    return [
        correction

        for correction
        in corrections

        if (
            _correction_is_approved(
                correction=(
                    correction
                ),

                review_path=(
                    review_path
                ),

                review_index=(
                    review_index
                ),
            )
        )
    ]


def _correction_exclusion_reasons(
    *,
    corrections: list[
        CorrectionEvent
    ],
    review_path: (
        Path
        | None
    ),
    review_index: dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ],
) -> list[
    str
]:
    """
    Corrections are not automatically trusted evidence.

    If a trajectory has correction evidence, at least one linked
    correction must cross the trusted review boundary before the
    trajectory can be used as correction-derived learning evidence.

    This is deliberately conservative.
    """

    if not corrections:

        return []

    usable = (
        _usable_corrections(
            corrections=(
                corrections
            ),

            review_path=(
                review_path
            ),

            review_index=(
                review_index
            ),
        )
    )

    if usable:

        return []

    if review_path is not None:

        return [
            CORRECTION_REVIEW_NOT_APPROVED
        ]

    reasons: list[
        str
    ] = []

    if any(
        correction.source
        not in {
            "trusted_review",
            "evaluation",
        }

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
    review_path: (
        Path
        | None
    ) = None,
) -> CurationReport:
    """
    Deterministically curate raw runtime evidence.

    Raw trajectories, corrections, and review decisions are never
    mutated.

    If review_path is configured, trusted review decisions become
    authoritative for trajectory/correction promotion.
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

    (
        reviews,
        review_index,
    ) = (
        _load_review_state(
            review_path
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
        # Trusted promotion boundary
        # ----------------------------------------------------

        if not (
            _trajectory_is_approved(
                trajectory=(
                    trajectory
                ),

                review_path=(
                    review_path
                ),

                review_index=(
                    review_index
                ),
            )
        ):

            if review_path is not None:

                reasons.append(
                    TRAJECTORY_REVIEW_NOT_APPROVED
                )

            else:

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
        # Correction promotion boundary
        # ----------------------------------------------------

        reasons.extend(
            _correction_exclusion_reasons(
                corrections=(
                    linked_corrections
                ),

                review_path=(
                    review_path
                ),

                review_index=(
                    review_index
                ),
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
                        trajectory.trajectory_id
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
                corrections=(
                    linked_corrections
                ),

                review_path=(
                    review_path
                ),

                review_index=(
                    review_index
                ),
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

            review_count=(
                len(
                    reviews
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
