from __future__ import annotations

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================
# RAW TRAJECTORY TYPES
# ============================================================


class TrajectoryStep(
    BaseModel
):
    task_id: str

    agent: str
    status: str

    # Stable machine-readable runtime outcome.
    #
    # Examples:
    #
    # success
    # grounding_failed
    # policy_denied
    # tool_execution_error
    # approval_required
    outcome_code: (
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

    tool_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    approval_id: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class TrajectorySignals(
    BaseModel
):
    delegated: bool

    route_count: int = 0

    specialist_count: int = 0

    specialist_success_count: int = 0

    specialist_error_count: int = 0

    approval_required_count: int = 0

    tool_proposed_count: int = 0

    tool_success_count: int = 0

    overall_success: bool = False

    had_error: bool = False

    waiting_approval: bool = False


# ============================================================
# EXECUTION REWARD
# ============================================================


class ExecutionReward(
    BaseModel
):
    """
    Deterministic runtime reward.

    This measures operational execution evidence only.

    It must never automatically be interpreted as semantic
    correctness.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "execution-reward.v1"
        ),

        alias="schema",
    )

    components: dict[
        str,
        float,
    ] = Field(
        default_factory=dict
    )

    total: float = 0.0

    # Execution success does not automatically make an example
    # trustworthy training data.
    quality_eligible: bool = False


# ============================================================
# TRAJECTORY QUALITY
# ============================================================


class TrajectoryQuality(
    BaseModel
):
    """
    Semantic and deterministic quality evidence associated with a
    trajectory.

    Semantic fields remain None until trusted evidence establishes
    whether they are correct.

    This distinction is intentional:

        runtime success
            !=
        semantic correctness
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "trajectory-quality.v1"
        ),

        alias="schema",
    )

    review_state: str = (
        "unreviewed"
    )

    # ========================================================
    # SEMANTIC CORRECTNESS
    #
    # None means:
    #
    #     "we do not yet have trustworthy evidence"
    #
    # rather than assuming True or False.
    # ========================================================

    route_correct: (
        bool | None
    ) = None

    tool_correct: (
        bool | None
    ) = None

    arguments_correct: (
        bool | None
    ) = None

    answer_correct: (
        bool | None
    ) = None

    answer_grounded: (
        bool | None
    ) = None

    # ========================================================
    # DETERMINISTIC RUNTIME EVIDENCE
    # ========================================================

    grounding_valid: (
        bool | None
    ) = None

    gateway_policy_passed: (
        bool | None
    ) = None

    tool_execution_valid: (
        bool | None
    ) = None

    # ========================================================
    # CORRECTION EVIDENCE
    # ========================================================

    user_corrected: bool = False

    correction_type: (
        str | None
    ) = None

    failure_types: list[
        str
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # DATASET ELIGIBILITY
    #
    # Quality evidence alone does not automatically promote an
    # example into a training dataset.
    # ========================================================

    quality_eligible: bool = False


# ============================================================
# CORRECTIONS
# ============================================================


class CorrectionValue(
    BaseModel
):
    """
    One explicitly corrected field/value pair.

    rejected_value:
        What the original runtime proposed.

    chosen_value:
        What later trusted evidence says should have been used.
    """

    field: str

    rejected_value: (
        Any | None
    ) = None

    chosen_value: (
        Any | None
    ) = None


class CorrectionEvent(
    BaseModel
):
    """
    Immutable correction evidence associated with an existing
    trajectory.

    Corrections are stored separately instead of rewriting the
    original raw trajectory.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "correction-event.v1"
        ),

        alias="schema",
    )

    correction_id: str

    observed_at: str

    trajectory_id: str

    # Optional specialist task target.
    #
    # This becomes necessary once one trajectory contains multiple
    # specialist executions.
    task_id: (
        str | None
    ) = None

    correction_type: str

    source: str = (
        "explicit_user"
    )

    values: list[
        CorrectionValue
    ] = Field(
        default_factory=list
    )

    note: (
        str | None
    ) = None

    # A correction is evidence.
    #
    # Promotion into a training dataset remains a separate trusted
    # operation.
    dataset_eligible: bool = False


# ============================================================
# PREFERENCE EXAMPLES
# ============================================================


class PreferenceOption(
    BaseModel
):
    """
    One side of a chosen/rejected preference pair.

    Structured behavior is preserved instead of prematurely
    flattening everything into training text.
    """

    agent: (
        str | None
    ) = None

    tool: (
        str | None
    ) = None

    arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    answer: (
        str | None
    ) = None


class PreferenceExample(
    BaseModel
):
    """
    Canonical chosen/rejected learning artifact derived from:

        trajectory
            +
        correction
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "preference-example.v1"
        ),

        alias="schema",
    )

    example_id: str

    created_at: str

    trajectory_id: str

    correction_id: str

    task_id: (
        str | None
    ) = None

    source: str

    correction_type: str

    user_request: str

    rejected: PreferenceOption

    chosen: PreferenceOption

    # Derived examples remain unapproved until an explicit dataset
    # promotion step succeeds.
    dataset_eligible: bool = False


# ============================================================
# DATASET PROMOTION
# ============================================================


class DatasetPromotion(
    BaseModel
):
    """
    Trusted provenance attached to one promoted dataset record.
    """

    promoted_at: str

    promoted_by: str

    reason: str


class PreferenceDatasetRecord(
    BaseModel
):
    """
    Training-eligible preference record produced by the trusted
    dataset promotion pipeline.

    Source PreferenceExample objects remain unchanged.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "preference-dataset-record.v1"
        ),

        alias="schema",
    )

    record_id: str

    source_example_id: str

    trajectory_id: str

    correction_id: str

    task_id: (
        str | None
    ) = None

    source: str

    correction_type: str

    user_request: str

    rejected: PreferenceOption

    chosen: PreferenceOption

    promotion: DatasetPromotion

    # Only records created by trusted dataset promotion become
    # eligible.
    dataset_eligible: bool = True


class DatasetManifest(
    BaseModel
):
    """
    Immutable metadata describing one versioned dataset snapshot.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "dataset-manifest.v1"
        ),

        alias="schema",
    )

    dataset_id: str

    version: str

    created_at: str

    dataset_type: str

    record_schema: str

    record_count: int

    content_sha256: str

    promoted_by: str

    promotion_reason: str

    source_example_ids: list[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# LEARNING TRAJECTORY
# ============================================================


class LearningTrajectory(
    BaseModel
):
    """
    Durable sanitized runtime observation.

    IMPORTANT:

    Older Phase 1A trajectory.v1 records were written before the
    quality field existed.

    Therefore:

        quality=None

    means:

        "legacy / not evaluated yet"

    rather than:

        "invalid trajectory"

    This preserves backward compatibility with previously captured
    runtime evidence.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "trajectory.v1"
        ),

        alias="schema",
    )

    trajectory_id: str

    observed_at: str

    job_id: str

    attempt: int

    user_request: str

    hub_model: str

    hub_status: str

    routes: list[
        str
    ] = Field(
        default_factory=list
    )

    steps: list[
        TrajectoryStep
    ] = Field(
        default_factory=list
    )

    final_answer: (
        str | None
    ) = None

    signals: TrajectorySignals

    execution_reward: (
        ExecutionReward
    )

    # ========================================================
    # BACKWARD-COMPATIBLE QUALITY FIELD
    #
    # Phase 1A trajectories do not contain this property.
    #
    # "= None" is important.
    #
    # Without the default, Pydantic would allow None as a value
    # while still requiring the field to exist.
    # ========================================================

    quality: (
        TrajectoryQuality
        | None
    ) = None

    # Raw trajectories are evidence only.
    #
    # They are never automatically training-eligible.
    dataset_eligible: bool = False