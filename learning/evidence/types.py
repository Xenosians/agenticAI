from __future__ import annotations

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.evidence.execution_provenance import (
    SpecialistExecutionProvenance,
)


# ============================================================
# RAW TRAJECTORY TYPES
# ============================================================


class TrajectoryStep(
    BaseModel
):
    task_id: str

    task_instructions: (
        str | None
    ) = None

    semantic_intent: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    # Exact live SemanticGuard decision returned during this
    # specialist execution.
    #
    # Historical records remain valid because this is optional.
    semantic_guard_decision: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    # Exact ToolGateway decision metadata returned during this
    # specialist execution.
    #
    # The provider result itself remains in tool_result.
    gateway_decision: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    execution_provenance: (
        SpecialistExecutionProvenance
        | None
    ) = None

    agent: str

    status: str

    outcome_code: (
        str | None
    ) = None

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

    quality_eligible: bool = False


# ============================================================
# TRAJECTORY QUALITY
# ============================================================


class TrajectoryQuality(
    BaseModel
):
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

    grounding_valid: (
        bool | None
    ) = None

    gateway_policy_passed: (
        bool | None
    ) = None

    tool_execution_valid: (
        bool | None
    ) = None

    user_corrected: bool = False

    correction_type: (
        str | None
    ) = None

    failure_types: list[
        str
    ] = Field(
        default_factory=list
    )

    quality_eligible: bool = False


# ============================================================
# CORRECTIONS
# ============================================================


class CorrectionValue(
    BaseModel
):
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

    dataset_eligible: bool = False


# ============================================================
# PREFERENCE EXAMPLES
# ============================================================


class PreferenceOption(
    BaseModel
):
    agent: (
        str | None
    ) = None

    tool_calls: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
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

    task_instructions: (
        str | None
    ) = None

    execution_provenance: (
        SpecialistExecutionProvenance
        | None
    ) = None

    source: str

    correction_type: str

    user_request: str

    rejected: PreferenceOption

    chosen: PreferenceOption

    dataset_eligible: bool = False


# ============================================================
# DATASET PROMOTION
# ============================================================


class DatasetPromotion(
    BaseModel
):
    promoted_at: str

    promoted_by: str

    reason: str


class PreferenceDatasetRecord(
    BaseModel
):
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

    task_instructions: (
        str | None
    ) = None

    execution_provenance: (
        SpecialistExecutionProvenance
        | None
    ) = None

    source: str

    correction_type: str

    user_request: str

    rejected: PreferenceOption

    chosen: PreferenceOption

    promotion: DatasetPromotion

    dataset_eligible: bool = True


class DatasetManifest(
    BaseModel
):
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

    Older trajectory.v1 records may omit optional fields introduced
    later. Optional defaults preserve backward compatibility.
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

    execution_reward: ExecutionReward

    quality: (
        TrajectoryQuality
        | None
    ) = None

    dataset_eligible: bool = False