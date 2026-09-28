from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from learning.code_corpus.indexer import (
    DEFAULT_CODE_CORPUS_ROOT,
)
from learning.code_corpus.types import (
    CodeCorpusChunk,
    CodeCorpusManifest,
)
from learning.continual.failure_diagnostics import (
    DEFAULT_FAILURE_MINING_ROOT,
    FailureDiagnosis,
    FailureMiningManifest,
)
from learning.continual.storage import (
    atomic_write_json,
    canonical_json,
    immutable_write_json,
    immutable_write_jsonl,
    load_jsonl_models,
    read_json_model,
    sha256_file,
)
from learning.curation.corpus_analysis import (
    load_trajectories,
)
from learning.curriculum.catalog import (
    get_curriculum,
)
from learning.curriculum.chapters import (
    CurriculumChapter,
    CurriculumDefinition,
)
from learning.curriculum.mastery import (
    CurriculumState,
    initial_curriculum_state,
    validate_state_against_curriculum,
)
from learning.curriculum.lessons import (
    CURRICULUM_LESSON_DATASET_ROOT,
    CurriculumLessonDatasetRecord,
)
from learning.evidence.hub_routing import (
    DEFAULT_HUB_ROUTING_LEDGER,
    HubRoutingLedgerRecord,
)
from learning.evidence.types import (
    LearningTrajectory,
    PreferenceDatasetRecord,
)
from learning.training.hub_preferences import (
    HUB_PREFERENCE_DATASET_ROOT,
    HubPreferenceRecord,
)
from learning.paths import (
    DATASETS_ROOT,
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
    TRAJECTORIES_PATH,
)


MemberKind = Literal[
    "preference",
    "code_chunk",
    "curriculum_lesson",
]

MemberRole = Literal[
    "current_behavior",
    "replay_behavior",
    "current_code",
    "replay_code",
    "foundation_code",
    "current_lesson",
    "replay_lesson",
]

Partition = Literal[
    "train",
    "validation",
]


DEFAULT_MEMBERSHIP_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "training-membership"
)


class MembershipCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member_id: str
    member_kind: MemberKind
    role: MemberRole

    target_component: str
    chapter_id: str

    source_id: str
    source_artifact: str
    source_artifact_sha256: str

    lineage_id: str
    split_group: str

    logical_repository: str | None = None
    relative_path: str | None = None
    language: str | None = None

    trajectory_id: str | None = None
    correction_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class TrainingMembershipRecord(BaseModel):
    """
    One exact member of a future training/validation set.

    This record references immutable/promoted evidence by identity. It does not
    duplicate raw training payloads and does not authorize optimization.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="training-membership-record.v1",
        alias="schema",
    )

    member_id: str
    member_kind: MemberKind
    role: MemberRole
    partition: Partition

    target_component: str
    chapter_id: str

    source_id: str
    source_artifact: str
    source_artifact_sha256: str

    lineage_id: str
    split_group: str

    logical_repository: str | None = None
    relative_path: str | None = None
    language: str | None = None

    trajectory_id: str | None = None
    correction_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    training_eligible: bool = True


class HardCaseReference(BaseModel):
    """
    Review queue only.

    A hard case is never a training member merely because runtime diagnostics
    found it. It must cross the trusted correction/review/materialization
    boundary first.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="training-hard-case-reference.v1",
        alias="schema",
    )

    diagnosis_id: str
    trajectory_id: str
    task_id: str | None = None

    failure_code: str
    component: str

    agent: str | None = None
    tool: str | None = None

    current_chapter_id: str

    hub_routing_attempt_ids: list[str] = Field(
        default_factory=list
    )
    exact_hub_provenance_ready: bool = False

    training_eligible: bool = False


class TrainingMembershipManifest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="training-membership-plan.v1",
        alias="schema",
    )

    plan_id: str
    created_at: str

    target_component: str
    curriculum_id: str
    curriculum_version: str

    current_chapter_id: str | None
    current_chapter_title: str | None
    mastered_chapter_ids: list[str] = Field(
        default_factory=list
    )

    seed: str
    validation_fraction: float

    source_fingerprints: dict[str, str] = Field(
        default_factory=dict
    )

    member_count: int
    train_member_count: int
    validation_member_count: int

    member_kind_counts: dict[str, int] = Field(
        default_factory=dict
    )
    role_counts: dict[str, int] = Field(
        default_factory=dict
    )
    partition_counts: dict[str, int] = Field(
        default_factory=dict
    )

    reviewed_current_behavior_count: int
    reviewed_current_lesson_count: int = 0
    reviewed_current_training_example_count: int = 0
    required_reviewed_current_behavior_count: int

    hard_case_count: int
    exact_provenance_hard_case_count: int
    hard_case_failure_counts: dict[str, int] = Field(
        default_factory=dict
    )

    excluded_counts: dict[str, int] = Field(
        default_factory=dict
    )

    membership_sha256: str
    hard_cases_sha256: str

    checks: dict[str, bool] = Field(
        default_factory=dict
    )
    blocked_reasons: list[str] = Field(
        default_factory=list
    )
    readiness_ready: bool = False

    training_authorized: bool = False
    promotion_authorized: bool = False


class TrainingMembershipBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: TrainingMembershipManifest
    output_directory: str


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def ensure_curriculum_state(
    *,
    curriculum: CurriculumDefinition,
    state_path: Path,
) -> CurriculumState:
    """
    Load the durable curriculum state when present.

    If this curriculum has never been planned before, initialize it at
    Chapter 1 using the same constructor as the curriculum planner.

    This is bookkeeping only; it does not train, promote, or activate.
    """

    resolved = state_path.expanduser().resolve()

    if resolved.is_file():
        return read_json_model(
            resolved,
            CurriculumState,
        )

    state = initial_curriculum_state(
        curriculum
    )

    atomic_write_json(
        resolved,
        state,
    )

    return state


def _stable_id(
    prefix: str,
    payload: dict[str, Any],
) -> str:
    digest = hashlib.sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"{prefix}-"
        f"{digest[:24]}"
    )


def _relative_artifact_path(
    path: Path,
) -> str:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    try:
        return str(
            resolved.relative_to(
                REPOSITORY_ROOT
            )
        )

    except ValueError:
        return str(
            resolved
        )


def _required_sha256(
    path: Path,
) -> str:
    value = sha256_file(
        path
    )

    if not value:
        raise ValueError(
            "Unable to fingerprint source artifact: "
            f"{path}"
        )

    return value


def _trajectory_failure_codes(
    trajectory: LearningTrajectory,
) -> set[str]:
    result: set[str] = set()

    for step in trajectory.steps:
        decision = (
            step.semantic_guard_decision
        )

        if isinstance(
            decision,
            dict,
        ):
            code = decision.get(
                "decision_code"
            )

            if (
                isinstance(
                    code,
                    str,
                )
                and code.strip()
            ):
                result.add(
                    code.strip()
                )

        if (
            isinstance(
                step.outcome_code,
                str,
            )
            and step.outcome_code.strip()
        ):
            result.add(
                step.outcome_code.strip()
            )

    return result


def chapter_ids_for_failure_codes(
    *,
    curriculum: CurriculumDefinition,
    failure_codes: set[str],
) -> list[str]:
    if not failure_codes:
        return []

    result: list[str] = []

    for chapter in curriculum.chapters:
        if (
            failure_codes
            .intersection(
                chapter.failure_codes
            )
        ):
            result.append(
                chapter.chapter_id
            )

    return result


def classify_behavior_role(
    *,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
    failure_codes: set[str],
) -> MemberRole | None:
    if state.current_chapter_id is None:
        return None

    matching = set(
        chapter_ids_for_failure_codes(
            curriculum=curriculum,
            failure_codes=failure_codes,
        )
    )

    if state.current_chapter_id in matching:
        return "current_behavior"

    if matching.intersection(
        state.mastered_chapter_ids
    ):
        return "replay_behavior"

    return None


def classify_code_role(
    *,
    target_component: str,
    state: CurriculumState,
    chunk: CodeCorpusChunk,
) -> MemberRole | None:
    if not chunk.training_eligible:
        return None

    if (
        target_component
        not in chunk.audiences
    ):
        return None

    if target_component == "hub":
        if chunk.source_kind != "general":
            return None

        return "foundation_code"

    if (
        state.current_chapter_id
        is not None
        and state.current_chapter_id
        in chunk.curriculum_hints
    ):
        return "current_code"

    if set(
        state.mastered_chapter_ids
    ).intersection(
        chunk.curriculum_hints
    ):
        return "replay_code"

    return None


def _target_for_preference_record(
    record: PreferenceDatasetRecord,
) -> str | None:
    provenance = (
        record.execution_provenance
    )

    if provenance is None:
        # Hub records require their own exact routing provenance bridge.
        # Do not guess ownership from correction_type.
        return None

    name = (
        provenance.agent_name.strip()
    )

    return (
        name
        if name
        else None
    )


def _latest_preference_records(
    *,
    dataset_root: Path,
) -> tuple[
    list[
        tuple[
            PreferenceDatasetRecord,
            str,
            str,
        ]
    ],
    dict[str, str],
    Counter[str],
]:
    """
    Load promoted preference dataset versions only.

    Raw corrections/reviews are intentionally not read here. Dataset promotion
    is already the trusted curation boundary.
    """

    root = (
        dataset_root
        .expanduser()
        .resolve()
        / "preference"
    )

    if not root.is_dir():
        return (
            [],
            {},
            Counter(),
        )

    versions = sorted(
        child
        for child in root.iterdir()
        if (
            child.is_dir()
            and child.name.startswith(
                "v"
            )
        )
    )

    latest_by_example: dict[
        str,
        tuple[
            PreferenceDatasetRecord,
            str,
            str,
        ]
    ] = {}

    source_fingerprints: dict[
        str,
        str,
    ] = {}

    exclusions: Counter[str] = Counter()

    for version in versions:
        manifest_path = (
            version
            / "manifest.json"
        )

        records_path = (
            version
            / "records.jsonl"
        )

        if not (
            manifest_path.is_file()
            and records_path.is_file()
        ):
            exclusions[
                "preference_dataset_incomplete"
            ] += 1
            continue

        source_fingerprints[
            _relative_artifact_path(
                manifest_path
            )
        ] = _required_sha256(
            manifest_path
        )

        records_sha = _required_sha256(
            records_path
        )

        source_key = (
            _relative_artifact_path(
                records_path
            )
        )

        source_fingerprints[
            source_key
        ] = records_sha

        records = load_jsonl_models(
            records_path,
            PreferenceDatasetRecord,
        )

        for record in records:
            if not record.dataset_eligible:
                exclusions[
                    "preference_record_not_eligible"
                ] += 1
                continue

            prior = latest_by_example.get(
                record.source_example_id
            )

            if prior is not None:
                exclusions[
                    "duplicate_promoted_preference"
                ] += 1

            latest_by_example[
                record.source_example_id
            ] = (
                record,
                source_key,
                records_sha,
            )

    result = list(
        latest_by_example.values()
    )

    result.sort(
        key=lambda item: (
            item[0].source_example_id,
            item[0].record_id,
        )
    )

    return (
        result,
        source_fingerprints,
        exclusions,
    )


def _latest_hub_preference_records(
    *,
    dataset_root: Path = HUB_PREFERENCE_DATASET_ROOT,
) -> tuple[
    list[
        tuple[
            HubPreferenceRecord,
            str,
            str,
        ]
    ],
    dict[str, str],
    Counter[str],
]:
    root = dataset_root.expanduser().resolve()

    if not root.is_dir():
        return (
            [],
            {},
            Counter(),
        )

    latest_by_correction: dict[
        str,
        tuple[
            HubPreferenceRecord,
            str,
            str,
        ],
    ] = {}

    source_fingerprints: dict[str, str] = {}
    exclusions: Counter[str] = Counter()

    versions = sorted(
        child
        for child in root.iterdir()
        if (
            child.is_dir()
            and child.name.startswith("v")
        )
    )

    for version in versions:
        manifest_path = version / "manifest.json"
        records_path = version / "records.jsonl"

        if not (
            manifest_path.is_file()
            and records_path.is_file()
        ):
            exclusions[
                "hub_preference_dataset_incomplete"
            ] += 1
            continue

        source_fingerprints[
            _relative_artifact_path(
                manifest_path
            )
        ] = _required_sha256(
            manifest_path
        )

        records_sha = _required_sha256(
            records_path
        )
        source_key = _relative_artifact_path(
            records_path
        )
        source_fingerprints[
            source_key
        ] = records_sha

        records = load_jsonl_models(
            records_path,
            HubPreferenceRecord,
        )

        for record in records:
            if not record.dataset_eligible:
                exclusions[
                    "hub_preference_record_not_eligible"
                ] += 1
                continue

            prior = latest_by_correction.get(
                record.correction_id
            )

            if prior is not None:
                exclusions[
                    "duplicate_promoted_hub_preference"
                ] += 1

            latest_by_correction[
                record.correction_id
            ] = (
                record,
                source_key,
                records_sha,
            )

    result = list(
        latest_by_correction.values()
    )
    result.sort(
        key=lambda item: (
            item[0].trajectory_id,
            item[0].correction_id,
            item[0].record_id,
        )
    )

    return (
        result,
        source_fingerprints,
        exclusions,
    )


def _hub_preference_candidates(
    *,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
    promoted_records: list[
        tuple[
            HubPreferenceRecord,
            str,
            str,
        ]
    ],
    exclusions: Counter[str],
) -> list[MembershipCandidate]:
    if curriculum.target_component != "hub":
        return []

    if state.current_chapter_id is None:
        return []

    result: list[MembershipCandidate] = []

    for (
        record,
        source_artifact,
        source_sha256,
    ) in promoted_records:
        role = classify_behavior_role(
            curriculum=curriculum,
            state=state,
            failure_codes=set(
                record.failure_codes
            ),
        )

        if role is None:
            exclusions[
                "hub_preference_chapter_unmatched"
            ] += 1
            continue

        identity = {
            "record_id": record.record_id,
            "target_component": "hub",
            "chapter_id": state.current_chapter_id,
            "role": role,
        }

        result.append(
            MembershipCandidate(
                member_id=_stable_id(
                    "membership",
                    identity,
                ),
                member_kind="preference",
                role=role,
                target_component="hub",
                chapter_id=state.current_chapter_id,
                source_id=record.record_id,
                source_artifact=source_artifact,
                source_artifact_sha256=source_sha256,
                lineage_id=record.correction_id,
                split_group=(
                    "trajectory:"
                    + record.trajectory_id
                ),
                trajectory_id=record.trajectory_id,
                correction_id=record.correction_id,
                metadata={
                    "source_attempt_id":
                        record.source_attempt_id,
                    "target_model_key":
                        record.target_model_key,
                    "source_messages_sha256":
                        record.source_messages_sha256,
                    "source_model_artifact_sha256":
                        record.source_model_artifact_sha256,
                    "preference_target":
                        "hub",
                },
            )
        )

    return result


def _latest_code_corpora(
    *,
    code_corpus_root: Path,
) -> tuple[
    list[
        tuple[
            CodeCorpusManifest,
            Path,
            Path,
        ]
    ],
    dict[str, str],
    Counter[str],
]:
    root = (
        code_corpus_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        return (
            [],
            {},
            Counter(),
        )

    latest: dict[
        str,
        tuple[
            CodeCorpusManifest,
            Path,
            Path,
        ]
    ] = {}

    source_fingerprints: dict[
        str,
        str,
    ] = {}

    exclusions: Counter[str] = Counter()

    for manifest_path in sorted(
        root.glob(
            "*/*/manifest.json"
        )
    ):
        try:
            manifest = (
                CodeCorpusManifest
                .model_validate_json(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        except Exception:
            exclusions[
                "invalid_code_corpus_manifest"
            ] += 1
            continue

        chunks_path = (
            manifest_path.parent
            / "chunks.jsonl"
        )

        if not chunks_path.is_file():
            exclusions[
                "code_corpus_chunks_missing"
            ] += 1
            continue

        repository = (
            manifest
            .source
            .logical_repository
        )

        candidate = (
            manifest,
            manifest_path,
            chunks_path,
        )

        prior = latest.get(
            repository
        )

        if (
            prior is None
            or (
                manifest.created_at,
                manifest.corpus_id,
            )
            > (
                prior[0].created_at,
                prior[0].corpus_id,
            )
        ):
            latest[
                repository
            ] = candidate

    result = [
        latest[key]
        for key in sorted(
            latest
        )
    ]

    for (
        manifest,
        manifest_path,
        chunks_path,
    ) in result:
        source_fingerprints[
            _relative_artifact_path(
                manifest_path
            )
        ] = _required_sha256(
            manifest_path
        )

        source_fingerprints[
            _relative_artifact_path(
                chunks_path
            )
        ] = _required_sha256(
            chunks_path
        )

    return (
        result,
        source_fingerprints,
        exclusions,
    )


def _latest_curriculum_lesson_records(
    *,
    curriculum: CurriculumDefinition,
    dataset_root: Path = CURRICULUM_LESSON_DATASET_ROOT,
) -> tuple[
    list[tuple[CurriculumLessonDatasetRecord, str, str]],
    dict[str, str],
    Counter[str],
]:
    root = dataset_root.expanduser().resolve() / curriculum.curriculum_id

    if not root.is_dir():
        return [], {}, Counter()

    latest_by_lesson: dict[
        str,
        tuple[CurriculumLessonDatasetRecord, str, str],
    ] = {}
    source_fingerprints: dict[str, str] = {}
    exclusions: Counter[str] = Counter()

    for version in sorted(child for child in root.iterdir() if child.is_dir() and child.name.startswith("v")):
        manifest_path = version / "manifest.json"
        records_path = version / "records.jsonl"

        if not (manifest_path.is_file() and records_path.is_file()):
            exclusions["curriculum_lesson_dataset_incomplete"] += 1
            continue

        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        if (
            manifest_payload.get("curriculum_id") != curriculum.curriculum_id
            or manifest_payload.get("curriculum_version") != curriculum.version
            or manifest_payload.get("target_component") != curriculum.target_component
        ):
            exclusions["curriculum_lesson_dataset_identity_mismatch"] += 1
            continue

        source_fingerprints[_relative_artifact_path(manifest_path)] = _required_sha256(manifest_path)
        records_sha = _required_sha256(records_path)
        source_key = _relative_artifact_path(records_path)
        source_fingerprints[source_key] = records_sha

        for record in load_jsonl_models(records_path, CurriculumLessonDatasetRecord):
            if not record.dataset_eligible:
                exclusions["curriculum_lesson_not_eligible"] += 1
                continue
            if (
                record.curriculum_id != curriculum.curriculum_id
                or record.curriculum_version != curriculum.version
                or record.target_component != curriculum.target_component
            ):
                exclusions["curriculum_lesson_record_identity_mismatch"] += 1
                continue
            if record.training_authorized is not False:
                exclusions["curriculum_lesson_training_flag_invalid"] += 1
                continue
            if record.lesson_id in latest_by_lesson:
                exclusions["duplicate_promoted_curriculum_lesson"] += 1
            latest_by_lesson[record.lesson_id] = (record, source_key, records_sha)

    result = list(latest_by_lesson.values())
    result.sort(key=lambda item: (item[0].chapter_id, item[0].lesson_id))
    return result, source_fingerprints, exclusions


def classify_curriculum_lesson_role(
    *,
    target_component: str,
    state: CurriculumState,
    record: CurriculumLessonDatasetRecord,
) -> MemberRole | None:
    if record.target_component != target_component:
        return None
    if state.current_chapter_id == record.chapter_id:
        return "current_lesson"
    if record.chapter_id in state.mastered_chapter_ids:
        return "replay_lesson"
    return None


def _curriculum_lesson_candidates(
    *,
    target_component: str,
    state: CurriculumState,
    promoted_records: list[tuple[CurriculumLessonDatasetRecord, str, str]],
    exclusions: Counter[str],
) -> list[MembershipCandidate]:
    if state.current_chapter_id is None:
        return []

    result: list[MembershipCandidate] = []

    for record, source_artifact, source_sha256 in promoted_records:
        role = classify_curriculum_lesson_role(
            target_component=target_component,
            state=state,
            record=record,
        )
        if role is None:
            exclusions["curriculum_lesson_chapter_unmatched"] += 1
            continue

        identity = {
            "record_id": record.record_id,
            "target_component": target_component,
            "chapter_id": state.current_chapter_id,
            "role": role,
        }

        result.append(MembershipCandidate(
            member_id=_stable_id("membership", identity),
            member_kind="curriculum_lesson",
            role=role,
            target_component=target_component,
            chapter_id=state.current_chapter_id,
            source_id=record.record_id,
            source_artifact=source_artifact,
            source_artifact_sha256=source_sha256,
            lineage_id=record.lesson_id,
            split_group="lesson:" + record.lesson_id,
            metadata={
                "lesson_id": record.lesson_id,
                "lesson_review_id": record.lesson_review_id,
                "source_lesson_sha256": record.source_lesson_sha256,
                "has_rejected_response": record.rejected_response is not None,
            },
        ))

    return result


def _latest_failure_run(
    *,
    failure_root: Path,
) -> tuple[
    FailureMiningManifest
    | None,
    list[
        FailureDiagnosis
    ],
    dict[str, str],
]:
    root = (
        failure_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        return (
            None,
            [],
            {},
        )

    candidates: list[
        tuple[
            FailureMiningManifest,
            Path,
        ]
    ] = []

    for manifest_path in root.glob(
        "*/manifest.json"
    ):
        try:
            manifest = (
                FailureMiningManifest
                .model_validate_json(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        except Exception:
            continue

        candidates.append(
            (
                manifest,
                manifest_path,
            )
        )

    if not candidates:
        return (
            None,
            [],
            {},
        )

    manifest, manifest_path = max(
        candidates,
        key=lambda item: (
            item[0].created_at,
            item[0].run_id,
        ),
    )

    diagnoses_path = (
        manifest_path.parent
        / "diagnoses.jsonl"
    )

    diagnoses = (
        load_jsonl_models(
            diagnoses_path,
            FailureDiagnosis,
        )
        if diagnoses_path.is_file()
        else []
    )

    fingerprints = {
        _relative_artifact_path(
            manifest_path
        ):
            _required_sha256(
                manifest_path
            ),
    }

    if diagnoses_path.is_file():
        fingerprints[
            _relative_artifact_path(
                diagnoses_path
            )
        ] = _required_sha256(
            diagnoses_path
        )

    candidate_path = (
        manifest_path.parent
        / "candidates.jsonl"
    )

    if candidate_path.is_file():
        fingerprints[
            _relative_artifact_path(
                candidate_path
            )
        ] = _required_sha256(
            candidate_path
        )

    return (
        manifest,
        diagnoses,
        fingerprints,
    )


def _load_hub_routing_ledger(
    *,
    path: Path,
) -> tuple[
    list[
        HubRoutingLedgerRecord
    ],
    dict[str, str],
]:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        return (
            [],
            {},
        )

    records = load_jsonl_models(
        resolved,
        HubRoutingLedgerRecord,
    )

    return (
        records,
        {
            _relative_artifact_path(
                resolved
            ):
                _required_sha256(
                    resolved
                ),
        },
    )


def _hard_cases_for_current_chapter(
    *,
    target_component: str,
    chapter: CurriculumChapter,
    diagnoses: list[FailureDiagnosis],
    hub_records: list[HubRoutingLedgerRecord],
) -> list[HardCaseReference]:
    accepted_hub_attempts: dict[
        str,
        list[str],
    ] = defaultdict(
        list
    )

    for record in hub_records:
        attempt = record.attempt

        if (
            attempt.validation_status
            == "accepted"
            and attempt.model_identity.complete
            and attempt.raw_response_exact
        ):
            accepted_hub_attempts[
                record.trajectory_id
            ].append(
                attempt.attempt_id
            )

    result: list[
        HardCaseReference
    ] = []

    allowed_codes = set(
        chapter.failure_codes
    )

    for diagnosis in diagnoses:
        if (
            diagnosis.failure_code
            not in allowed_codes
        ):
            continue

        if target_component == "hub":
            if diagnosis.component != "hub_router":
                continue

        else:
            if diagnosis.agent != target_component:
                continue

        attempt_ids = sorted(
            set(
                accepted_hub_attempts.get(
                    diagnosis.trajectory_id,
                    [],
                )
            )
        )

        result.append(
            HardCaseReference(
                diagnosis_id=(
                    diagnosis.diagnosis_id
                ),
                trajectory_id=(
                    diagnosis.trajectory_id
                ),
                task_id=(
                    diagnosis.task_id
                ),
                failure_code=(
                    diagnosis.failure_code
                ),
                component=(
                    diagnosis.component
                ),
                agent=(
                    diagnosis.agent
                ),
                tool=(
                    diagnosis.tool
                ),
                current_chapter_id=(
                    chapter.chapter_id
                ),
                hub_routing_attempt_ids=(
                    attempt_ids
                ),
                exact_hub_provenance_ready=(
                    bool(
                        attempt_ids
                    )
                    if target_component
                    == "hub"
                    else True
                ),
                training_eligible=False,
            )
        )

    result.sort(
        key=lambda item: (
            item.failure_code,
            item.diagnosis_id,
        )
    )

    return result


def _preference_candidates(
    *,
    target_component: str,
    curriculum: CurriculumDefinition,
    state: CurriculumState,
    trajectories_by_id: dict[
        str,
        LearningTrajectory,
    ],
    promoted_records: list[
        tuple[
            PreferenceDatasetRecord,
            str,
            str,
        ]
    ],
    exclusions: Counter[str],
) -> list[MembershipCandidate]:
    result: list[
        MembershipCandidate
    ] = []

    for (
        record,
        source_artifact,
        source_sha256,
    ) in promoted_records:
        target = (
            _target_for_preference_record(
                record
            )
        )

        if target is None:
            exclusions[
                "preference_target_provenance_missing"
            ] += 1
            continue

        if target != target_component:
            exclusions[
                "preference_target_mismatch"
            ] += 1
            continue

        trajectory = (
            trajectories_by_id
            .get(
                record.trajectory_id
            )
        )

        if trajectory is None:
            exclusions[
                "preference_trajectory_missing"
            ] += 1
            continue

        role = classify_behavior_role(
            curriculum=curriculum,
            state=state,
            failure_codes=(
                _trajectory_failure_codes(
                    trajectory
                )
            ),
        )

        if role is None:
            exclusions[
                "preference_chapter_unmatched"
            ] += 1
            continue

        current_chapter_id = (
            state.current_chapter_id
        )

        if current_chapter_id is None:
            continue

        identity = {
            "record_id":
                record.record_id,

            "target_component":
                target_component,

            "chapter_id":
                current_chapter_id,

            "role":
                role,
        }

        result.append(
            MembershipCandidate(
                member_id=(
                    _stable_id(
                        "membership",
                        identity,
                    )
                ),
                member_kind=(
                    "preference"
                ),
                role=role,
                target_component=(
                    target_component
                ),
                chapter_id=(
                    current_chapter_id
                ),
                source_id=(
                    record.record_id
                ),
                source_artifact=(
                    source_artifact
                ),
                source_artifact_sha256=(
                    source_sha256
                ),
                lineage_id=(
                    record.source_example_id
                ),
                split_group=(
                    "trajectory:"
                    + record.trajectory_id
                ),
                trajectory_id=(
                    record.trajectory_id
                ),
                correction_id=(
                    record.correction_id
                ),
                metadata={
                    "source_example_id":
                        record.source_example_id,

                    "correction_type":
                        record.correction_type,

                    "promotion_source":
                        record.promotion.promoted_by,
                },
            )
        )

    return result


def _code_candidates(
    *,
    target_component: str,
    state: CurriculumState,
    corpora: list[
        tuple[
            CodeCorpusManifest,
            Path,
            Path,
        ]
    ],
    exclusions: Counter[str],
) -> list[MembershipCandidate]:
    result: list[
        MembershipCandidate
    ] = []

    if state.current_chapter_id is None:
        return result

    seen_content: set[
        str
    ] = set()

    for (
        manifest,
        _manifest_path,
        chunks_path,
    ) in corpora:
        source_artifact = (
            _relative_artifact_path(
                chunks_path
            )
        )

        source_sha256 = (
            _required_sha256(
                chunks_path
            )
        )

        chunks = load_jsonl_models(
            chunks_path,
            CodeCorpusChunk,
        )

        for chunk in chunks:
            if not chunk.training_eligible:
                exclusions[
                    "code_chunk_not_training_eligible"
                ] += 1
                continue

            if (
                target_component
                not in chunk.audiences
            ):
                exclusions[
                    "code_chunk_target_mismatch"
                ] += 1
                continue

            role = classify_code_role(
                target_component=(
                    target_component
                ),
                state=state,
                chunk=chunk,
            )

            if role is None:
                exclusions[
                    "code_chunk_chapter_unmatched"
                ] += 1
                continue

            if (
                chunk.content_sha256
                in seen_content
            ):
                exclusions[
                    "duplicate_code_content"
                ] += 1
                continue

            seen_content.add(
                chunk.content_sha256
            )

            identity = {
                "chunk_id":
                    chunk.chunk_id,

                "target_component":
                    target_component,

                "chapter_id":
                    state.current_chapter_id,

                "role":
                    role,
            }

            result.append(
                MembershipCandidate(
                    member_id=(
                        _stable_id(
                            "membership",
                            identity,
                        )
                    ),
                    member_kind=(
                        "code_chunk"
                    ),
                    role=role,
                    target_component=(
                        target_component
                    ),
                    chapter_id=(
                        state.current_chapter_id
                    ),
                    source_id=(
                        chunk.chunk_id
                    ),
                    source_artifact=(
                        source_artifact
                    ),
                    source_artifact_sha256=(
                        source_sha256
                    ),
                    lineage_id=(
                        chunk.content_sha256
                    ),
                    split_group=(
                        "code-file:"
                        + chunk.logical_repository
                        + ":"
                        + chunk.revision_id
                        + ":"
                        + chunk.relative_path
                    ),
                    logical_repository=(
                        chunk.logical_repository
                    ),
                    relative_path=(
                        chunk.relative_path
                    ),
                    language=(
                        chunk.language
                    ),
                    metadata={
                        "corpus_id":
                            manifest.corpus_id,

                        "revision_kind":
                            chunk.revision_kind,

                        "revision_id":
                            chunk.revision_id,

                        "symbol":
                            chunk.symbol,

                        "symbol_kind":
                            chunk.symbol_kind,

                        "start_line":
                            chunk.start_line,

                        "end_line":
                            chunk.end_line,

                        "content_sha256":
                            chunk.content_sha256,
                    },
                )
            )

    return result


def cap_code_candidates(
    candidates: list[MembershipCandidate],
    *,
    max_code_members: int,
    max_repository_fraction: float,
    seed: str,
) -> tuple[
    list[MembershipCandidate],
    int,
]:
    if max_code_members < 1:
        raise ValueError(
            "max_code_members must be positive"
        )

    if not (
        0.0
        < max_repository_fraction
        <= 1.0
    ):
        raise ValueError(
            "max_repository_fraction must be in (0, 1]"
        )

    behavior = [
        item
        for item in candidates
        if item.member_kind
        != "code_chunk"
    ]

    code = [
        item
        for item in candidates
        if item.member_kind
        == "code_chunk"
    ]

    if len(code) <= max_code_members:
        return (
            candidates,
            0,
        )

    role_weight = {
        "current_code": 3,
        "replay_code": 2,
        "foundation_code": 1,
    }

    def rank(
        item: MembershipCandidate,
    ) -> tuple:
        tie = hashlib.sha256(
            (
                seed
                + ":"
                + item.member_id
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            -role_weight.get(
                item.role,
                0,
            ),
            tie,
        )

    ordered = sorted(
        code,
        key=rank,
    )

    repo_limit = max(
        1,
        int(
            max_code_members
            * max_repository_fraction
        ),
    )

    selected: list[
        MembershipCandidate
    ] = []

    deferred: list[
        MembershipCandidate
    ] = []

    repository_counts: Counter[
        str
    ] = Counter()

    for item in ordered:
        repository = (
            item.logical_repository
            or "unknown"
        )

        if (
            repository_counts[
                repository
            ]
            >= repo_limit
        ):
            deferred.append(
                item
            )
            continue

        selected.append(
            item
        )

        repository_counts[
            repository
        ] += 1

        if (
            len(
                selected
            )
            >= max_code_members
        ):
            break

    for item in deferred:
        if (
            len(
                selected
            )
            >= max_code_members
        ):
            break

        selected.append(
            item
        )

    dropped = (
        len(
            code
        )
        - len(
            selected
        )
    )

    return (
        behavior
        + selected,
        dropped,
    )


def assign_partitions(
    candidates: list[MembershipCandidate],
    *,
    validation_fraction: float,
    seed: str,
) -> list[TrainingMembershipRecord]:
    """
    Deterministic group split.

    Same trajectory and same source file cannot appear in both train and
    validation. Buckets are split independently by kind+role so rare replay or
    current-chapter evidence is not silently swallowed by a much larger code
    pool.
    """

    if not (
        0.0
        < validation_fraction
        < 0.5
    ):
        raise ValueError(
            "validation_fraction must be in (0, 0.5)"
        )

    buckets: dict[
        tuple[
            str,
            str,
        ],
        list[
            MembershipCandidate
        ],
    ] = defaultdict(
        list
    )

    for item in candidates:
        buckets[
            (
                item.member_kind,
                item.role,
            )
        ].append(
            item
        )

    result: list[
        TrainingMembershipRecord
    ] = []

    for bucket_key in sorted(
        buckets
    ):
        items = buckets[
            bucket_key
        ]

        groups: dict[
            str,
            list[
                MembershipCandidate
            ],
        ] = defaultdict(
            list
        )

        for item in items:
            groups[
                item.split_group
            ].append(
                item
            )

        ordered_groups = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                (
                    seed
                    + ":"
                    + bucket_key[0]
                    + ":"
                    + bucket_key[1]
                    + ":"
                    + group
                ).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )

        if len(
            ordered_groups
        ) < 2:
            validation_groups: set[
                str
            ] = set()

        else:
            validation_count = max(
                1,
                round(
                    len(
                        ordered_groups
                    )
                    * validation_fraction
                ),
            )

            validation_count = min(
                validation_count,
                len(
                    ordered_groups
                )
                - 1,
            )

            validation_groups = set(
                ordered_groups[
                    :validation_count
                ]
            )

        for group in ordered_groups:
            partition: Partition = (
                "validation"
                if group
                in validation_groups
                else "train"
            )

            for item in sorted(
                groups[
                    group
                ],
                key=lambda value: (
                    value.member_id
                ),
            ):
                result.append(
                    TrainingMembershipRecord(
                        **item.model_dump(
                            mode="python"
                        ),
                        partition=partition,
                        training_eligible=True,
                    )
                )

    result.sort(
        key=lambda item: (
            item.partition,
            item.member_kind,
            item.role,
            item.member_id,
        )
    )

    return result


def build_training_membership_plan(
    *,
    curriculum_name: str,
    state_path: Path | None = None,
    output_root: Path = DEFAULT_MEMBERSHIP_ROOT,
    trajectories_path: Path = TRAJECTORIES_PATH,
    dataset_root: Path = DATASETS_ROOT,
    code_corpus_root: Path = DEFAULT_CODE_CORPUS_ROOT,
    failure_root: Path = DEFAULT_FAILURE_MINING_ROOT,
    hub_routing_path: Path = DEFAULT_HUB_ROUTING_LEDGER,
    validation_fraction: float = 0.20,
    seed: str = "training-membership-v1",
    max_code_members: int = 2048,
    max_repository_fraction: float = 0.60,
) -> TrainingMembershipBuildResult:
    curriculum = get_curriculum(
        curriculum_name
    )

    resolved_state_path = (
        state_path
        if state_path is not None
        else (
            RUNTIME_LEARNING_ROOT
            / "continual"
            / "curriculum"
            / curriculum.curriculum_id
            / "state.json"
        )
    ).expanduser().resolve()

    state = ensure_curriculum_state(
        curriculum=curriculum,
        state_path=resolved_state_path,
    )

    validate_state_against_curriculum(
        curriculum=curriculum,
        state=state,
    )

    source_fingerprints: dict[
        str,
        str,
    ] = {
        _relative_artifact_path(
            resolved_state_path
        ):
            _required_sha256(
                resolved_state_path
            ),
    }

    exclusions: Counter[
        str
    ] = Counter()

    resolved_trajectories_path = (
        trajectories_path
        .expanduser()
        .resolve()
    )

    trajectories = load_trajectories(
        resolved_trajectories_path
    )

    trajectories_by_id = {
        item.trajectory_id:
            item

        for item in trajectories
    }

    if resolved_trajectories_path.is_file():
        source_fingerprints[
            _relative_artifact_path(
                resolved_trajectories_path
            )
        ] = _required_sha256(
            resolved_trajectories_path
        )

    (
        promoted_records,
        preference_fingerprints,
        preference_exclusions,
    ) = _latest_preference_records(
        dataset_root=dataset_root,
    )

    source_fingerprints.update(
        preference_fingerprints
    )

    exclusions.update(
        preference_exclusions
    )

    hub_promoted_records = []

    if curriculum.target_component == "hub":
        (
            hub_promoted_records,
            hub_preference_fingerprints,
            hub_preference_exclusions,
        ) = _latest_hub_preference_records()

        source_fingerprints.update(
            hub_preference_fingerprints
        )

        exclusions.update(
            hub_preference_exclusions
        )

    (
        corpora,
        code_fingerprints,
        code_exclusions,
    ) = _latest_code_corpora(
        code_corpus_root=(
            code_corpus_root
        ),
    )

    source_fingerprints.update(
        code_fingerprints
    )

    exclusions.update(
        code_exclusions
    )

    (
        lesson_promoted_records,
        lesson_fingerprints,
        lesson_exclusions,
    ) = _latest_curriculum_lesson_records(
        curriculum=curriculum,
    )

    source_fingerprints.update(
        lesson_fingerprints
    )

    exclusions.update(
        lesson_exclusions
    )

    (
        _failure_manifest,
        diagnoses,
        failure_fingerprints,
    ) = _latest_failure_run(
        failure_root=(
            failure_root
        ),
    )

    source_fingerprints.update(
        failure_fingerprints
    )

    (
        hub_records,
        hub_fingerprints,
    ) = _load_hub_routing_ledger(
        path=hub_routing_path,
    )

    source_fingerprints.update(
        hub_fingerprints
    )

    candidates = (
        _preference_candidates(
            target_component=(
                curriculum.target_component
            ),
            curriculum=curriculum,
            state=state,
            trajectories_by_id=(
                trajectories_by_id
            ),
            promoted_records=(
                promoted_records
            ),
            exclusions=exclusions,
        )
        + _hub_preference_candidates(
            curriculum=curriculum,
            state=state,
            promoted_records=(
                hub_promoted_records
            ),
            exclusions=exclusions,
        )
        + _curriculum_lesson_candidates(
            target_component=(
                curriculum.target_component
            ),
            state=state,
            promoted_records=(
                lesson_promoted_records
            ),
            exclusions=exclusions,
        )
        + _code_candidates(
            target_component=(
                curriculum.target_component
            ),
            state=state,
            corpora=corpora,
            exclusions=exclusions,
        )
    )

    (
        candidates,
        code_dropped,
    ) = cap_code_candidates(
        candidates,
        max_code_members=(
            max_code_members
        ),
        max_repository_fraction=(
            max_repository_fraction
        ),
        seed=seed,
    )

    if code_dropped:
        exclusions[
            "code_member_cap"
        ] += code_dropped

    members = assign_partitions(
        candidates,
        validation_fraction=(
            validation_fraction
        ),
        seed=seed,
    )

    current_chapter = (
        curriculum.chapter(
            state.current_chapter_id
        )
        if state.current_chapter_id
        is not None
        else None
    )

    hard_cases = (
        _hard_cases_for_current_chapter(
            target_component=(
                curriculum.target_component
            ),
            chapter=current_chapter,
            diagnoses=diagnoses,
            hub_records=hub_records,
        )
        if current_chapter
        is not None
        else []
    )

    current_behavior_count = sum(
        1
        for item in members
        if (
            item.role
            == "current_behavior"
        )
    )

    current_lesson_count = sum(
        1
        for item in members
        if (
            item.role
            == "current_lesson"
        )
    )

    current_training_example_count = (
        current_behavior_count
        + current_lesson_count
    )

    train_count = sum(
        1
        for item in members
        if item.partition
        == "train"
    )

    validation_count = sum(
        1
        for item in members
        if item.partition
        == "validation"
    )

    required_reviewed = (
        current_chapter
        .min_reviewed_examples
        if current_chapter
        is not None
        else 0
    )

    checks = {
        "current_chapter_active":
            (
                current_chapter
                is not None
            ),

        "minimum_reviewed_current_examples":
            (
                current_training_example_count
                >= required_reviewed
                if current_chapter
                is not None
                else False
            ),

        "train_members_present":
            train_count > 0,

        "validation_members_present":
            validation_count > 0,

        "sources_cryptographically_pinned":
            all(
                bool(
                    value
                )
                for value in (
                    source_fingerprints
                    .values()
                )
            ),

        "hub_provenance_ledger_available":
            (
                bool(
                    hub_records
                )
                if curriculum.target_component
                == "hub"
                else True
            ),
    }

    blocked_reasons = [
        name
        for (
            name,
            passed,
        ) in checks.items()
        if not passed
    ]

    identity = {
        "curriculum_id":
            curriculum.curriculum_id,

        "curriculum_version":
            curriculum.version,

        "target_component":
            curriculum.target_component,

        "state_sha256":
            source_fingerprints[
                _relative_artifact_path(
                    resolved_state_path
                )
            ],

        "member_ids": [
            item.member_id
            for item in members
        ],

        "hard_case_ids": [
            item.diagnosis_id
            for item in hard_cases
        ],

        "source_fingerprints":
            source_fingerprints,

        "validation_fraction":
            validation_fraction,

        "seed":
            seed,

        "max_code_members":
            max_code_members,

        "max_repository_fraction":
            max_repository_fraction,

        "algorithm":
            "curriculum-training-membership-v1",
    }

    plan_id = (
        _stable_id(
            "training-plan",
            identity,
        )
    )

    target_directory = (
        output_root
        .expanduser()
        .resolve()
        / curriculum.target_component
        / plan_id
    )

    manifest_path = (
        target_directory
        / "manifest.json"
    )

    if manifest_path.is_file():
        manifest = (
            TrainingMembershipManifest
            .model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        return TrainingMembershipBuildResult(
            manifest=manifest,
            output_directory=str(
                target_directory
            ),
        )

    target_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    membership_sha256 = (
        immutable_write_jsonl(
            target_directory
            / "members.jsonl",
            members,
        )
    )

    hard_cases_sha256 = (
        immutable_write_jsonl(
            target_directory
            / "hard-cases.jsonl",
            hard_cases,
        )
    )

    manifest = (
        TrainingMembershipManifest(
            plan_id=plan_id,
            created_at=_utc_now(),
            target_component=(
                curriculum.target_component
            ),
            curriculum_id=(
                curriculum.curriculum_id
            ),
            curriculum_version=(
                curriculum.version
            ),
            current_chapter_id=(
                state.current_chapter_id
            ),
            current_chapter_title=(
                current_chapter.title
                if current_chapter
                is not None
                else None
            ),
            mastered_chapter_ids=list(
                state.mastered_chapter_ids
            ),
            seed=seed,
            validation_fraction=(
                validation_fraction
            ),
            source_fingerprints=dict(
                sorted(
                    source_fingerprints.items()
                )
            ),
            member_count=len(
                members
            ),
            train_member_count=(
                train_count
            ),
            validation_member_count=(
                validation_count
            ),
            member_kind_counts=dict(
                Counter(
                    item.member_kind
                    for item in members
                )
            ),
            role_counts=dict(
                Counter(
                    item.role
                    for item in members
                )
            ),
            partition_counts=dict(
                Counter(
                    item.partition
                    for item in members
                )
            ),
            reviewed_current_behavior_count=(
                current_behavior_count
            ),
            reviewed_current_lesson_count=(
                current_lesson_count
            ),
            reviewed_current_training_example_count=(
                current_training_example_count
            ),
            required_reviewed_current_behavior_count=(
                required_reviewed
            ),
            hard_case_count=len(
                hard_cases
            ),
            exact_provenance_hard_case_count=sum(
                1
                for item in hard_cases
                if item.exact_hub_provenance_ready
            ),
            hard_case_failure_counts=dict(
                Counter(
                    item.failure_code
                    for item in hard_cases
                )
            ),
            excluded_counts=dict(
                sorted(
                    exclusions.items()
                )
            ),
            membership_sha256=(
                membership_sha256
            ),
            hard_cases_sha256=(
                hard_cases_sha256
            ),
            checks=checks,
            blocked_reasons=(
                blocked_reasons
            ),
            readiness_ready=(
                all(
                    checks.values()
                )
            ),
            training_authorized=False,
            promotion_authorized=False,
        )
    )

    immutable_write_json(
        manifest_path,
        manifest,
    )

    return (
        TrainingMembershipBuildResult(
            manifest=manifest,
            output_directory=str(
                target_directory
            ),
        )
    )
