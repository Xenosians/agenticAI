from __future__ import annotations

import hashlib
import os
import shutil
import tempfile

from pathlib import Path
from typing import Callable

from learning.continual.corpus_materializer import (
    CorpusSnapshotManifest,
    NormalizedCorpusRecord,
)

from learning.continual.storage import (
    fingerprint_directory,
    immutable_write_json,
    immutable_write_jsonl,
    load_jsonl_models,
    sha256_file,
)

from learning.developer_contract import (
    build_developer_issue_messages,
)

from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)

from learning.training.hub_training_contracts import (
    canonical_json,
    sha256_text,
)

from learning.training.hub_training_materializer import (
    HubDpoRecord,
    HubTrainingMaterializationManifest,
    HubTrainingMaterializationResult,
    HubSftRecord,
)


DEFAULT_DEVELOPER_CORPUS_MATERIALIZATION_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "developer-sft-materialized"
)


def _required_sha(
    path: Path,
) -> str:

    value = sha256_file(
        path
    )

    if not value:
        raise ValueError(
            f"Could not fingerprint: {path}"
        )

    return value


def _load_snapshot(
    directory: Path,
) -> tuple[
    CorpusSnapshotManifest,
    list[NormalizedCorpusRecord],
    str,
]:

    directory = (
        directory
        .expanduser()
        .resolve()
    )

    manifest_path = (
        directory
        / "manifest.json"
    )

    records_path = (
        directory
        / "records.jsonl"
    )

    if not manifest_path.is_file():
        raise ValueError(
            "Corpus snapshot manifest does not exist."
        )

    if not records_path.is_file():
        raise ValueError(
            "Corpus snapshot records do not exist."
        )

    manifest = (
        CorpusSnapshotManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    records_sha = (
        _required_sha(
            records_path
        )
    )

    if (
        records_sha
        != manifest.records_sha256
    ):
        raise ValueError(
            "Corpus snapshot records SHA-256 mismatch."
        )

    records = load_jsonl_models(
        records_path,
        NormalizedCorpusRecord,
    )

    if (
        len(
            records
        )
        != manifest.accepted_count
    ):
        raise ValueError(
            "Corpus snapshot accepted_count does not "
            "match records.jsonl."
        )

    return (
        manifest,
        records,
        _required_sha(
            manifest_path
        ),
    )


def _validate_snapshot(
    manifest: CorpusSnapshotManifest,
) -> None:

    if (
        manifest.target_component
        != "developer-specialist"
    ):
        raise ValueError(
            "Developer SFT bridge requires "
            "target_component=developer-specialist."
        )

    if (
        "sft"
        not in manifest.objectives
    ):
        raise ValueError(
            "Developer SFT bridge requires an SFT corpus."
        )

    if not manifest.revision:
        raise PermissionError(
            "Developer corpus revision is not pinned."
        )

    if not (
        manifest
        .decontamination_policy_sha256
    ):
        raise PermissionError(
            "Developer corpus has no decontamination policy fingerprint."
        )

    # Snapshots created immediately before this bridge was wired
    # legitimately contain this one historical blocker.
    allowed_historical_blockers = {
        "objective_specific_optimizer_bridge_not_wired",
    }

    unexpected = [
        blocker

        for blocker
        in manifest.training_blockers

        if blocker
        not in allowed_historical_blockers
    ]

    if unexpected:
        raise PermissionError(
            "Developer corpus has unresolved training blockers: "
            + ", ".join(
                sorted(
                    unexpected
                )
            )
        )


def _selection_key(
    record: NormalizedCorpusRecord,
) -> str:

    return hashlib.sha256(
        (
            record.record_id
            + ":"
            + record.content_sha256
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _validation_record(
    record: NormalizedCorpusRecord,
) -> bool:

    digest = hashlib.sha256(
        (
            "developer-validation:"
            + record.record_id
        ).encode(
            "utf-8"
        )
    ).digest()

    # Deterministic ~20% bridge-level validation partition.
    #
    # This is NOT the reserved external benchmark holdout and therefore
    # must never be treated as sufficient evidence for promotion.
    return (
        int.from_bytes(
            digest[:8],
            "big",
        )
        % 5
        == 0
    )


def _developer_training_messages(
    record: NormalizedCorpusRecord,
) -> list[dict[str, str]]:

    if (
        record.metadata.get(
            "adapter"
        )
        != "issue_patch"
    ):
        return record.prompt_messages

    problem_statement = (
        record.metadata.get(
            "problem_statement"
        )
    )

    if not (
        isinstance(
            problem_statement,
            str,
        )
        and problem_statement.strip()
    ):
        return record.prompt_messages

    repository = (
        record.metadata.get(
            "repository"
        )
    )

    base_commit = (
        record.metadata.get(
            "base_commit"
        )
    )

    return build_developer_issue_messages(
        problem_statement=(
            problem_statement
        ),
        repository=(
            repository
            if isinstance(
                repository,
                str,
            )
            else None
        ),
        base_commit=(
            base_commit
            if isinstance(
                base_commit,
                str,
            )
            else None
        ),
    )


def _phase5_record(
    *,
    snapshot: CorpusSnapshotManifest,
    record: NormalizedCorpusRecord,
    partition: str,
) -> HubSftRecord:

    if not (
        record.objective
        == "sft"
        and record.prompt_messages
        and record.chosen
    ):
        raise ValueError(
            "Normalized developer record is not valid SFT material."
        )

    identity = {
        "snapshot_id":
            snapshot.snapshot_id,

        "record_id":
            record.record_id,

        "partition":
            partition,

        "target":
            "developer-specialist",

        "content_sha256":
            record.content_sha256,
    }

    record_id = (
        "developer-sft-"
        + sha256_text(
            canonical_json(
                identity
            )
        )[:24]
    )

    source_identity = (
        record.source_record_id
        or str(
            record.upstream_index
        )
    )

    return HubSftRecord(
        record_id=record_id,

        member_id=(
            "corpus:"
            + snapshot.snapshot_id
            + ":"
            + record.record_id
        ),

        partition=partition,

        role="developer-specialist",

        source_kind=(
            "external_developer_corpus"
        ),

        prompt_messages=(
            _developer_training_messages(
                record
            )
        ),

        chosen=(
            record.chosen
        ),

        source_id=(
            record.record_id
        ),

        source_artifact=(
            snapshot.snapshot_id
        ),

        source_artifact_sha256=(
            snapshot.records_sha256
        ),

        lineage_id=(
            "external-corpus:"
            + snapshot.source_id
            + ":"
            + str(
                snapshot.revision
            )
            + ":"
            + source_identity
        ),
    )


def materialize_developer_sft_snapshot(
    *,
    snapshot_directory: Path,
    target_model_key: str,
    base_model_path: Path,
    target_contract_path: Path,
    token_length_resolver: Callable[
        [NormalizedCorpusRecord],
        tuple[int, int, int],
    ],
    max_sequence_tokens: int = 1024,
    max_records: int = 32,
    output_root: Path = (
        DEFAULT_DEVELOPER_CORPUS_MATERIALIZATION_ROOT
    ),
) -> HubTrainingMaterializationResult:

    if max_records < 2:
        raise ValueError(
            "Developer SFT bridge requires at least 2 records."
        )

    if max_sequence_tokens < 128:
        raise ValueError(
            "max_sequence_tokens must be at least 128."
        )

    (
        snapshot,
        all_records,
        snapshot_manifest_sha,
    ) = _load_snapshot(
        snapshot_directory
    )

    _validate_snapshot(
        snapshot
    )

    eligible = [
        record

        for record
        in all_records

        if (
            record.objective
            == "sft"
            and record.target_component
            == "developer-specialist"
            and record.prompt_messages
            and record.chosen
        )
    ]

    sequence_rejected = {
        "bridge_sequence_encoding_failed":
            0,

        "bridge_prompt_too_long":
            0,

        "bridge_full_sequence_too_long":
            0,

        "bridge_no_completion_tokens":
            0,
    }

    sequence_eligible = []

    for record in sorted(
        eligible,
        key=_selection_key,
    ):

        try:
            (
                prompt_tokens,
                full_tokens,
                common_prefix_tokens,
            ) = token_length_resolver(
                record
            )

        except Exception:
            sequence_rejected[
                "bridge_sequence_encoding_failed"
            ] += 1

            continue

        if common_prefix_tokens <= 0:

            sequence_rejected[
                "bridge_sequence_encoding_failed"
            ] += 1

            continue

        # Match the trainer's real failure condition.
        if (
            common_prefix_tokens
            >= max_sequence_tokens
        ):

            sequence_rejected[
                "bridge_prompt_too_long"
            ] += 1

            continue

        # Never truncate a verified patch.
        #
        # Partial diffs are not valid SFT targets.
        if (
            full_tokens
            > max_sequence_tokens
        ):

            sequence_rejected[
                "bridge_full_sequence_too_long"
            ] += 1

            continue

        if (
            full_tokens
            <= common_prefix_tokens
        ):

            sequence_rejected[
                "bridge_no_completion_tokens"
            ] += 1

            continue

        sequence_eligible.append(
            record
        )

    if len(
        sequence_eligible
    ) < 2:
        raise ValueError(
            "Developer snapshot contains fewer than two "
            f"complete examples within {max_sequence_tokens} "
            "tokens. Materialize more corpus rows instead "
            "of truncating patches."
        )

    selected = (
        sequence_eligible[
            :max_records
        ]
    )

    validation_records = [
        record

        for record
        in selected

        if _validation_record(
            record
        )
    ]

    training_records = [
        record

        for record
        in selected

        if record
        not in validation_records
    ]

    # Deterministically guarantee both partitions for tiny snapshots.
    if not validation_records:

        validation_records = [
            selected[-1]
        ]

        training_records = (
            selected[:-1]
        )

    if not training_records:

        training_records = [
            selected[0]
        ]

        validation_records = (
            selected[1:]
        )

    if not validation_records:
        raise ValueError(
            "Developer bridge could not construct "
            "a validation partition."
        )

    base_model_path = (
        base_model_path
        .expanduser()
        .resolve()
    )

    target_contract_path = (
        target_contract_path
        .expanduser()
        .resolve()
    )

    if not base_model_path.is_dir():
        raise ValueError(
            "Developer base model directory does not exist."
        )

    if not target_contract_path.is_file():
        raise ValueError(
            "Developer specialist definition does not exist."
        )

    base_model_sha = (
        fingerprint_directory(
            base_model_path
        )
    )

    target_contract_sha = (
        _required_sha(
            target_contract_path
        )
    )

    identity = {
        "schema":
            "developer-corpus-bridge.v1",

        "snapshot_id":
            snapshot.snapshot_id,

        "snapshot_records_sha256":
            snapshot.records_sha256,

        "revision":
            snapshot.revision,

        "decontamination_policy_sha256":
            snapshot.decontamination_policy_sha256,

        "target_model_key":
            target_model_key,

        "base_model_sha256":
            base_model_sha,

        "target_contract_sha256":
            target_contract_sha,

        "max_sequence_tokens":
            max_sequence_tokens,

        "sequence_budget_verified":
            True,

        "selected_records":
            [
                record.record_id
                for record
                in selected
            ],
    }

    materialization_id = (
        "developer-sft-materialization-"
        + sha256_text(
            canonical_json(
                identity
            )
        )[:24]
    )

    root = (
        output_root
        .expanduser()
        .resolve()
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        root
        / materialization_id
    )

    # Idempotent rerun.
    if target.is_dir():

        existing = (
            HubTrainingMaterializationManifest
            .model_validate_json(
                (
                    target
                    / "manifest.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
        )

        if (
            existing.materialization_id
            != materialization_id
        ):
            raise ValueError(
                "Existing developer materialization identity mismatch."
            )

        return HubTrainingMaterializationResult(
            manifest=existing,
            output_directory=str(
                target
            ),
        )

    sft_train = [
        _phase5_record(
            snapshot=snapshot,
            record=record,
            partition="train",
        )

        for record
        in training_records
    ]

    sft_validation = [
        _phase5_record(
            snapshot=snapshot,
            record=record,
            partition="validation",
        )

        for record
        in validation_records
    ]

    dpo_train: list[
        HubDpoRecord
    ] = []

    dpo_validation: list[
        HubDpoRecord
    ] = []

    temporary = Path(
        tempfile.mkdtemp(
            prefix=(
                ".developer-sft-"
            ),
            dir=root,
        )
    )

    try:

        sft_train_sha = (
            immutable_write_jsonl(
                temporary
                / "sft-train.jsonl",
                sft_train,
            )
        )

        sft_validation_sha = (
            immutable_write_jsonl(
                temporary
                / "sft-validation.jsonl",
                sft_validation,
            )
        )

        dpo_train_sha = (
            immutable_write_jsonl(
                temporary
                / "dpo-train.jsonl",
                dpo_train,
            )
        )

        dpo_validation_sha = (
            immutable_write_jsonl(
                temporary
                / "dpo-validation.jsonl",
                dpo_validation,
            )
        )

        manifest = (
            HubTrainingMaterializationManifest(
                materialization_id=(
                    materialization_id
                ),

                created_at=(
                    snapshot.created_at
                ),

                target_component=(
                    "developer-specialist"
                ),

                target_model_key=(
                    target_model_key
                ),

                source_plan_id=(
                    snapshot.snapshot_id
                ),

                source_plan_directory=(
                    snapshot.output_directory
                ),

                source_plan_manifest_sha256=(
                    snapshot_manifest_sha
                ),

                source_plan_members_sha256=(
                    snapshot.records_sha256
                ),

                curriculum_id=(
                    "external-corpus:"
                    + snapshot.source_id
                ),

                curriculum_version=(
                    str(
                        snapshot.revision
                    )
                ),

                chapter_id=(
                    "developer-sft:"
                    + snapshot.source_id
                ),

                base_model_path=str(
                    base_model_path
                ),

                base_model_sha256=(
                    base_model_sha
                ),

                # These fields remain for backwards-compatible Phase-5
                # schema shape. They are not trusted for developer eval.
                hub_system_prompt_sha256="",
                hub_capability_catalog_sha256="",
                hub_agent_definitions_sha256="",

                curriculum_prompt_sha256=(
                    target_contract_sha
                ),

                evaluation_contract=(
                    "developer_sft_loss"
                ),

                target_contract_path=str(
                    target_contract_path
                ),

                target_contract_sha256=(
                    target_contract_sha
                ),

                sequence_budget_tokens=(
                    max_sequence_tokens
                ),

                sequence_budget_verified=True,

                source_fingerprint_count=1,

                source_fingerprints_verified=True,

                sft_train_count=len(
                    sft_train
                ),

                sft_validation_count=len(
                    sft_validation
                ),

                dpo_train_count=0,
                dpo_validation_count=0,

                sft_train_sha256=(
                    sft_train_sha
                ),

                sft_validation_sha256=(
                    sft_validation_sha
                ),

                dpo_train_sha256=(
                    dpo_train_sha
                ),

                dpo_validation_sha256=(
                    dpo_validation_sha
                ),

                contract_validated_record_count=0,

                excluded_counts={
                    "source_rejected":
                        sum(
                            snapshot
                            .rejected_counts
                            .values()
                        ),

                    "bridge_unselected":
                        (
                            len(
                                sequence_eligible
                            )
                            - len(
                                selected
                            )
                        ),

                    **sequence_rejected,
                },

                lineage_isolated=True,

                ready_for_training=True,

                # Authorization remains runtime-explicit.
                training_authorized=False,

                # Corpus training can NEVER silently activate.
                promotion_authorized=False,
            )
        )

        immutable_write_json(
            temporary
            / "manifest.json",
            manifest,
        )

        os.replace(
            temporary,
            target,
        )

    except Exception:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

        raise

    return HubTrainingMaterializationResult(
        manifest=manifest,
        output_directory=str(
            target
        ),
    )
