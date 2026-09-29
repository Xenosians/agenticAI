from __future__ import annotations

import gc
import hashlib
import json
import shutil
import tempfile

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from config.path_portability import (
    resolve_portable_path,
)

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)

from learning.continual.corpus_decontamination import (
    stable_identity,
)

from learning.continual.corpus_materializer import (
    CORPUS_SNAPSHOT_ROOT,
    CorpusSnapshotManifest,
    NormalizedCorpusRecord,
)

from learning.continual.storage import (
    canonical_json,
    fingerprint_directory,
    immutable_write_json,
    load_jsonl_models,
    sha256_file,
)

from learning.evaluation.developer_holdout import (
    DeveloperHoldoutManifest,
    DeveloperHoldoutRecord,
)

from learning.paths import (
    EVALUATIONS_ROOT,
)

from learning.training.phase5_hybrid_qlora import (
    Phase5TrainingRunManifest,
    Phase5TrainingSettings,
    _dtype,
    _fallback_ids,
    _input_device,
    _load_model,
    _template_ids,
)


DEFAULT_DEVELOPER_HOLDOUT_EVAL_ROOT = (
    EVALUATIONS_ROOT
    / "developer-heldout-runs"
)


class DeveloperHoldoutCaseResult(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="developer-heldout-case.v1",
        alias="schema",
    )

    record_id: str

    source_record_id: (
        str
        | None
    ) = None

    sequence_tokens: int

    base_loss: float
    candidate_loss: float

    loss_improvement: float

    candidate_better: bool


class DeveloperHoldoutEvaluationReport(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="developer-heldout-evaluation.v1",
        alias="schema",
    )

    evaluation_id: str
    created_at: str

    checkpoint_id: str
    adapter_sha256: str

    holdout_id: str
    holdout_records_sha256: str

    source_id: str
    revision: str | None

    target_component: str
    target_model_key: str

    # Evaluation may use a larger forward-only context than the
    # candidate's original training sequence budget. Keep both
    # values explicit so future readers cannot confuse them.
    training_max_sequence_tokens: int

    max_sequence_tokens: int

    holdout_count: int
    evaluated_count: int

    excluded_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    base_mean_loss: float
    candidate_mean_loss: float

    absolute_loss_improvement: float
    relative_loss_improvement_percent: float

    candidate_better_cases: int
    base_better_cases: int
    tied_cases: int

    lineage_verified: bool
    training_overlap_count: int

    base_model_sha256_before: str
    base_model_sha256_after: str
    base_model_unchanged: bool

    loss_improved: bool

    behavioral_evaluation_complete: bool = False
    promotion_authorized: bool = False

    case_results: list[
        DeveloperHoldoutCaseResult
    ] = Field(
        default_factory=list
    )

    output_directory: str


def _utc_now() -> str:
    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _sha256_text(
    value: str,
) -> str:
    return (
        hashlib
        .sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _increment(
    values: dict[str, int],
    key: str,
) -> None:
    values[key] = (
        values.get(
            key,
            0,
        )
        + 1
    )


def _load_holdout(
    directory: Path,
) -> tuple[
    DeveloperHoldoutManifest,
    list[DeveloperHoldoutRecord],
    str,
]:

    directory = (
        resolve_portable_path(
            directory
        )
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
            "Developer holdout manifest does not exist."
        )

    if not records_path.is_file():
        raise ValueError(
            "Developer holdout records do not exist."
        )

    manifest = (
        DeveloperHoldoutManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        manifest.evaluation_only
        is not True

        or manifest.training_eligible
        is not False

        or manifest.training_authorized
        is not False

        or manifest.promotion_authorized
        is not False
    ):
        raise PermissionError(
            "Developer holdout governance flags are invalid."
        )

    observed_sha = (
        sha256_file(
            records_path
        )
    )

    if (
        observed_sha
        != manifest.records_sha256
    ):
        raise ValueError(
            "Developer holdout records SHA-256 mismatch."
        )

    records = (
        load_jsonl_models(
            records_path,
            DeveloperHoldoutRecord,
        )
    )

    if (
        len(records)
        != manifest.holdout_count
    ):
        raise ValueError(
            "Developer holdout record count does not "
            "match its manifest."
        )

    manifest_sha = (
        sha256_file(
            manifest_path
        )
    )

    if not manifest_sha:
        raise ValueError(
            "Could not fingerprint developer holdout manifest."
        )

    return (
        manifest,
        records,
        manifest_sha,
    )


def _load_training_run(
    checkpoint,
) -> tuple[
    Phase5TrainingRunManifest,
    Path,
    str,
]:

    adapter_directory = (
        resolve_portable_path(
            checkpoint.adapter_directory
        )
    )

    run_root = (
        adapter_directory
        .parent
    )

    manifest_path = (
        run_root
        / "manifest.json"
    )

    if not manifest_path.is_file():
        raise ValueError(
            "Candidate training run manifest does not exist."
        )

    manifest = (
        Phase5TrainingRunManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        manifest.registered_checkpoint_id
        != checkpoint.checkpoint_id
    ):
        raise ValueError(
            "Checkpoint registration does not match "
            "the training run manifest."
        )

    if (
        manifest.materialization_id
        != checkpoint.source_split_id
    ):
        raise ValueError(
            "Checkpoint source split does not match "
            "the training run."
        )

    if (
        manifest.target_component
        != checkpoint.target_agent
        or manifest.target_model_key
        != checkpoint.target_model_key
    ):
        raise ValueError(
            "Checkpoint target does not match "
            "the training run."
        )

    if (
        manifest.base_model_sha256_before
        != checkpoint.base_model_sha256
        or manifest.base_model_sha256_after
        != checkpoint.base_model_sha256
    ):
        raise ValueError(
            "Checkpoint base-model fingerprint does not "
            "match the training run."
        )

    manifest_sha = (
        sha256_file(
            manifest_path
        )
    )

    if not manifest_sha:
        raise ValueError(
            "Could not fingerprint training run manifest."
        )

    return (
        manifest,
        run_root,
        manifest_sha,
    )


def _find_training_snapshot(
    snapshot_id: str,
) -> tuple[
    CorpusSnapshotManifest,
    list[NormalizedCorpusRecord],
]:

    matches = []

    if (
        CORPUS_SNAPSHOT_ROOT
        .is_dir()
    ):

        for manifest_path in (
            CORPUS_SNAPSHOT_ROOT
            .glob(
                "*/*/manifest.json"
            )
        ):

            try:

                manifest = (
                    CorpusSnapshotManifest
                    .model_validate_json(
                        manifest_path
                        .read_text(
                            encoding="utf-8"
                        )
                    )
                )

            except Exception:
                continue

            if (
                manifest.snapshot_id
                == snapshot_id
            ):
                matches.append(
                    (
                        manifest,
                        manifest_path,
                    )
                )

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one training corpus snapshot "
            f"for {snapshot_id!r}; found {len(matches)}."
        )

    manifest, manifest_path = (
        matches[0]
    )

    records_path = (
        manifest_path
        .parent
        / "records.jsonl"
    )

    observed_sha = (
        sha256_file(
            records_path
        )
    )

    if (
        observed_sha
        != manifest.records_sha256
    ):
        raise ValueError(
            "Training corpus snapshot records SHA-256 mismatch."
        )

    records = (
        load_jsonl_models(
            records_path,
            NormalizedCorpusRecord,
        )
    )

    if (
        len(records)
        != manifest.accepted_count
    ):
        raise ValueError(
            "Training corpus snapshot count mismatch."
        )

    return (
        manifest,
        records,
    )


def _normalized_training_identity(
    record: NormalizedCorpusRecord,
) -> str:

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

    return (
        stable_identity(
            source_record_id=(
                record.source_record_id
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

            content_sha256=(
                record.content_sha256
            ),
        )
    )


def _overlap_identities(
    *,
    training_records: list[
        NormalizedCorpusRecord
    ],
    holdout_records: list[
        DeveloperHoldoutRecord
    ],
) -> list[str]:

    training = {
        _normalized_training_identity(
            record
        )

        for record
        in training_records
    }

    holdout = {
        record.source_identity

        for record
        in holdout_records
    }

    return sorted(
        training
        & holdout
    )


def _verify_lineage(
    *,
    checkpoint,
    holdout_manifest: DeveloperHoldoutManifest,
    holdout_records: list[
        DeveloperHoldoutRecord
    ],
) -> None:

    (
        training_snapshot,
        training_records,
    ) = (
        _find_training_snapshot(
            checkpoint.source_cycle_id
        )
    )

    if (
        training_snapshot.source_id
        != holdout_manifest.source_id
    ):
        raise ValueError(
            "Training and holdout sources differ."
        )

    if (
        training_snapshot.revision
        != holdout_manifest.revision
    ):
        raise ValueError(
            "Training and holdout revisions differ."
        )

    if (
        training_snapshot
        .decontamination_policy_sha256
        != holdout_manifest
        .decontamination_policy_sha256
    ):
        raise ValueError(
            "Training and holdout decontamination "
            "policy fingerprints differ."
        )

    overlap = (
        _overlap_identities(
            training_records=(
                training_records
            ),
            holdout_records=(
                holdout_records
            ),
        )
    )

    if overlap:
        raise PermissionError(
            "Training/holdout contamination detected: "
            + ", ".join(
                overlap[:5]
            )
        )


class _SequenceExcluded(
    ValueError
):
    pass


def _encode_complete_holdout(
    *,
    tokenizer,
    record: DeveloperHoldoutRecord,
    max_length: int,
):

    full_messages = [
        *record.prompt_messages,
        {
            "role":
                "assistant",

            "content":
                record.reference_completion,
        },
    ]

    try:

        prompt_ids = (
            _template_ids(
                tokenizer,
                record.prompt_messages,
                add_generation_prompt=True,
            )
        )

        full_ids = (
            _template_ids(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
            )
        )

    except Exception:

        prompt_ids = (
            _fallback_ids(
                tokenizer,
                record.prompt_messages,
                add_generation_prompt=True,
            )
        )

        full_ids = (
            _fallback_ids(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
            )
        )

    prompt_ids = (
        prompt_ids
        .flatten()
    )

    full_ids = (
        full_ids
        .flatten()
    )

    common = 0

    prefix_limit = min(
        int(
            prompt_ids.numel()
        ),
        int(
            full_ids.numel()
        ),
    )

    while (
        common
        < prefix_limit

        and int(
            prompt_ids[
                common
            ]
        )
        == int(
            full_ids[
                common
            ]
        )
    ):
        common += 1

    if common <= 0:
        raise _SequenceExcluded(
            "encoding_failed"
        )

    if common >= max_length:
        raise _SequenceExcluded(
            "prompt_too_long"
        )

    full_length = int(
        full_ids.numel()
    )

    if (
        full_length
        > max_length
    ):
        raise _SequenceExcluded(
            "full_sequence_too_long"
        )

    if (
        full_length
        <= common
    ):
        raise _SequenceExcluded(
            "no_completion_tokens"
        )

    labels = (
        full_ids
        .clone()
    )

    labels[
        :common
    ] = -100

    attention_mask = (
        full_ids
        .new_ones(
            full_ids.shape
        )
    )

    return (
        full_ids.unsqueeze(0),
        attention_mask.unsqueeze(0),
        labels.unsqueeze(0),
        full_length,
    )


def _encoded_loss(
    *,
    loaded,
    encoded,
) -> float:

    (
        input_ids,
        attention_mask,
        labels,
        _sequence_tokens,
    ) = encoded

    device = (
        _input_device(
            loaded.model
        )
    )

    input_ids = (
        input_ids
        .to(
            device
        )
    )

    attention_mask = (
        attention_mask
        .to(
            device
        )
    )

    labels = (
        labels
        .to(
            device
        )
    )

    output = (
        loaded.model(
            input_ids=(
                input_ids
            ),

            attention_mask=(
                attention_mask
            ),

            labels=(
                labels
            ),

            use_cache=False,
        )
    )

    return float(
        output
        .loss
        .detach()
        .float()
        .item()
    )


def _relative_improvement_percent(
    *,
    base_loss: float,
    candidate_loss: float,
) -> float:

    if base_loss == 0:
        return 0.0

    return (
        (
            base_loss
            - candidate_loss
        )
        / abs(
            base_loss
        )
        * 100.0
    )


def evaluate_developer_holdout(
    *,
    checkpoint_id: str,
    holdout_directory: Path,
    backend: str,
    max_sequence_tokens: int | None = None,
    output_root: Path = (
        DEFAULT_DEVELOPER_HOLDOUT_EVAL_ROOT
    ),
) -> DeveloperHoldoutEvaluationReport:
    """
    Compare one registered candidate adapter against its frozen
    base model on a deterministic evaluation-only developer holdout.

    No optimizer is created.
    No gradients are computed.
    No checkpoint is activated.
    No promotion decision is created.
    """

    checkpoint_store = (
        AdapterCheckpointStore()
    )

    checkpoint = (
        checkpoint_store
        .verify_adapter(
            checkpoint_id
        )
    )

    if (
        checkpoint.target_agent
        != "developer-specialist"
    ):
        raise ValueError(
            "Developer holdout evaluation requires a "
            "developer-specialist checkpoint."
        )

    (
        holdout_manifest,
        holdout_records,
        holdout_manifest_sha,
    ) = (
        _load_holdout(
            holdout_directory
        )
    )

    if (
        holdout_manifest.target_component
        != "developer-specialist"
    ):
        raise ValueError(
            "Holdout target is not developer-specialist."
        )

    (
        training_run,
        _training_run_root,
        training_run_manifest_sha,
    ) = (
        _load_training_run(
            checkpoint
        )
    )

    _verify_lineage(
        checkpoint=checkpoint,
        holdout_manifest=(
            holdout_manifest
        ),
        holdout_records=(
            holdout_records
        ),
    )

    settings_payload = (
        training_run.settings
    )

    training_max_length = int(
        settings_payload[
            "max_length"
        ]
    )

    if max_sequence_tokens is None:
        max_length = (
            training_max_length
        )

    else:
        max_length = int(
            max_sequence_tokens
        )

    if max_length < 128:
        raise ValueError(
            "Heldout evaluation max_sequence_tokens "
            "must be at least 128."
        )

    compute_dtype = str(
        settings_payload[
            "compute_dtype"
        ]
    )

    recipe = (
        Phase5TrainingSettings(
            max_length=(
                max_length
            ),
            compute_dtype=(
                compute_dtype
            ),
        )
    )

    base_model_path = (
        resolve_portable_path(
            training_run
            .base_model_path
        )
    )

    base_before = (
        fingerprint_directory(
            base_model_path
        )
    )

    if (
        base_before
        != checkpoint.base_model_sha256
    ):
        raise ValueError(
            "Frozen base model does not match "
            "the candidate checkpoint."
        )

    identity = {
        "checkpoint_id":
            checkpoint.checkpoint_id,

        "adapter_sha256":
            checkpoint.adapter_sha256,

        "holdout_id":
            holdout_manifest.holdout_id,

        "holdout_records_sha256":
            holdout_manifest.records_sha256,

        "holdout_manifest_sha256":
            holdout_manifest_sha,

        "training_run_manifest_sha256":
            training_run_manifest_sha,

        "base_model_sha256":
            base_before,

        "training_max_sequence_tokens":
            training_max_length,

        "evaluation_max_sequence_tokens":
            max_length,

        "compute_dtype":
            compute_dtype,
    }

    evaluation_id = (
        "developer-heldout-eval-"
        + _sha256_text(
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

    output_directory = (
        root
        / evaluation_id
    )

    report_path = (
        output_directory
        / "report.json"
    )

    if report_path.is_file():

        return (
            DeveloperHoldoutEvaluationReport
            .model_validate_json(
                report_path
                .read_text(
                    encoding="utf-8"
                )
            )
        )

    loaded = None

    case_results: list[
        DeveloperHoldoutCaseResult
    ] = []

    excluded_counts: dict[
        str,
        int
    ] = {}

    try:

        loaded = (
            _load_model(
                model_path=(
                    base_model_path
                ),

                backend=backend,

                settings=recipe,

                seed_adapter_directory=(
                    resolve_portable_path(
                        checkpoint
                        .adapter_directory
                    )
                ),
            )
        )

        torch = (
            loaded.torch
        )

        loaded.model.eval()

        dtype = (
            _dtype(
                torch,
                compute_dtype,
            )
        )

        disable_adapter = getattr(
            loaded.model,
            "disable_adapter",
            None,
        )

        if not callable(
            disable_adapter
        ):
            raise RuntimeError(
                "Candidate model does not expose "
                "disable_adapter(); frozen-base comparison "
                "cannot be performed safely."
            )

        with (
            torch.inference_mode()
        ):

            for record in holdout_records:

                try:

                    encoded = (
                        _encode_complete_holdout(
                            tokenizer=(
                                loaded.tokenizer
                            ),

                            record=record,

                            max_length=(
                                max_length
                            ),
                        )
                    )

                except _SequenceExcluded as exc:

                    _increment(
                        excluded_counts,
                        str(
                            exc
                        ),
                    )

                    continue

                with (
                    torch.autocast(
                        device_type="cuda",
                        dtype=dtype,
                    )
                ):

                    with disable_adapter():

                        base_loss = (
                            _encoded_loss(
                                loaded=loaded,
                                encoded=(
                                    encoded
                                ),
                            )
                        )

                    candidate_loss = (
                        _encoded_loss(
                            loaded=loaded,
                            encoded=(
                                encoded
                            ),
                        )
                    )

                loss_improvement = (
                    base_loss
                    - candidate_loss
                )

                case_results.append(
                    DeveloperHoldoutCaseResult(
                        record_id=(
                            record.record_id
                        ),

                        source_record_id=(
                            record
                            .source_record_id
                        ),

                        sequence_tokens=(
                            encoded[3]
                        ),

                        base_loss=(
                            base_loss
                        ),

                        candidate_loss=(
                            candidate_loss
                        ),

                        loss_improvement=(
                            loss_improvement
                        ),

                        candidate_better=(
                            loss_improvement
                            > 0
                        ),
                    )
                )

                del encoded

                if (
                    torch.cuda
                    .is_available()
                ):
                    torch.cuda.empty_cache()

        if not case_results:
            raise RuntimeError(
                "No reserved holdout records fit the "
                "candidate training sequence budget."
            )

        base_mean = (
            sum(
                item.base_loss

                for item
                in case_results
            )
            / len(
                case_results
            )
        )

        candidate_mean = (
            sum(
                item.candidate_loss

                for item
                in case_results
            )
            / len(
                case_results
            )
        )

        absolute_improvement = (
            base_mean
            - candidate_mean
        )

        relative_improvement = (
            _relative_improvement_percent(
                base_loss=(
                    base_mean
                ),
                candidate_loss=(
                    candidate_mean
                ),
            )
        )

        epsilon = 1e-9

        candidate_better_cases = sum(
            1

            for item
            in case_results

            if (
                item.loss_improvement
                > epsilon
            )
        )

        base_better_cases = sum(
            1

            for item
            in case_results

            if (
                item.loss_improvement
                < -epsilon
            )
        )

        tied_cases = (
            len(
                case_results
            )
            - candidate_better_cases
            - base_better_cases
        )

        base_after = (
            fingerprint_directory(
                base_model_path
            )
        )

        if (
            base_after
            != base_before
        ):
            raise RuntimeError(
                "Frozen base model changed during "
                "heldout evaluation."
            )

        report = (
            DeveloperHoldoutEvaluationReport(
                evaluation_id=(
                    evaluation_id
                ),

                created_at=(
                    _utc_now()
                ),

                checkpoint_id=(
                    checkpoint
                    .checkpoint_id
                ),

                adapter_sha256=(
                    checkpoint
                    .adapter_sha256
                ),

                holdout_id=(
                    holdout_manifest
                    .holdout_id
                ),

                holdout_records_sha256=(
                    holdout_manifest
                    .records_sha256
                ),

                source_id=(
                    holdout_manifest
                    .source_id
                ),

                revision=(
                    holdout_manifest
                    .revision
                ),

                target_component=(
                    checkpoint
                    .target_agent
                ),

                target_model_key=(
                    checkpoint
                    .target_model_key
                ),

                training_max_sequence_tokens=(
                    training_max_length
                ),

                max_sequence_tokens=(
                    max_length
                ),

                holdout_count=(
                    holdout_manifest
                    .holdout_count
                ),

                evaluated_count=len(
                    case_results
                ),

                excluded_counts=dict(
                    sorted(
                        excluded_counts
                        .items()
                    )
                ),

                base_mean_loss=(
                    base_mean
                ),

                candidate_mean_loss=(
                    candidate_mean
                ),

                absolute_loss_improvement=(
                    absolute_improvement
                ),

                relative_loss_improvement_percent=(
                    relative_improvement
                ),

                candidate_better_cases=(
                    candidate_better_cases
                ),

                base_better_cases=(
                    base_better_cases
                ),

                tied_cases=(
                    tied_cases
                ),

                lineage_verified=True,

                training_overlap_count=0,

                base_model_sha256_before=(
                    base_before
                ),

                base_model_sha256_after=(
                    base_after
                ),

                base_model_unchanged=True,

                loss_improved=(
                    candidate_mean
                    < base_mean
                ),

                behavioral_evaluation_complete=False,

                promotion_authorized=False,

                case_results=(
                    case_results
                ),

                output_directory=str(
                    output_directory
                ),
            )
        )

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = Path(
            tempfile.mkdtemp(
                prefix=".developer-heldout-eval-",
                dir=root,
            )
        )

        try:

            immutable_write_json(
                temporary
                / "report.json",
                report,
            )

            (
                os_replace_target
            ) = output_directory

            if (
                os_replace_target
                .exists()
            ):
                shutil.rmtree(
                    temporary,
                    ignore_errors=True,
                )

            else:
                temporary.replace(
                    os_replace_target
                )

        except Exception:

            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )

            raise

        return report

    finally:

        if loaded is not None:

            try:
                del loaded.model
            except Exception:
                pass

            try:
                del loaded.tokenizer
            except Exception:
                pass

        gc.collect()

        try:

            import torch

            if (
                torch.cuda
                .is_available()
            ):
                torch.cuda.empty_cache()

        except Exception:
            pass
