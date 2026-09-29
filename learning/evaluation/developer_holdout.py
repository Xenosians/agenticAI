from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.continual.corpus_decontamination import (
    DEFAULT_DECONTAMINATION_PATH,
    decontamination_reason,
    load_decontamination_manifest,
    policy_sha256,
    stable_identity,
)

from learning.continual.corpus_materializer import (
    NormalizedCorpusRecord,
    _filter_reason,
    _iter_source,
    normalize_source_row,
)

from learning.continual.corpus_registry import (
    CorpusRegistrySource,
    load_corpus_registry,
)

from learning.continual.storage import (
    canonical_json,
    immutable_write_json,
    sha256_file,
)

from learning.paths import (
    EVALUATIONS_ROOT,
    REPOSITORY_ROOT,
)


DEFAULT_DEVELOPER_HOLDOUT_ROOT = (
    EVALUATIONS_ROOT
    / "developer-holdout"
)


class DeveloperHoldoutRecord(
    BaseModel
):
    """
    Evaluation-only developer example.

    reference_completion is ground-truth evaluation evidence.
    It must never be routed into a training materialization.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "developer-holdout-record.v1"
        ),
        alias="schema",
    )

    record_id: str

    source_id: str

    source_record_id: (
        str
        | None
    ) = None

    source_identity: str

    upstream_index: int = Field(
        ge=0
    )

    prompt_messages: list[
        dict[str, str]
    ]

    reference_completion: str

    content_sha256: str

    language: (
        str
        | None
    ) = None

    task: (
        str
        | None
    ) = None

    verified_outcome: bool

    repository: (
        str
        | None
    ) = None

    base_commit: (
        str
        | None
    ) = None

    repository_license: (
        str
        | None
    ) = None

    adapter: (
        str
        | None
    ) = None


class DeveloperHoldoutManifest(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "developer-holdout-materialization.v1"
        ),
        alias="schema",
    )

    holdout_id: str
    created_at: str

    source_id: str
    provider: str

    dataset_id: (
        str
        | None
    ) = None

    subset: (
        str
        | None
    ) = None

    revision: (
        str
        | None
    ) = None

    split: (
        str
        | None
    ) = None

    target_component: str

    source_config_sha256: str
    decontamination_policy_sha256: str

    scan_start: int = Field(
        ge=0
    )

    scan_end: int = Field(
        ge=0
    )

    scanned_count: int = Field(
        ge=0
    )

    holdout_count: int = Field(
        ge=1
    )

    rejected_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    records_sha256: str

    evaluation_only: bool = True

    training_eligible: bool = False
    training_authorized: bool = False

    promotion_authorized: bool = False

    output_directory: str


class DeveloperHoldoutMaterializationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    manifest: DeveloperHoldoutManifest

    records_path: str
    manifest_path: str


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


def _safe_name(
    value: str,
) -> str:
    return (
        re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            value,
        )
        .strip("-")
        or "source"
    )


def _increment(
    counts: dict[str, int],
    key: str,
) -> None:
    counts[
        key
    ] = (
        counts.get(
            key,
            0,
        )
        + 1
    )


def _source_config_sha256(
    source: CorpusRegistrySource,
) -> str:
    return _sha256_text(
        canonical_json(
            source.model_dump(
                mode="json",
                by_alias=True,
            )
        )
    )


def _to_holdout_record(
    record: NormalizedCorpusRecord,
) -> DeveloperHoldoutRecord:

    if (
        record.objective
        != "sft"
        or not record.prompt_messages
        or not record.chosen
    ):
        raise ValueError(
            "Developer holdout currently supports "
            "complete SFT examples only."
        )

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

    repository_license = (
        record.metadata.get(
            "repository_license"
        )
    )

    adapter = (
        record.metadata.get(
            "adapter"
        )
    )

    identity = stable_identity(
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

    return (
        DeveloperHoldoutRecord(
            record_id=(
                "developer-holdout-"
                + _sha256_text(
                    (
                        record.source_id
                        + ":"
                        + identity
                    )
                )[:24]
            ),

            source_id=(
                record.source_id
            ),

            source_record_id=(
                record.source_record_id
            ),

            source_identity=(
                identity
            ),

            upstream_index=(
                record.upstream_index
            ),

            prompt_messages=(
                record.prompt_messages
            ),

            reference_completion=(
                record.chosen
            ),

            content_sha256=(
                record.content_sha256
            ),

            language=(
                record.language
            ),

            task=(
                record.task
            ),

            verified_outcome=(
                record.verified_outcome
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

            repository_license=(
                repository_license
                if isinstance(
                    repository_license,
                    str,
                )
                else None
            ),

            adapter=(
                adapter
                if isinstance(
                    adapter,
                    str,
                )
                else None
            ),
        )
    )


def _validate_source(
    source: CorpusRegistrySource,
) -> None:

    if not source.enabled:
        raise ValueError(
            "Corpus source is disabled: "
            + source.source_id
        )

    if source.provider in {
        "runtime",
        "generated",
    }:
        raise ValueError(
            "Runtime/generated evidence cannot be "
            "materialized as an external developer holdout."
        )

    if (
        source.target_component
        != "developer-specialist"
    ):
        raise ValueError(
            "Developer holdout requires "
            "target_component=developer-specialist."
        )

    if (
        "sft"
        not in source.objectives
    ):
        raise ValueError(
            "Developer holdout currently requires "
            "an SFT source."
        )

    if not (
        source.filters
        .benchmark_decontamination
    ):
        raise PermissionError(
            "Developer holdout requires benchmark "
            "decontamination to be enabled."
        )

    if (
        source.provider
        == "huggingface"
        and not source.revision
    ):
        raise PermissionError(
            "Developer holdout requires a pinned "
            "Hugging Face revision."
        )


def _load_existing(
    directory: Path,
) -> DeveloperHoldoutMaterializationResult:

    manifest_path = (
        directory
        / "manifest.json"
    )

    records_path = (
        directory
        / "records.jsonl"
    )

    if not (
        manifest_path.is_file()
        and records_path.is_file()
    ):
        raise ValueError(
            "Existing developer holdout artifact "
            "is incomplete."
        )

    manifest = (
        DeveloperHoldoutManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    observed = sha256_file(
        records_path
    )

    if (
        observed
        != manifest.records_sha256
    ):
        raise ValueError(
            "Developer holdout records SHA-256 "
            "verification failed."
        )

    return (
        DeveloperHoldoutMaterializationResult(
            manifest=manifest,
            records_path=str(
                records_path
            ),
            manifest_path=str(
                manifest_path
            ),
        )
    )


def materialize_developer_holdout(
    *,
    registry_path: Path,
    source_id: str,
    scan_start: int = 0,
    scan_limit: int = 500,
    max_records: int | None = None,
    output_root: Path = (
        DEFAULT_DEVELOPER_HOLDOUT_ROOT
    ),
    decontamination_path: Path = (
        DEFAULT_DECONTAMINATION_PATH
    ),
) -> DeveloperHoldoutMaterializationResult:
    """
    Re-stream a pinned external corpus and capture ONLY the
    deterministic reserved holdout partition.

    This function deliberately does not use or mutate the training
    CorpusCursorStore.

    Normal language/license/verification/secret filters still apply.
    Explicit benchmark blocks remain blocked.

    Only records whose decontamination result is exactly
    reserved_developer_holdout enter the artifact.
    """

    if scan_start < 0:
        raise ValueError(
            "scan_start must be non-negative."
        )

    if scan_limit <= 0:
        raise ValueError(
            "scan_limit must be positive."
        )

    if (
        max_records is not None
        and max_records <= 0
    ):
        raise ValueError(
            "max_records must be positive when supplied."
        )

    registry = load_corpus_registry(
        registry_path
    )

    source = next(
        (
            item

            for item
            in registry.sources

            if (
                item.source_id
                == source_id
            )
        ),
        None,
    )

    if source is None:
        raise ValueError(
            "Unknown corpus source: "
            + source_id
        )

    _validate_source(
        source
    )

    decontamination = (
        load_decontamination_manifest(
            decontamination_path
        )
    )

    policy = (
        decontamination
        .policy_for(
            source.source_id
        )
    )

    if policy is None:
        raise PermissionError(
            "Developer holdout has no pinned "
            "decontamination policy."
        )

    policy_sha = (
        policy_sha256(
            policy
        )
    )

    source_config_sha = (
        _source_config_sha256(
            source
        )
    )

    accepted: list[
        DeveloperHoldoutRecord
    ] = []

    seen_hashes: set[str] = set()

    rejected_counts: dict[
        str,
        int,
    ] = {}

    scanned_count = 0
    scan_end = scan_start

    iterator = _iter_source(
        source,
        start_index=scan_start,
        max_rows=scan_limit,
    )

    try:

        for index, row in iterator:

            if (
                scanned_count
                >= scan_limit
            ):
                break

            scanned_count += 1

            scan_end = (
                index
                + 1
            )

            try:

                record = (
                    normalize_source_row(
                        source,
                        index=index,
                        row=row,
                    )
                )

            except Exception:

                _increment(
                    rejected_counts,
                    "normalization_error",
                )

                continue

            if record is None:

                _increment(
                    rejected_counts,
                    "unsupported_or_incomplete",
                )

                continue

            # Apply every normal corpus safety/quality filter,
            # but do not reject the reserved partition yet.
            normal_reason = (
                _filter_reason(
                    source,
                    record,
                    seen_hashes=(
                        seen_hashes
                    ),
                    decontamination_policy=None,
                )
            )

            if normal_reason is not None:

                _increment(
                    rejected_counts,
                    normal_reason,
                )

                continue

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

            holdout_reason = (
                decontamination_reason(
                    policy=policy,
                    source_record_id=(
                        record
                        .source_record_id
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
                        record
                        .content_sha256
                    ),
                )
            )

            if (
                holdout_reason
                != "reserved_developer_holdout"
            ):

                _increment(
                    rejected_counts,
                    (
                        holdout_reason
                        or "not_reserved_holdout"
                    ),
                )

                continue

            seen_hashes.add(
                record.content_sha256
            )

            accepted.append(
                _to_holdout_record(
                    record
                )
            )

            if (
                max_records
                is not None
                and len(
                    accepted
                )
                >= max_records
            ):
                break

    finally:

        close = getattr(
            iterator,
            "close",
            None,
        )

        if callable(
            close
        ):
            close()

    if not accepted:
        raise ValueError(
            "No eligible reserved developer holdout "
            "records were found in the requested scan window."
        )

    records_payload = "".join(
        canonical_json(
            record.model_dump(
                mode="json",
                by_alias=True,
            )
        )
        + "\n"

        for record
        in accepted
    )

    records_sha = (
        _sha256_text(
            records_payload
        )
    )

    identity = {
        "source_id":
            source.source_id,

        "revision":
            source.revision,

        "scan_start":
            scan_start,

        "scan_end":
            scan_end,

        "records_sha256":
            records_sha,

        "source_config_sha256":
            source_config_sha,

        "decontamination_policy_sha256":
            policy_sha,
    }

    holdout_id = (
        "developer-holdout-"
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

    source_root = (
        root
        / _safe_name(
            source.source_id
        )
    )

    final_directory = (
        source_root
        / holdout_id
    )

    if final_directory.exists():

        return _load_existing(
            final_directory
        )

    manifest = (
        DeveloperHoldoutManifest(
            holdout_id=(
                holdout_id
            ),

            created_at=(
                _utc_now()
            ),

            source_id=(
                source.source_id
            ),

            provider=(
                source.provider
            ),

            dataset_id=(
                source.dataset_id
            ),

            subset=(
                source.subset
            ),

            revision=(
                source.revision
            ),

            split=(
                str(
                    source.metadata.get(
                        "split",
                        "train",
                    )
                )
                if (
                    source.provider
                    == "huggingface"
                )
                else None
            ),

            target_component=(
                source.target_component
            ),

            source_config_sha256=(
                source_config_sha
            ),

            decontamination_policy_sha256=(
                policy_sha
            ),

            scan_start=(
                scan_start
            ),

            scan_end=(
                scan_end
            ),

            scanned_count=(
                scanned_count
            ),

            holdout_count=len(
                accepted
            ),

            rejected_counts=dict(
                sorted(
                    rejected_counts.items()
                )
            ),

            records_sha256=(
                records_sha
            ),

            evaluation_only=True,

            training_eligible=False,
            training_authorized=False,

            promotion_authorized=False,

            output_directory=str(
                final_directory
            ),
        )
    )

    source_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=".developer-holdout-",
            dir=source_root,
        )
    )

    try:

        (
            temporary
            / "records.jsonl"
        ).write_text(
            records_payload,
            encoding="utf-8",
        )

        immutable_write_json(
            temporary
            / "manifest.json",
            manifest,
        )

        os.replace(
            temporary,
            final_directory,
        )

    except Exception:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

        raise

    return (
        DeveloperHoldoutMaterializationResult(
            manifest=manifest,
            records_path=str(
                final_directory
                / "records.jsonl"
            ),
            manifest_path=str(
                final_directory
                / "manifest.json"
            ),
        )
    )
