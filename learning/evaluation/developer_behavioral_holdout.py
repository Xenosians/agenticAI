from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.continual.corpus_decontamination import (
    DEFAULT_DECONTAMINATION_PATH,
    load_decontamination_manifest,
    policy_sha256,
    stable_identity,
)

from learning.continual.corpus_materializer import (
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
    load_jsonl_models,
    sha256_file,
)

from learning.evaluation.developer_holdout import (
    DeveloperHoldoutManifest,
    DeveloperHoldoutRecord,
)

from learning.paths import (
    EVALUATIONS_ROOT,
    REPOSITORY_ROOT,
)


DEFAULT_DEVELOPER_BEHAVIORAL_HOLDOUT_ROOT = (
    EVALUATIONS_ROOT
    / "developer-behavioral-holdout"
)


DEFAULT_CORPUS_REGISTRY = (
    REPOSITORY_ROOT
    / "config"
    / "continual_corpus_registry.json"
)


class DeveloperBehavioralHoldoutRecord(
    BaseModel
):
    """
    Executable SWE evaluation evidence.

    The reference/gold source patch is deliberately excluded.
    Only its SHA-256 is retained for provenance.

    test_patch and transition-test metadata are evaluator-only.
    They must not be placed in the candidate prompt.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "developer-behavioral-holdout-record.v1"
        ),
        alias="schema",
    )

    record_id: str

    parent_record_id: str

    source_id: str

    source_record_id: (
        str
        | None
    ) = None

    source_identity: str

    upstream_index: int = Field(
        ge=0
    )

    problem_statement: str

    repository: str

    base_commit: str

    repository_license: (
        str
        | None
    ) = None

    language: (
        str
        | None
    ) = None

    interface: (
        str
        | None
    ) = None

    image_name: str

    test_patch: str

    fail_to_pass: list[str]

    pass_to_pass: list[str]

    install_config: dict[
        str,
        Any,
    ]

    reference_patch_sha256: str

    gold_patch_included: bool = False

    evaluation_only: bool = True

    training_eligible: bool = False


class DeveloperBehavioralHoldoutManifest(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "developer-behavioral-holdout-materialization.v1"
        ),
        alias="schema",
    )

    holdout_id: str

    created_at: str

    parent_holdout_id: str

    parent_manifest_sha256: str

    parent_records_sha256: str

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

    records_sha256: str

    identity_verified: bool = True

    gold_patch_included: bool = False

    test_patch_included: bool = True

    evaluation_only: bool = True

    training_eligible: bool = False

    training_authorized: bool = False

    promotion_authorized: bool = False

    output_directory: str


class DeveloperBehavioralHoldoutMaterializationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    manifest: (
        DeveloperBehavioralHoldoutManifest
    )

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


def _source_config_sha256(
    source: CorpusRegistrySource,
) -> str:

    return (
        _sha256_text(
            canonical_json(
                source.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )
        )
    )


def _required_text(
    row: dict[
        str,
        Any,
    ],
    key: str,
) -> str:

    value = row.get(
        key
    )

    if not isinstance(
        value,
        str,
    ):

        raise ValueError(
            "Behavioral holdout field "
            f"{key!r} is missing or is not text."
        )

    value = value.strip()

    if not value:

        raise ValueError(
            "Behavioral holdout field "
            f"{key!r} is empty."
        )

    return value


def _optional_text(
    row: dict[
        str,
        Any,
    ],
    key: str,
) -> str | None:

    value = row.get(
        key
    )

    if not isinstance(
        value,
        str,
    ):

        return None

    value = value.strip()

    return (
        value
        or None
    )


def _string_list(
    row: dict[
        str,
        Any,
    ],
    key: str,
    *,
    require_nonempty: bool,
) -> list[str]:

    value = row.get(
        key
    )

    if not isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        raise ValueError(
            "Behavioral holdout field "
            f"{key!r} is not a list."
        )

    result: list[str] = []

    for item in value:

        if not isinstance(
            item,
            str,
        ):

            raise ValueError(
                "Behavioral holdout field "
                f"{key!r} contains a non-string value."
            )

        item = item.strip()

        if item:

            result.append(
                item
            )

    if (
        require_nonempty
        and not result
    ):

        raise ValueError(
            "Behavioral holdout field "
            f"{key!r} is empty."
        )

    return result


def _load_parent_holdout(
    directory: Path,
) -> tuple[
    DeveloperHoldoutManifest,
    list[
        DeveloperHoldoutRecord
    ],
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

    if not (
        manifest_path.is_file()
        and records_path.is_file()
    ):

        raise ValueError(
            "Developer holdout artifact is incomplete."
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
        sha256_file(
            records_path
        )
        != manifest.records_sha256
    ):

        raise ValueError(
            "Parent developer holdout records "
            "SHA-256 verification failed."
        )

    records = (
        load_jsonl_models(
            records_path,
            DeveloperHoldoutRecord,
        )
    )

    if (
        len(
            records
        )
        != manifest.holdout_count
    ):

        raise ValueError(
            "Parent developer holdout count "
            "does not match its record file."
        )

    identities = [
        item.source_identity

        for item
        in records
    ]

    if (
        len(
            identities
        )
        != len(
            set(
                identities
            )
        )
    ):

        raise ValueError(
            "Parent developer holdout contains "
            "duplicate source identities."
        )

    return (
        manifest,
        records,
        sha256_file(
            manifest_path
        ),
    )


def _resolve_source(
    *,
    parent: DeveloperHoldoutManifest,
    registry_path: Path,
    decontamination_path: Path,
) -> CorpusRegistrySource:

    registry = (
        load_corpus_registry(
            registry_path
        )
    )

    source = next(
        (
            item

            for item
            in registry.sources

            if (
                item.source_id
                == parent.source_id
            )
        ),
        None,
    )

    if source is None:

        raise ValueError(
            "Parent holdout source is not present "
            "in the current corpus registry."
        )

    if not source.enabled:

        raise ValueError(
            "Parent holdout source is disabled."
        )

    observed_source_sha = (
        _source_config_sha256(
            source
        )
    )

    if (
        observed_source_sha
        != parent.source_config_sha256
    ):

        raise ValueError(
            "Corpus source configuration changed "
            "since the parent holdout was materialized."
        )

    if (
        source.revision
        != parent.revision
    ):

        raise ValueError(
            "Corpus source revision changed "
            "since the parent holdout was materialized."
        )

    if (
        source.target_component
        != parent.target_component
    ):

        raise ValueError(
            "Corpus target component changed "
            "since the parent holdout was materialized."
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
            "Parent holdout source has no current "
            "decontamination policy."
        )

    observed_policy_sha = (
        policy_sha256(
            policy
        )
    )

    if (
        observed_policy_sha
        != parent
        .decontamination_policy_sha256
    ):

        raise ValueError(
            "Decontamination policy changed "
            "since the parent holdout was materialized."
        )

    return source


def _behavioral_record(
    *,
    parent: DeveloperHoldoutRecord,
    row: dict[
        str,
        Any,
    ],
) -> DeveloperBehavioralHoldoutRecord:

    problem_statement = (
        _required_text(
            row,
            "problem_statement",
        )
    )

    raw_gold_patch = (
        _required_text(
            row,
            "patch",
        )
    )

    test_patch = (
        _required_text(
            row,
            "test_patch",
        )
    )

    image_name = (
        _required_text(
            row,
            "image_name",
        )
    )

    fail_to_pass = (
        _string_list(
            row,
            "FAIL_TO_PASS",
            require_nonempty=True,
        )
    )

    pass_to_pass = (
        _string_list(
            row,
            "PASS_TO_PASS",
            require_nonempty=False,
        )
    )

    install_config = (
        row.get(
            "install_config"
        )
    )

    if not isinstance(
        install_config,
        dict,
    ):

        raise ValueError(
            "Behavioral holdout field "
            "'install_config' is not an object."
        )

    if not (
        parent.repository
        and parent.repository.strip()
    ):

        raise ValueError(
            "Parent holdout record has no repository."
        )

    if not (
        parent.base_commit
        and parent.base_commit.strip()
    ):

        raise ValueError(
            "Parent holdout record has no base commit."
        )

    parent_gold = (
        parent.reference_completion.strip()
    )

    if (
        raw_gold_patch
        != parent_gold
    ):

        raise ValueError(
            "Raw source gold patch does not match "
            "the immutable parent holdout record."
        )

    reference_patch_sha = (
        _sha256_text(
            parent_gold
        )
    )

    return (
        DeveloperBehavioralHoldoutRecord(
            record_id=(
                "developer-behavioral-holdout-"
                + _sha256_text(
                    (
                        parent.source_id
                        + ":"
                        + parent.source_identity
                    )
                )[:24]
            ),

            parent_record_id=(
                parent.record_id
            ),

            source_id=(
                parent.source_id
            ),

            source_record_id=(
                parent.source_record_id
            ),

            source_identity=(
                parent.source_identity
            ),

            upstream_index=(
                parent.upstream_index
            ),

            problem_statement=(
                problem_statement
            ),

            repository=(
                parent.repository
            ),

            base_commit=(
                parent.base_commit
            ),

            repository_license=(
                parent.repository_license
            ),

            language=(
                parent.language
            ),

            interface=(
                _optional_text(
                    row,
                    "interface",
                )
            ),

            image_name=(
                image_name
            ),

            test_patch=(
                test_patch
            ),

            fail_to_pass=(
                fail_to_pass
            ),

            pass_to_pass=(
                pass_to_pass
            ),

            install_config=dict(
                install_config
            ),

            reference_patch_sha256=(
                reference_patch_sha
            ),

            gold_patch_included=False,

            evaluation_only=True,

            training_eligible=False,
        )
    )


def _load_existing(
    directory: Path,
) -> DeveloperBehavioralHoldoutMaterializationResult:

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
            "Existing developer behavioral holdout "
            "artifact is incomplete."
        )

    manifest = (
        DeveloperBehavioralHoldoutManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        sha256_file(
            records_path
        )
        != manifest.records_sha256
    ):

        raise ValueError(
            "Existing developer behavioral holdout "
            "records SHA-256 verification failed."
        )

    return (
        DeveloperBehavioralHoldoutMaterializationResult(
            manifest=manifest,

            records_path=str(
                records_path
            ),

            manifest_path=str(
                manifest_path
            ),
        )
    )


def materialize_developer_behavioral_holdout(
    *,
    holdout_directory: Path,
    registry_path: Path = (
        DEFAULT_CORPUS_REGISTRY
    ),
    decontamination_path: Path = (
        DEFAULT_DECONTAMINATION_PATH
    ),
    output_root: Path = (
        DEFAULT_DEVELOPER_BEHAVIORAL_HOLDOUT_ROOT
    ),
) -> DeveloperBehavioralHoldoutMaterializationResult:
    """
    Materialize executable evaluator-only metadata for the EXACT
    immutable developer loss holdout.

    Selection is not recomputed. The existing parent holdout is the
    authority for source identities, source hashes, revision, and
    decontamination provenance.

    The source is re-streamed only to recover evaluator metadata that
    was deliberately omitted from the loss-only holdout.

    The gold patch is verified against the parent record, hashed, then
    discarded. It is never written into this artifact.
    """

    (
        parent_manifest,
        parent_records,
        parent_manifest_sha,
    ) = (
        _load_parent_holdout(
            holdout_directory
        )
    )

    source = (
        _resolve_source(
            parent=(
                parent_manifest
            ),

            registry_path=(
                registry_path
            ),

            decontamination_path=(
                decontamination_path
            ),
        )
    )

    parent_by_identity = {
        record.source_identity:
            record

        for record
        in parent_records
    }

    recovered: dict[
        str,
        DeveloperBehavioralHoldoutRecord,
    ] = {}

    scan_limit = (
        parent_manifest.scan_end
        - parent_manifest.scan_start
    )

    if scan_limit <= 0:

        raise ValueError(
            "Parent holdout has an invalid source scan window."
        )

    iterator = (
        _iter_source(
            source,
            start_index=(
                parent_manifest
                .scan_start
            ),
            max_rows=(
                scan_limit
            ),
        )
    )

    scanned_count = 0

    try:

        for index, row in iterator:

            if (
                scanned_count
                >= scan_limit
            ):

                break

            scanned_count += 1

            try:

                normalized = (
                    normalize_source_row(
                        source,
                        index=index,
                        row=row,
                    )
                )

            except Exception:

                continue

            if normalized is None:

                continue

            repository = (
                normalized
                .metadata
                .get(
                    "repository"
                )
            )

            base_commit = (
                normalized
                .metadata
                .get(
                    "base_commit"
                )
            )

            identity = (
                stable_identity(
                    source_record_id=(
                        normalized
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
                        normalized
                        .content_sha256
                    ),
                )
            )

            parent = (
                parent_by_identity
                .get(
                    identity
                )
            )

            if parent is None:

                continue

            if identity in recovered:

                raise ValueError(
                    "Source scan produced duplicate "
                    "parent holdout identity: "
                    + identity
                )

            if (
                normalized.content_sha256
                != parent.content_sha256
            ):

                raise ValueError(
                    "Source content changed for parent "
                    "holdout identity: "
                    + identity
                )

            if (
                normalized.upstream_index
                != parent.upstream_index
            ):

                raise ValueError(
                    "Source index changed for parent "
                    "holdout identity: "
                    + identity
                )

            try:

                recovered[
                    identity
                ] = (
                    _behavioral_record(
                        parent=parent,
                        row=row,
                    )
                )

            except ValueError as exc:

                source_name = (
                    parent.source_record_id
                    or parent.source_identity
                )

                raise ValueError(
                    "Behavioral metadata is incomplete "
                    f"for {source_name}: {exc}"
                ) from exc

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

    missing = [
        (
            record.source_record_id
            or record.source_identity
        )

        for record
        in parent_records

        if (
            record.source_identity
            not in recovered
        )
    ]

    if missing:

        raise ValueError(
            "Could not recover executable metadata "
            "for every parent holdout record: "
            + ", ".join(
                missing[
                    :5
                ]
            )
            + (
                " ..."
                if len(
                    missing
                )
                > 5
                else ""
            )
        )

    records = [
        recovered[
            parent.source_identity
        ]

        for parent
        in parent_records
    ]

    if (
        len(
            records
        )
        != parent_manifest.holdout_count
    ):

        raise RuntimeError(
            "Behavioral holdout identity coverage "
            "does not equal the parent holdout."
        )

    records_payload = "".join(
        (
            canonical_json(
                record.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )
            + "\n"
        )

        for record
        in records
    )

    records_sha = (
        _sha256_text(
            records_payload
        )
    )

    identity_payload = {
        "parent_holdout_id":
            parent_manifest.holdout_id,

        "parent_manifest_sha256":
            parent_manifest_sha,

        "parent_records_sha256":
            parent_manifest.records_sha256,

        "source_config_sha256":
            parent_manifest.source_config_sha256,

        "decontamination_policy_sha256":
            (
                parent_manifest
                .decontamination_policy_sha256
            ),

        "records_sha256":
            records_sha,

        "artifact_contract":
            (
                "developer-behavioral-holdout.v1"
            ),
    }

    holdout_id = (
        "developer-behavioral-holdout-"
        + _sha256_text(
            canonical_json(
                identity_payload
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
            parent_manifest
            .source_id
        )
    )

    output_directory = (
        source_root
        / holdout_id
    )

    if output_directory.exists():

        return (
            _load_existing(
                output_directory
            )
        )

    manifest = (
        DeveloperBehavioralHoldoutManifest(
            holdout_id=(
                holdout_id
            ),

            created_at=(
                _utc_now()
            ),

            parent_holdout_id=(
                parent_manifest.holdout_id
            ),

            parent_manifest_sha256=(
                parent_manifest_sha
            ),

            parent_records_sha256=(
                parent_manifest
                .records_sha256
            ),

            source_id=(
                parent_manifest.source_id
            ),

            provider=(
                parent_manifest.provider
            ),

            dataset_id=(
                parent_manifest.dataset_id
            ),

            subset=(
                parent_manifest.subset
            ),

            revision=(
                parent_manifest.revision
            ),

            split=(
                parent_manifest.split
            ),

            target_component=(
                parent_manifest
                .target_component
            ),

            source_config_sha256=(
                parent_manifest
                .source_config_sha256
            ),

            decontamination_policy_sha256=(
                parent_manifest
                .decontamination_policy_sha256
            ),

            scan_start=(
                parent_manifest.scan_start
            ),

            scan_end=(
                parent_manifest.scan_end
            ),

            scanned_count=(
                scanned_count
            ),

            holdout_count=len(
                records
            ),

            records_sha256=(
                records_sha
            ),

            identity_verified=True,

            gold_patch_included=False,

            test_patch_included=True,

            evaluation_only=True,

            training_eligible=False,

            training_authorized=False,

            promotion_authorized=False,

            output_directory=str(
                output_directory
            ),
        )
    )

    source_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=(
                ".developer-behavioral-holdout-"
            ),
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

        if output_directory.exists():

            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )

        else:

            os.replace(
                temporary,
                output_directory,
            )

    except Exception:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

        raise

    return (
        DeveloperBehavioralHoldoutMaterializationResult(
            manifest=manifest,

            records_path=str(
                output_directory
                / "records.jsonl"
            ),

            manifest_path=str(
                output_directory
                / "manifest.json"
            ),
        )
    )
