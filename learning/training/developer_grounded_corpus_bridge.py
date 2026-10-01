from __future__ import annotations

import hashlib
import os
import shutil
import tempfile

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from learning.continual.corpus_materializer import (
    CorpusSnapshotManifest,
    NormalizedCorpusRecord,
)

from learning.continual.storage import (
    fingerprint_directory,
    immutable_write_json,
    immutable_write_jsonl,
)

from learning.evaluation.developer_candidate_patch import (
    CandidateContextFile,
    _build_messages,
)

from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)

from learning.training.developer_corpus_bridge import (
    _load_snapshot,
    _required_sha,
    _selection_key,
    _validate_snapshot,
    _validation_record,
)

from learning.training.phase5_contracts import (
    canonical_json,
    sha256_text,
)

from learning.training.phase5_materializer import (
    Phase5DpoRecord,
    Phase5MaterializationManifest,
    Phase5MaterializationResult,
    Phase5SftRecord,
)


DEFAULT_DEVELOPER_GROUNDED_MATERIALIZATION_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "developer-grounded-sft-materialized"
)

_ISSUE_PREFIX = (
    "Resolve the software issue below with a minimal, "
    "testable source patch."
)


def _issue_statement(
    record: NormalizedCorpusRecord,
) -> str:

    user_messages = [
        item.get(
            "content",
            "",
        ).strip()

        for item
        in record.prompt_messages

        if (
            item.get("role") == "user"

            and item.get(
                "content",
                "",
            ).strip()
        )
    ]

    if not user_messages:

        raise ValueError(
            "Developer SFT record has no user issue message."
        )

    issue = user_messages[-1]

    prefix = (
        _ISSUE_PREFIX
        + "\n\n"
    )

    if issue.startswith(
        prefix
    ):

        issue = issue[
            len(prefix):
        ]

    issue = issue.strip()

    if not issue:

        raise ValueError(
            "Developer SFT issue text is empty."
        )

    return issue


def _grounded_selection(
    record: NormalizedCorpusRecord,
    grounded_fraction: float,
) -> bool:

    if grounded_fraction <= 0.0:
        return False

    if grounded_fraction >= 1.0:
        return True

    digest = hashlib.sha256(
        (
            "developer-grounded-sft:"
            + record.record_id
        ).encode(
            "utf-8"
        )
    ).digest()

    value = (
        int.from_bytes(
            digest[:8],
            "big",
        )
        / float(
            2 ** 64
        )
    )

    return (
        value
        < grounded_fraction
    )


def _patch_preimage_files(
    patch: str,
    *,
    context_chars: int,
    max_files: int,
) -> tuple[
    list[str],
    list[
        CandidateContextFile
    ],
]:

    if context_chars <= 0:

        raise ValueError(
            "context_chars must be positive."
        )

    if max_files <= 0:

        raise ValueError(
            "max_files must be positive."
        )

    current_path: str | None = None

    current_lines: list[str] = []

    in_hunk = False

    paths: list[str] = []

    preimages: dict[
        str,
        list[str],
    ] = {}

    def flush() -> None:

        nonlocal current_lines

        if (
            current_path is None
            or not current_lines
        ):

            current_lines = []

            return

        bucket = (
            preimages
            .setdefault(
                current_path,
                [],
            )
        )

        if bucket:

            bucket.append(
                "..."
            )

        bucket.extend(
            current_lines
        )

        current_lines = []

    for raw_line in (
        patch.splitlines()
    ):

        if raw_line.startswith(
            "diff --git "
        ):

            flush()

            parts = (
                raw_line.split()
            )

            if (
                len(parts) >= 4

                and parts[
                    2
                ].startswith(
                    "a/"
                )

                and parts[
                    3
                ].startswith(
                    "b/"
                )
            ):

                old_path = (
                    parts[
                        2
                    ][
                        2:
                    ]
                )

                new_path = (
                    parts[
                        3
                    ][
                        2:
                    ]
                )

                current_path = (
                    new_path
                    if new_path
                    else old_path
                )

                if (
                    current_path

                    and current_path
                    not in paths
                ):

                    paths.append(
                        current_path
                    )

            else:

                current_path = None

            in_hunk = False

            continue

        if raw_line.startswith(
            "@@"
        ):

            flush()

            in_hunk = True

            continue

        if not (
            in_hunk
            and current_path
        ):

            continue

        if raw_line.startswith(
            "\\ No newline at end of file"
        ):

            continue

        # Gold/new lines must NEVER enter the training prompt.
        if raw_line.startswith(
            "+"
        ):

            continue

        if raw_line.startswith(
            "-"
        ):

            current_lines.append(
                raw_line[
                    1:
                ]
            )

            continue

        if raw_line.startswith(
            " "
        ):

            current_lines.append(
                raw_line[
                    1:
                ]
            )

    flush()

    selected_paths = [
        path

        for path
        in paths

        if preimages.get(
            path
        )
    ][
        :max_files
    ]

    remaining = (
        context_chars
    )

    files: list[
        CandidateContextFile
    ] = []

    for path in (
        selected_paths
    ):

        if remaining <= 0:

            break

        full_excerpt = (
            "\n"
            .join(
                preimages[
                    path
                ]
            )
            .strip()
        )

        if not full_excerpt:

            continue

        excerpt = (
            full_excerpt[
                :remaining
            ]
        )

        files.append(
            CandidateContextFile(
                path=path,

                excerpt=excerpt,

                captured_chars=len(
                    full_excerpt
                ),

                excerpt_chars=len(
                    excerpt
                ),

                truncated=(
                    len(
                        excerpt
                    )
                    < len(
                        full_excerpt
                    )
                ),
            )
        )

        remaining -= len(
            excerpt
        )

    return (
        selected_paths,
        files,
    )


def _grounded_messages(
    record: NormalizedCorpusRecord,
    *,
    context_chars: int,
    max_files: int,
) -> list[
    dict[str, str]
]:

    repository = (
        record
        .metadata
        .get(
            "repository"
        )
    )

    base_commit = (
        record
        .metadata
        .get(
            "base_commit"
        )
    )

    if not (
        isinstance(
            repository,
            str,
        )

        and repository.strip()
    ):

        raise ValueError(
            "Grounded developer record has no repository."
        )

    if not (
        isinstance(
            base_commit,
            str,
        )

        and base_commit.strip()
    ):

        raise ValueError(
            "Grounded developer record has no base commit."
        )

    if not record.chosen:

        raise ValueError(
            "Grounded developer record has no target patch."
        )

    (
        tree,
        files,
    ) = (
        _patch_preimage_files(
            record.chosen,

            context_chars=(
                context_chars
            ),

            max_files=(
                max_files
            ),
        )
    )

    if not files:

        raise ValueError(
            "Verified patch yielded no pre-change source context."
        )

    prompt_context = (
        SimpleNamespace(
            problem_statement=(
                _issue_statement(
                    record
                )
            ),

            repository=(
                repository.strip()
            ),

            base_commit=(
                base_commit.strip()
            ),

            repository_tree=(
                tree
            ),

            files=(
                files
            ),
        )
    )

    # Deliberately reuse the exact inference prompt formatter.
    return (
        _build_messages(
            prompt_context
        )
    )


def _with_messages(
    record: NormalizedCorpusRecord,
    messages: list[
        dict[str, str]
    ],
) -> NormalizedCorpusRecord:

    return (
        record.model_copy(
            update={
                "prompt_messages":
                    messages,
            }
        )
    )


def _sequence_reason(
    *,
    record: NormalizedCorpusRecord,

    token_length_resolver: Callable[
        [
            NormalizedCorpusRecord
        ],
        tuple[
            int,
            int,
            int,
        ],
    ],

    max_sequence_tokens: int,
) -> str | None:

    try:

        (
            _prompt_tokens,
            full_tokens,
            common_prefix_tokens,
        ) = (
            token_length_resolver(
                record
            )
        )

    except Exception:

        return (
            "bridge_sequence_encoding_failed"
        )

    if common_prefix_tokens <= 0:

        return (
            "bridge_sequence_encoding_failed"
        )

    if (
        common_prefix_tokens
        >= max_sequence_tokens
    ):

        return (
            "bridge_prompt_too_long"
        )

    if (
        full_tokens
        > max_sequence_tokens
    ):

        return (
            "bridge_full_sequence_too_long"
        )

    if (
        full_tokens
        <= common_prefix_tokens
    ):

        return (
            "bridge_no_completion_tokens"
        )

    return None


def _phase5_record(
    *,
    snapshot: CorpusSnapshotManifest,
    record: NormalizedCorpusRecord,
    partition: str,
    grounded: bool,
) -> Phase5SftRecord:

    prompt_sha = (
        sha256_text(
            canonical_json(
                record.prompt_messages
            )
        )
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

        "prompt_sha256":
            prompt_sha,

        "grounded":
            grounded,
    }

    materialized_record_id = (
        "developer-grounded-sft-"
        + sha256_text(
            canonical_json(
                identity
            )
        )[
            :24
        ]
    )

    source_identity = (
        record.source_record_id
        or str(
            record.upstream_index
        )
    )

    return (
        Phase5SftRecord(
            record_id=(
                materialized_record_id
            ),

            member_id=(
                "corpus:"
                + snapshot.snapshot_id
                + ":"
                + record.record_id
                + (
                    ":grounded"
                    if grounded
                    else ":issue-only"
                )
            ),

            partition=(
                partition
            ),

            role=(
                "developer-specialist"
            ),

            source_kind=(
                "external_developer_grounded_corpus"
                if grounded
                else "external_developer_corpus"
            ),

            prompt_messages=(
                record.prompt_messages
            ),

            chosen=(
                record.chosen
                or ""
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
    )


def materialize_developer_grounded_sft_snapshot(
    *,
    snapshot_directory: Path,
    target_model_key: str,
    base_model_path: Path,
    target_contract_path: Path,

    token_length_resolver: Callable[
        [
            NormalizedCorpusRecord
        ],
        tuple[
            int,
            int,
            int,
        ],
    ],

    max_sequence_tokens: int = 1536,

    max_records: int = 64,

    grounded_fraction: float = 0.75,

    context_chars: int = 1400,

    max_files: int = 4,

    output_root: Path = (
        DEFAULT_DEVELOPER_GROUNDED_MATERIALIZATION_ROOT
    ),
) -> Phase5MaterializationResult:

    if max_records < 2:

        raise ValueError(
            "Grounded developer SFT requires at least 2 records."
        )

    if max_sequence_tokens < 128:

        raise ValueError(
            "max_sequence_tokens must be at least 128."
        )

    if not (
        0.0
        <= grounded_fraction
        <= 1.0
    ):

        raise ValueError(
            "grounded_fraction must be between 0 and 1."
        )

    if context_chars < 256:

        raise ValueError(
            "context_chars must be at least 256."
        )

    if max_files < 1:

        raise ValueError(
            "max_files must be positive."
        )

    (
        snapshot,
        all_records,
        snapshot_manifest_sha,
    ) = (
        _load_snapshot(
            snapshot_directory
        )
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

            and (
                record
                .metadata
                .get(
                    "adapter"
                )
                == "issue_patch"
            )
        )
    ]

    if len(
        eligible
    ) < 2:

        raise ValueError(
            "Developer snapshot contains fewer than two "
            "eligible issue-patch records."
        )

    excluded = {
        "bridge_sequence_encoding_failed":
            0,

        "bridge_prompt_too_long":
            0,

        "bridge_full_sequence_too_long":
            0,

        "bridge_no_completion_tokens":
            0,

        "grounding_missing_repository":
            0,

        "grounding_missing_base_commit":
            0,

        "grounding_no_preimage_context":
            0,
    }

    sequence_eligible: list[
        tuple[
            NormalizedCorpusRecord,
            bool,
        ]
    ] = []

    for source_record in sorted(
        eligible,
        key=_selection_key,
    ):

        grounded = (
            _grounded_selection(
                source_record,
                grounded_fraction,
            )
        )

        candidate = (
            source_record
        )

        if grounded:

            repository = (
                source_record
                .metadata
                .get(
                    "repository"
                )
            )

            base_commit = (
                source_record
                .metadata
                .get(
                    "base_commit"
                )
            )

            if not (
                isinstance(
                    repository,
                    str,
                )

                and repository.strip()
            ):

                excluded[
                    "grounding_missing_repository"
                ] += 1

                continue

            if not (
                isinstance(
                    base_commit,
                    str,
                )

                and base_commit.strip()
            ):

                excluded[
                    "grounding_missing_base_commit"
                ] += 1

                continue

            try:

                messages = (
                    _grounded_messages(
                        source_record,

                        context_chars=(
                            context_chars
                        ),

                        max_files=(
                            max_files
                        ),
                    )
                )

            except ValueError:

                excluded[
                    "grounding_no_preimage_context"
                ] += 1

                continue

            candidate = (
                _with_messages(
                    source_record,
                    messages,
                )
            )

        reason = (
            _sequence_reason(
                record=(
                    candidate
                ),

                token_length_resolver=(
                    token_length_resolver
                ),

                max_sequence_tokens=(
                    max_sequence_tokens
                ),
            )
        )

        if reason is not None:

            excluded[
                reason
            ] += 1

            continue

        sequence_eligible.append(
            (
                candidate,
                grounded,
            )
        )

    if len(
        sequence_eligible
    ) < 2:

        raise ValueError(
            "Grounded developer bridge produced fewer than two "
            f"complete examples within {max_sequence_tokens} tokens."
        )

    selected = (
        sequence_eligible[
            :max_records
        ]
    )

    validation_ids = {
        record.record_id

        for (
            record,
            _grounded,
        )
        in selected

        if _validation_record(
            record
        )
    }

    if not validation_ids:

        validation_ids = {
            selected[
                -1
            ][
                0
            ].record_id
        }

    training_pairs = [
        pair

        for pair
        in selected

        if (
            pair[
                0
            ].record_id
            not in validation_ids
        )
    ]

    validation_pairs = [
        pair

        for pair
        in selected

        if (
            pair[
                0
            ].record_id
            in validation_ids
        )
    ]

    if not training_pairs:

        training_pairs = [
            selected[
                0
            ]
        ]

        validation_pairs = (
            selected[
                1:
            ]
        )

    if not validation_pairs:

        raise ValueError(
            "Grounded developer bridge could not construct "
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

    selected_identity = [
        {
            "record_id":
                record.record_id,

            "grounded":
                grounded,

            "prompt_sha256":
                sha256_text(
                    canonical_json(
                        record.prompt_messages
                    )
                ),
        }

        for (
            record,
            grounded,
        )
        in selected
    ]

    identity = {
        "schema":
            "developer-grounded-corpus-bridge.v1",

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

        "grounded_fraction":
            grounded_fraction,

        "context_chars":
            context_chars,

        "max_files":
            max_files,

        "selected_records":
            selected_identity,
    }

    materialization_id = (
        "developer-grounded-sft-materialization-"
        + sha256_text(
            canonical_json(
                identity
            )
        )[
            :24
        ]
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

    if target.is_dir():

        existing = (
            Phase5MaterializationManifest
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
                "Existing grounded developer materialization "
                "identity mismatch."
            )

        return (
            Phase5MaterializationResult(
                manifest=existing,

                output_directory=str(
                    target
                ),
            )
        )

    sft_train = [
        _phase5_record(
            snapshot=snapshot,

            record=record,

            partition="train",

            grounded=grounded,
        )

        for (
            record,
            grounded,
        )
        in training_pairs
    ]

    sft_validation = [
        _phase5_record(
            snapshot=snapshot,

            record=record,

            partition="validation",

            grounded=grounded,
        )

        for (
            record,
            grounded,
        )
        in validation_pairs
    ]

    dpo_train: list[
        Phase5DpoRecord
    ] = []

    dpo_validation: list[
        Phase5DpoRecord
    ] = []

    grounded_count = sum(
        1

        for (
            _record,
            grounded,
        )
        in selected

        if grounded
    )

    issue_only_count = (
        len(
            selected
        )
        - grounded_count
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=(
                ".developer-grounded-sft-"
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
            Phase5MaterializationManifest(
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
                    "external-grounded-corpus:"
                    + snapshot.source_id
                ),

                curriculum_version=(
                    str(
                        snapshot.revision
                    )
                ),

                chapter_id=(
                    "developer-grounded-sft:"
                    + snapshot.source_id
                ),

                base_model_path=str(
                    base_model_path
                ),

                base_model_sha256=(
                    base_model_sha
                ),

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

                    "grounded_selected":
                        grounded_count,

                    "issue_only_selected":
                        issue_only_count,

                    **excluded,
                },

                lineage_isolated=True,

                ready_for_training=True,

                training_authorized=False,

                promotion_authorized=False,
            )
        )

        immutable_write_json(
            temporary
            / "manifest.json",

            manifest,
        )

        immutable_write_json(
            temporary
            / "grounding-report.json",

            {
                "schema":
                    "developer-grounded-corpus-report.v1",

                "materialization_id":
                    materialization_id,

                "snapshot_id":
                    snapshot.snapshot_id,

                "grounded_fraction_requested":
                    grounded_fraction,

                "grounded_selected":
                    grounded_count,

                "issue_only_selected":
                    issue_only_count,

                "selected_total":
                    len(
                        selected
                    ),

                "context_source":
                    "verified_patch_preimage_only",

                "target_added_lines_in_prompt":
                    False,

                "context_chars":
                    context_chars,

                "max_files":
                    max_files,

                "max_sequence_tokens":
                    max_sequence_tokens,

                "excluded_counts":
                    excluded,
            },
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

    return (
        Phase5MaterializationResult(
            manifest=manifest,

            output_directory=str(
                target
            ),
        )
    )
