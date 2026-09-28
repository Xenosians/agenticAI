from __future__ import annotations

import hashlib
import json
import shutil

from pathlib import Path

from learning.continual.storage import (
    canonical_json,
    immutable_write_json,
    immutable_write_jsonl,
    load_jsonl_models,
    sha256_file,
)
from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)
from learning.training.phase5_materializer import (
    Phase5DpoRecord,
    Phase5MaterializationManifest,
    Phase5SftRecord,
)
from learning.continuous.corpus import (
    study_split,
)
from learning.continuous.types import (
    CorpusPage,
)


DEFAULT_CONTINUOUS_MATERIALIZATION_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continuous"
    / "materialized"
)


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def find_latest_ready_materialization(
    *,
    root: Path = (
        RUNTIME_LEARNING_ROOT
        / "phase5"
        / "materialized"
    ),
) -> Path:
    root = (
        root
        .expanduser()
        .resolve()
    )

    candidates = []

    if not root.is_dir():
        raise ValueError(
            "No Phase-5 materialization root exists."
        )

    for manifest_path in root.glob(
        "*/manifest.json"
    ):
        try:
            manifest = (
                Phase5MaterializationManifest
                .model_validate_json(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
        except Exception:
            continue

        if manifest.ready_for_training:
            candidates.append(
                (
                    manifest.created_at,
                    manifest_path.parent,
                )
            )

    if not candidates:
        raise ValueError(
            "No ready Phase-5 materialization exists."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            str(
                item[1]
            ),
        )
    )

    return candidates[
        -1
    ][
        1
    ]


def _code_study_record(
    *,
    page: CorpusPage,
    partition: str,
) -> Phase5SftRecord:
    prefix, continuation = (
        study_split(
            page.content
        )
    )

    if page.target_role == "developer":
        system = (
            "You are a developer model studying trusted source code. "
            "Continue the supplied trusted source fragment faithfully. "
            "Preserve syntax, identifiers, formatting intent, and local "
            "coding patterns. Do not invent unrelated APIs."
        )
    elif page.target_role == "hub":
        system = (
            "You are studying trusted general language/code material. "
            "Continue the supplied source faithfully and compactly."
        )
    else:
        system = (
            "Continue the supplied trusted source faithfully."
        )

    prompt = [
        {
            "role":
                "system",

            "content":
                system,
        },
        {
            "role":
                "user",

            "content":
                (
                    f"Source: {page.title}\n"
                    f"Target role: {page.target_role}\n"
                    "Continue the source from exactly where the prefix ends.\n\n"
                    f"{prefix}"
                ),
        },
    ]

    payload = {
        "page_id":
            page.page_id,

        "source_sha256":
            page.source_sha256,

        "partition":
            partition,

        "prompt":
            prompt,

        "chosen":
            continuation,
    }

    record_id = (
        "continuous-sft-"
        + _sha256_text(
            canonical_json(
                payload
            )
        )[:24]
    )

    return Phase5SftRecord(
        record_id=record_id,
        member_id=(
            "corpus:"
            + page.page_id
        ),
        partition=partition,
        role=(
            "replay_code"
            if page.replay
            else "current_code"
        ),
        source_kind=(
            "continuous_corpus"
        ),
        prompt_messages=(
            prompt
        ),
        chosen=(
            continuation
        ),
        source_id=(
            page.page_id
        ),
        source_artifact=(
            page.source_id
        ),
        source_artifact_sha256=(
            page.source_sha256
        ),
        lineage_id=(
            "corpus:"
            + page.source_id
            + ":"
            + str(
                page.page_index
            )
        ),
    )


def augment_materialization_with_corpus(
    *,
    base_directory: Path,
    cycle_id: str,
    pages: list[CorpusPage],
    output_root: Path = (
        DEFAULT_CONTINUOUS_MATERIALIZATION_ROOT
    ),
) -> Path:
    """
    Overlay a small progressive corpus window onto an already-governed
    behavior materialization.

    The original materialization is never modified.
    """
    base_directory = (
        base_directory
        .expanduser()
        .resolve()
    )

    manifest = (
        Phase5MaterializationManifest
        .model_validate_json(
            (
                base_directory
                / "manifest.json"
            )
            .read_text(
                encoding="utf-8"
            )
        )
    )

    sft_train = load_jsonl_models(
        base_directory
        / "sft-train.jsonl",
        Phase5SftRecord,
    )

    sft_validation = load_jsonl_models(
        base_directory
        / "sft-validation.jsonl",
        Phase5SftRecord,
    )

    dpo_train = load_jsonl_models(
        base_directory
        / "dpo-train.jsonl",
        Phase5DpoRecord,
    )

    dpo_validation = load_jsonl_models(
        base_directory
        / "dpo-validation.jsonl",
        Phase5DpoRecord,
    )

    page_ids = [
        page.page_id
        for page in pages
    ]

    overlay_identity = {
        "base_materialization":
            manifest.materialization_id,

        "cycle_id":
            cycle_id,

        "page_ids":
            page_ids,

        "page_hashes": [
            page.source_sha256
            for page in pages
        ],
    }

    overlay_id = (
        "continuous-materialization-"
        + _sha256_text(
            canonical_json(
                overlay_identity
            )
        )[:24]
    )

    target = (
        output_root
        .expanduser()
        .resolve()
        / overlay_id
    )

    if (
        target
        / "manifest.json"
    ).is_file():
        return target

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        # Deterministic 80/20 page split while preserving source-page lineage.
        for page in pages:
            bucket = int(
                _sha256_text(
                    page.page_id
                )[:8],
                16,
            ) % 5

            partition = (
                "validation"
                if bucket == 0
                else "train"
            )

            record = (
                _code_study_record(
                    page=page,
                    partition=(
                        partition
                    ),
                )
            )

            if partition == "validation":
                sft_validation.append(
                    record
                )
            else:
                sft_train.append(
                    record
                )

        # Existing behavior validation remains non-empty, so an unlucky
        # small corpus batch with no corpus validation page is still valid.
        sft_train_sha = immutable_write_jsonl(
            target
            / "sft-train.jsonl",
            sft_train,
        )

        sft_validation_sha = immutable_write_jsonl(
            target
            / "sft-validation.jsonl",
            sft_validation,
        )

        dpo_train_sha = immutable_write_jsonl(
            target
            / "dpo-train.jsonl",
            dpo_train,
        )

        dpo_validation_sha = immutable_write_jsonl(
            target
            / "dpo-validation.jsonl",
            dpo_validation,
        )

        overlay_manifest = (
            manifest.model_copy(
                update={
                    "materialization_id":
                        overlay_id,

                    "created_at":
                        manifest.created_at,

                    "sft_train_count":
                        len(
                            sft_train
                        ),

                    "sft_validation_count":
                        len(
                            sft_validation
                        ),

                    "dpo_train_count":
                        len(
                            dpo_train
                        ),

                    "dpo_validation_count":
                        len(
                            dpo_validation
                        ),

                    "sft_train_sha256":
                        sft_train_sha,

                    "sft_validation_sha256":
                        sft_validation_sha,

                    "dpo_train_sha256":
                        dpo_train_sha,

                    "dpo_validation_sha256":
                        dpo_validation_sha,

                    "ready_for_training":
                        True,

                    "training_authorized":
                        False,

                    "promotion_authorized":
                        False,
                }
            )
        )

        immutable_write_json(
            target
            / "manifest.json",
            overlay_manifest,
        )

        immutable_write_json(
            target
            / "continuous-overlay.json",
            {
                "schema":
                    "continuous-corpus-overlay.v1",

                "cycle_id":
                    cycle_id,

                "base_materialization_id":
                    manifest.materialization_id,

                "overlay_materialization_id":
                    overlay_id,

                "page_ids":
                    page_ids,

                "page_count":
                    len(
                        pages
                    ),

                "page_source_sha256": {
                    page.page_id:
                        page.source_sha256
                    for page in pages
                },
            },
        )

    except Exception:
        shutil.rmtree(
            target,
            ignore_errors=True,
        )
        raise

    return target
