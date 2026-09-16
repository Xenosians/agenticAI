import hashlib
import json

from pathlib import (
    Path,
)

import pytest

from learning.dpo_materializer import (
    SpecialistDpoManifest,
    SpecialistDpoRecord,
)

from learning.dpo_qlora import (
    SpecialistDpoQloraSettings,
    build_dpo_dataset_rows,
    load_specialist_dpo_partition,
    validate_specialist_dpo_pair,
)


def canonical_json(
    value,
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


def sha256_text(
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


def dpo_record(
    *,
    dpo_record_id: str,
    source_record_id: str,
    target_agent: str = (
        "account-specialist"
    ),
    target_model_key: str = (
        "qwen2.5-0.5b-funccall"
    ),
) -> SpecialistDpoRecord:

    return (
        SpecialistDpoRecord(
            dpo_record_id=(
                dpo_record_id
            ),

            source_record_id=(
                source_record_id
            ),

            source_trajectory_id=(
                f"trajectory-{source_record_id}"
            ),

            source_correction_id=(
                f"correction-{source_record_id}"
            ),

            target_agent=(
                target_agent
            ),

            target_model_key=(
                target_model_key
            ),

            change_type=(
                "arguments"
            ),

            prompt_messages=[
                {
                    "role":
                        "system",

                    "content":
                        "Synthetic worker system prompt.",
                },

                {
                    "role":
                        "user",

                    "content":
                        "Check whether jdoe is locked.",
                },

                {
                    "role":
                        "user",

                    "content":
                        (
                            "Additional task context "
                            "from the routing stage:\n"
                            "Inspect the requested account."
                        ),
                },
            ],

            chosen=(
                canonical_json(
                    [
                        {
                            "name":
                                "account_status",

                            "arguments": {
                                "user_id":
                                    "jdoe"
                            },
                        }
                    ]
                )
            ),

            rejected=(
                canonical_json(
                    [
                        {
                            "name":
                                "account_status",

                            "arguments": {
                                "user_id":
                                    "wrong-user"
                            },
                        }
                    ]
                )
            ),
        )
    )


def write_partition(
    *,
    root: Path,
    partition: str,
    records: list[
        SpecialistDpoRecord
    ],
    split_id: str = (
        "split-test"
    ),
    target_agent: str = (
        "account-specialist"
    ),
    target_model_key: str = (
        "qwen2.5-0.5b-funccall"
    ),
) -> Path:

    directory = (
        root
        / partition
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    lines = [
        canonical_json(
            record.model_dump(
                mode="json",
                by_alias=True,
            )
        )

        for record
        in records
    ]

    content = (
        "\n".join(
            lines
        )
        + "\n"
    )

    (
        directory
        / "records.jsonl"
    ).write_text(
        content,
        encoding="utf-8",
    )

    manifest = (
        SpecialistDpoManifest(
            created_at=(
                "2026-09-16T00:00:00+00:00"
            ),

            target_agent=(
                target_agent
            ),

            target_model_key=(
                target_model_key
            ),

            source_split_id=(
                split_id
            ),

            source_partition=(
                partition
            ),

            source_sha256=(
                f"{partition}-source-hash"
            ),

            record_count=(
                len(
                    records
                )
            ),

            excluded_record_count=0,

            exclusion_reason_counts={},

            content_sha256=(
                sha256_text(
                    content
                )
            ),

            source_record_ids=[
                record.source_record_id

                for record
                in records
            ],
        )
    )

    (
        directory
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        directory
    )


def test_valid_materialized_partition_loads(
    tmp_path: Path,
):

    directory = (
        write_partition(
            root=(
                tmp_path
            ),

            partition=(
                "train"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-train-1"
                    ),

                    source_record_id=(
                        "train-1"
                    ),
                )
            ],
        )
    )

    loaded = (
        load_specialist_dpo_partition(
            directory=(
                directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    assert (
        loaded.manifest.record_count
        == 1
    )

    assert (
        loaded.records[
            0
        ].source_record_id
        == "train-1"
    )


def test_tampered_materialized_partition_is_refused(
    tmp_path: Path,
):

    directory = (
        write_partition(
            root=(
                tmp_path
            ),

            partition=(
                "train"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-train-1"
                    ),

                    source_record_id=(
                        "train-1"
                    ),
                )
            ],
        )
    )

    records_path = (
        directory
        / "records.jsonl"
    )

    records_path.write_text(
        (
            records_path.read_text(
                encoding="utf-8"
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "SHA-256 does not match"
        ),
    ):

        load_specialist_dpo_partition(
            directory=(
                directory
            ),

            expected_partition=(
                "train"
            ),
        )


def test_train_validation_pair_is_disjoint(
    tmp_path: Path,
):

    train_directory = (
        write_partition(
            root=(
                tmp_path
                / "train-root"
            ),

            partition=(
                "train"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-train-1"
                    ),

                    source_record_id=(
                        "train-1"
                    ),
                )
            ],
        )
    )

    validation_directory = (
        write_partition(
            root=(
                tmp_path
                / "validation-root"
            ),

            partition=(
                "validation"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-validation-1"
                    ),

                    source_record_id=(
                        "validation-1"
                    ),
                )
            ],
        )
    )

    train = (
        load_specialist_dpo_partition(
            directory=(
                train_directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    validation = (
        load_specialist_dpo_partition(
            directory=(
                validation_directory
            ),

            expected_partition=(
                "validation"
            ),
        )
    )

    validate_specialist_dpo_pair(
        train=(
            train
        ),

        validation=(
            validation
        ),

        expected_model_key=(
            "qwen2.5-0.5b-funccall"
        ),
    )


def test_train_validation_source_overlap_is_refused(
    tmp_path: Path,
):

    train_directory = (
        write_partition(
            root=(
                tmp_path
                / "train-root"
            ),

            partition=(
                "train"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-train-1"
                    ),

                    source_record_id=(
                        "shared-source"
                    ),
                )
            ],
        )
    )

    validation_directory = (
        write_partition(
            root=(
                tmp_path
                / "validation-root"
            ),

            partition=(
                "validation"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-validation-1"
                    ),

                    source_record_id=(
                        "shared-source"
                    ),
                )
            ],
        )
    )

    train = (
        load_specialist_dpo_partition(
            directory=(
                train_directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    validation = (
        load_specialist_dpo_partition(
            directory=(
                validation_directory
            ),

            expected_partition=(
                "validation"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "source record overlap"
        ),
    ):

        validate_specialist_dpo_pair(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                "qwen2.5-0.5b-funccall"
            ),
        )


def test_wrong_model_key_is_refused(
    tmp_path: Path,
):

    train_directory = (
        write_partition(
            root=(
                tmp_path
                / "train-root"
            ),

            partition=(
                "train"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-train-1"
                    ),

                    source_record_id=(
                        "train-1"
                    ),
                )
            ],
        )
    )

    validation_directory = (
        write_partition(
            root=(
                tmp_path
                / "validation-root"
            ),

            partition=(
                "validation"
            ),

            records=[
                dpo_record(
                    dpo_record_id=(
                        "dpo-validation-1"
                    ),

                    source_record_id=(
                        "validation-1"
                    ),
                )
            ],
        )
    )

    train = (
        load_specialist_dpo_partition(
            directory=(
                train_directory
            ),

            expected_partition=(
                "train"
            ),
        )
    )

    validation = (
        load_specialist_dpo_partition(
            directory=(
                validation_directory
            ),

            expected_partition=(
                "validation"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match the requested "
            "training model key"
        ),
    ):

        validate_specialist_dpo_pair(
            train=(
                train
            ),

            validation=(
                validation
            ),

            expected_model_key=(
                "wrong-model"
            ),
        )


def test_dataset_rows_preserve_conversational_prompt(
):

    source = (
        dpo_record(
            dpo_record_id=(
                "dpo-1"
            ),

            source_record_id=(
                "source-1"
            ),
        )
    )

    rows = (
        build_dpo_dataset_rows(
            [
                source
            ]
        )
    )

    assert (
        len(
            rows
        )
        == 1
    )

    row = (
        rows[
            0
        ]
    )

    assert (
        row[
            "prompt"
        ]
        == source.prompt_messages
    )

    assert (
        row[
            "chosen"
        ][
            0
        ][
            "role"
        ]
        == "assistant"
    )

    assert (
        row[
            "chosen"
        ][
            0
        ][
            "content"
        ]
        == source.chosen
    )

    assert (
        row[
            "rejected"
        ][
            0
        ][
            "content"
        ]
        == source.rejected
    )


def test_settings_are_locked_to_phase4a_qlora(
    tmp_path: Path,
):

    settings = (
        SpecialistDpoQloraSettings(
            base_model_path=(
                tmp_path
                / "model"
            ),

            expected_model_key=(
                "qwen2.5-0.5b-funccall"
            ),

            output_root=(
                tmp_path
                / "output"
            ),
        )
    )

    assert (
        settings.bnb_4bit_quant_type
        == "nf4"
    )

    assert (
        settings.lora_target_modules
        == "all-linear"
    )

    assert (
        settings.max_length
        == 1024
    )

    assert (
        settings.per_device_train_batch_size
        == 1
    )

    assert (
        settings.gradient_accumulation_steps
        == 8
    )


def test_non_nf4_quantization_is_refused(
    tmp_path: Path,
):

    with pytest.raises(
        ValueError,
        match=(
            "requires NF4"
        ),
    ):

        SpecialistDpoQloraSettings(
            base_model_path=(
                tmp_path
                / "model"
            ),

            expected_model_key=(
                "qwen2.5-0.5b-funccall"
            ),

            output_root=(
                tmp_path
                / "output"
            ),

            bnb_4bit_quant_type=(
                "fp4"
            ),
        )