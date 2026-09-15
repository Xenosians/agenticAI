import hashlib
import json

from pathlib import (
    Path,
)

from learning.training_export import (
    TrainingSplitManifest,
)

from learning.training_preflight import (
    DpoQloraPreflight,
    DpoQloraRecipe,
)

from learning.types import (
    DatasetPromotion,
    PreferenceDatasetRecord,
    PreferenceOption,
)


def canonical_record_line(
    record: PreferenceDatasetRecord,
) -> str:

    return (
        json.dumps(
            record.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        + "\n"
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


def record(
    *,
    record_id: str,
    trajectory_id: str,
    correction_id: str,
    request: str,
) -> PreferenceDatasetRecord:

    return (
        PreferenceDatasetRecord(
            record_id=(
                record_id
            ),

            source_example_id=(
                f"example-{record_id}"
            ),

            trajectory_id=(
                trajectory_id
            ),

            correction_id=(
                correction_id
            ),

            task_id=(
                f"task-{record_id}"
            ),

            task_instructions=(
                f"Handle task {record_id}."
            ),

            source=(
                "trusted_review"
            ),

            correction_type=(
                "answer"
            ),

            user_request=(
                request
            ),

            rejected=(
                PreferenceOption(
                    agent=(
                        "developer-specialist"
                    ),

                    tool=(
                        "workspace_git_status"
                    ),

                    arguments={
                        "repository":
                            "frontend"
                    },

                    answer=(
                        "Rejected answer."
                    ),
                )
            ),

            chosen=(
                PreferenceOption(
                    agent=(
                        "developer-specialist"
                    ),

                    tool=(
                        "workspace_git_status"
                    ),

                    arguments={
                        "repository":
                            "frontend"
                    },

                    answer=(
                        "Chosen answer."
                    ),
                )
            ),

            promotion=(
                DatasetPromotion(
                    promoted_at=(
                        "2026-09-15T00:00:00+00:00"
                    ),

                    promoted_by=(
                        "trusted_review"
                    ),

                    reason=(
                        "Synthetic training preflight."
                    ),
                )
            ),

            dataset_eligible=True,
        )
    )


def build_split(
    tmp_path: Path,
) -> Path:

    split_dir = (
        tmp_path
        / "split"
    )

    split_dir.mkdir(
        parents=True
    )

    train_record = (
        record(
            record_id=(
                "train-1"
            ),

            trajectory_id=(
                "trajectory-1"
            ),

            correction_id=(
                "correction-1"
            ),

            request=(
                "Synthetic train request."
            ),
        )
    )

    validation_record = (
        record(
            record_id=(
                "validation-1"
            ),

            trajectory_id=(
                "trajectory-2"
            ),

            correction_id=(
                "correction-2"
            ),

            request=(
                "Synthetic validation request."
            ),
        )
    )

    train_blob = (
        canonical_record_line(
            train_record
        )
    )

    validation_blob = (
        canonical_record_line(
            validation_record
        )
    )

    (
        split_dir
        / "train.jsonl"
    ).write_text(
        train_blob,
        encoding="utf-8",
    )

    (
        split_dir
        / "validation.jsonl"
    ).write_text(
        validation_blob,
        encoding="utf-8",
    )

    manifest = (
        TrainingSplitManifest(
            split_id=(
                "split-test"
            ),

            created_at=(
                "2026-09-15T00:00:01+00:00"
            ),

            source_dataset_id=(
                "preference"
            ),

            source_dataset_version=(
                "v000001"
            ),

            source_content_sha256=(
                "source-hash"
            ),

            split_seed=(
                "test-seed"
            ),

            requested_validation_fraction=(
                0.5
            ),

            actual_validation_fraction=(
                0.5
            ),

            record_count=2,

            train_record_count=1,

            validation_record_count=1,

            unique_request_group_count=2,

            held_out_request_count=0,

            train_sha256=(
                sha256_text(
                    train_blob
                )
            ),

            validation_sha256=(
                sha256_text(
                    validation_blob
                )
            ),

            train_record_ids=[
                "train-1"
            ],

            validation_record_ids=[
                "validation-1"
            ],
        )
    )

    (
        split_dir
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
        split_dir
    )


def build_model_directory(
    tmp_path: Path,
) -> Path:

    model_path = (
        tmp_path
        / "model"
    )

    model_path.mkdir(
        parents=True
    )

    (
        model_path
        / "config.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer_config.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    return (
        model_path
    )


def build_preflight(
    tmp_path: Path,
) -> DpoQloraPreflight:

    split_dir = (
        build_split(
            tmp_path
        )
    )

    model_path = (
        build_model_directory(
            tmp_path
        )
    )

    recipe = (
        DpoQloraRecipe(
            base_model_path=(
                model_path
            ),

            output_root=(
                tmp_path
                / "training"
            ),

            require_cuda=False,

            require_bfloat16=False,
        )
    )

    return (
        DpoQloraPreflight(
            split_dir=(
                split_dir
            ),

            recipe=(
                recipe
            ),

            required_packages=[],

            inspect_cuda=False,
        )
    )


def test_clean_split_passes_preflight(
    tmp_path: Path,
):

    preflight = (
        build_preflight(
            tmp_path
        )
    )

    report = (
        preflight.run()
    )

    assert (
        report.ready
        is True
    )

    assert (
        report.split_id
        == "split-test"
    )

    assert (
        report.train_record_count
        == 1
    )

    assert (
        report.validation_record_count
        == 1
    )

    assert (
        report.failed_checks
        == []
    )


def test_tampered_training_split_fails(
    tmp_path: Path,
):

    preflight = (
        build_preflight(
            tmp_path
        )
    )

    train_path = (
        preflight.split_dir
        / "train.jsonl"
    )

    train_path.write_text(
        (
            train_path.read_text(
                encoding="utf-8"
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    report = (
        preflight.run()
    )

    assert (
        report.ready
        is False
    )

    assert (
        "train_sha256"
        in report.failed_checks
    )


def test_missing_model_directory_fails(
    tmp_path: Path,
):

    split_dir = (
        build_split(
            tmp_path
        )
    )

    recipe = (
        DpoQloraRecipe(
            base_model_path=(
                tmp_path
                / "missing-model"
            ),

            output_root=(
                tmp_path
                / "training"
            ),

            require_cuda=False,

            require_bfloat16=False,
        )
    )

    preflight = (
        DpoQloraPreflight(
            split_dir=(
                split_dir
            ),

            recipe=(
                recipe
            ),

            required_packages=[],

            inspect_cuda=False,
        )
    )

    report = (
        preflight.run()
    )

    assert (
        report.ready
        is False
    )

    assert (
        "base_model_directory"
        in report.failed_checks
    )


def test_missing_training_dependency_fails(
    tmp_path: Path,
):

    split_dir = (
        build_split(
            tmp_path
        )
    )

    model_path = (
        build_model_directory(
            tmp_path
        )
    )

    recipe = (
        DpoQloraRecipe(
            base_model_path=(
                model_path
            ),

            output_root=(
                tmp_path
                / "training"
            ),

            require_cuda=False,

            require_bfloat16=False,
        )
    )

    preflight = (
        DpoQloraPreflight(
            split_dir=(
                split_dir
            ),

            recipe=(
                recipe
            ),

            required_packages=[
                "definitely_not_a_real_training_package"
            ],

            inspect_cuda=False,
        )
    )

    report = (
        preflight.run()
    )

    assert (
        report.ready
        is False
    )

    assert (
        (
            "dependency_"
            "definitely_not_a_real_training_package"
        )
        in report.failed_checks
    )


def test_dry_run_manifest_written_only_after_pass(
    tmp_path: Path,
):

    preflight = (
        build_preflight(
            tmp_path
        )
    )

    report = (
        preflight.run()
    )

    target = (
        preflight
        .write_dry_run_manifest(
            report
        )
    )

    assert (
        target.is_file()
    )

    payload = (
        json.loads(
            target.read_text(
                encoding="utf-8"
            )
        )
    )

    assert (
        payload[
            "schema"
        ]
        == (
            "training-dry-run-manifest.v1"
        )
    )

    assert (
        payload[
            "split_id"
        ]
        == "split-test"
    )

    assert (
        payload[
            "recipe"
        ][
            "objective"
        ]
        == "dpo"
    )


def test_dry_run_manifest_refuses_failed_preflight(
    tmp_path: Path,
):

    preflight = (
        build_preflight(
            tmp_path
        )
    )

    (
        preflight.split_dir
        / "train.jsonl"
    ).write_text(
        "tampered\n",
        encoding="utf-8",
    )

    report = (
        preflight.run()
    )

    assert (
        report.ready
        is False
    )

    try:

        preflight.write_dry_run_manifest(
            report
        )

    except ValueError as exc:

        assert (
            "preflight is not ready"
            in str(
                exc
            )
        )

    else:

        raise AssertionError(
            "Failed preflight unexpectedly "
            "produced a dry-run manifest."
        )