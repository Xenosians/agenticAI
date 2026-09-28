from pathlib import Path

from learning.continual.corpus_materializer import (
    CorpusSnapshotManifest,
    NormalizedCorpusRecord,
)

from learning.continual.storage import (
    immutable_write_json,
    immutable_write_jsonl,
)

from learning.training.developer_corpus_bridge import (
    materialize_developer_sft_snapshot,
)


def test_developer_snapshot_materializes_sft_train_and_validation(
    tmp_path: Path,
):

    base_model = (
        tmp_path
        / "base-model"
    )

    base_model.mkdir()

    (
        base_model
        / "config.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    contract = (
        tmp_path
        / "developer-specialist.md"
    )

    contract.write_text(
        "developer contract",
        encoding="utf-8",
    )

    snapshot_dir = (
        tmp_path
        / "snapshot"
    )

    snapshot_dir.mkdir()

    records = []

    for index in range(
        8
    ):

        records.append(
            NormalizedCorpusRecord(
                record_id=(
                    f"record-{index}"
                ),

                source_id=(
                    "swe-rebench-v2"
                ),

                source_record_id=(
                    f"instance-{index}"
                ),

                upstream_index=index,

                objective="sft",

                target_component=(
                    "developer-specialist"
                ),

                prompt_messages=[
                    {
                        "role":
                            "user",

                        "content":
                            f"Fix issue {index}",
                    }
                ],

                chosen=(
                    f"patch-{index}"
                ),

                verified_outcome=True,

                content_sha256=(
                    f"{index + 1:064x}"
                ),

                metadata={
                    "repository":
                        "owner/repo",

                    "base_commit":
                        f"commit-{index}",

                    "repository_license":
                        "MIT",
                },
            )
        )

    records_sha = (
        immutable_write_jsonl(
            snapshot_dir
            / "records.jsonl",
            records,
        )
    )

    snapshot = (
        CorpusSnapshotManifest(
            snapshot_id=(
                "snapshot-test"
            ),

            created_at=(
                "2026-01-01T00:00:00+00:00"
            ),

            source_id=(
                "swe-rebench-v2"
            ),

            provider=(
                "huggingface"
            ),

            dataset_id=(
                "example/dataset"
            ),

            revision=(
                "a" * 40
            ),

            split="train",

            target_component=(
                "developer-specialist"
            ),

            objectives=[
                "sft"
            ],

            source_config_sha256=(
                "b" * 64
            ),

            decontamination_policy_sha256=(
                "c" * 64
            ),

            cursor_start=0,
            cursor_end=8,

            scanned_count=8,
            accepted_count=8,

            rejected_counts={},

            records_sha256=(
                records_sha
            ),

            training_eligible=False,

            training_blockers=[
                "objective_specific_optimizer_bridge_not_wired"
            ],

            output_directory=str(
                snapshot_dir
            ),
        )
    )

    immutable_write_json(
        snapshot_dir
        / "manifest.json",
        snapshot,
    )

    result = (
        materialize_developer_sft_snapshot(
            snapshot_directory=(
                snapshot_dir
            ),

            target_model_key=(
                "hub-main"
            ),

            base_model_path=(
                base_model
            ),

            target_contract_path=(
                contract
            ),

            max_records=8,

            output_root=(
                tmp_path
                / "materialized"
            ),
        )
    )

    manifest = result.manifest

    assert (
        manifest.target_component
        == "developer-specialist"
    )

    assert (
        manifest.evaluation_contract
        == "developer_sft_loss"
    )

    assert (
        manifest.sft_train_count
        > 0
    )

    assert (
        manifest.sft_validation_count
        > 0
    )

    assert (
        manifest.sft_train_count
        + manifest.sft_validation_count
        == 8
    )

    assert (
        manifest.dpo_train_count
        == 0
    )

    assert (
        manifest.ready_for_training
        is True
    )

    assert (
        manifest.training_authorized
        is False
    )

    assert (
        manifest.promotion_authorized
        is False
    )
