import json

from pathlib import Path

import pytest

from learning.continual.corpus_decontamination import (
    CorpusDecontaminationPolicy,
    decontamination_reason,
)

from learning.evaluation.developer_holdout import (
    DeveloperHoldoutRecord,
    materialize_developer_holdout,
)

from learning.continual.storage import (
    load_jsonl_models,
)

from learning.training.developer_corpus_bridge import (
    materialize_developer_sft_snapshot,
)


def _write_registry(
    path: Path,
    corpus: Path,
) -> None:

    path.write_text(
        json.dumps(
            {
                "schema":
                    "continual-corpus-registry.v1",

                "mix": {
                    "max_external_fraction":
                        0.5,

                    "target_runtime_fraction":
                        0.3,

                    "target_replay_fraction":
                        0.2,
                },

                "sources": [
                    {
                        "source_id":
                            "developer-holdout-test",

                        "provider":
                            "local",

                        "dataset_id":
                            None,

                        "subset":
                            None,

                        "revision":
                            None,

                        "local_path":
                            str(
                                corpus
                            ),

                        "license":
                            "mit",

                        "upstream_url":
                            None,

                        "target_component":
                            "developer-specialist",

                        "audience": [
                            "developer-specialist"
                        ],

                        "objectives": [
                            "sft"
                        ],

                        "trust":
                            "curated",

                        "enabled":
                            True,

                        "training_eligible":
                            True,

                        "evaluation_only":
                            False,

                        "verified_reward":
                            False,

                        "materialization_mode":
                            "local",

                        "contamination_group":
                            "unit-test",

                        "sampling": {
                            "priority":
                                100,

                            "max_records_per_cycle":
                                100,

                            "max_records_per_snapshot":
                                100,

                            "replay_weight":
                                0.0,

                            "max_cycle_fraction":
                                1.0,
                        },

                        "filters": {
                            "languages":
                                [],

                            "include_tasks":
                                [],

                            "exclude_tasks":
                                [],

                            "include_paths":
                                [],

                            "exclude_paths":
                                [],

                            "require_verified_outcome":
                                False,

                            "require_permissive_source_license":
                                False,

                            "deduplicate":
                                True,

                            "secret_scan":
                                True,

                            "benchmark_decontamination":
                                True,
                        },

                        "metadata": {
                            "adapter":
                                "issue_patch"
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_decontamination(
    path: Path,
) -> None:

    path.write_text(
        json.dumps(
            {
                "schema":
                    "continual-corpus-decontamination.v1",

                "policies": [
                    {
                        "source_id":
                            "developer-holdout-test",

                        "holdout_modulus":
                            10,

                        "holdout_buckets": [
                            0
                        ],

                        "blocked_instance_ids":
                            [],

                        "blocked_repositories":
                            [],

                        "blocked_repository_commits":
                            [],

                        "metadata": {
                            "purpose":
                                "unit-test"
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_corpus(
    path: Path,
) -> None:

    rows = [
        {
            "instance_id":
                f"instance-{index}",

            "repo":
                "owner/repo",

            "base_commit":
                f"commit-{index}",

            "license":
                "MIT",

            "problem_statement":
                f"Fix bug {index}.",

            "patch":
                (
                    "diff --git a/a.py b/a.py\n"
                    f"+# patch {index}"
                ),
        }

        for index
        in range(
            100
        )
    ]

    path.write_text(
        "".join(
            json.dumps(
                row
            )
            + "\n"

            for row
            in rows
        ),
        encoding="utf-8",
    )


def test_materializes_only_reserved_developer_holdout(
    tmp_path: Path,
):

    corpus = (
        tmp_path
        / "corpus.jsonl"
    )

    registry = (
        tmp_path
        / "registry.json"
    )

    decontamination = (
        tmp_path
        / "decontamination.json"
    )

    _write_corpus(
        corpus
    )

    _write_registry(
        registry,
        corpus,
    )

    _write_decontamination(
        decontamination
    )

    result = (
        materialize_developer_holdout(
            registry_path=(
                registry
            ),

            source_id=(
                "developer-holdout-test"
            ),

            scan_start=0,
            scan_limit=100,

            output_root=(
                tmp_path
                / "holdouts"
            ),

            decontamination_path=(
                decontamination
            ),
        )
    )

    manifest = result.manifest

    assert (
        manifest.evaluation_only
        is True
    )

    assert (
        manifest.training_eligible
        is False
    )

    assert (
        manifest.training_authorized
        is False
    )

    assert (
        manifest.promotion_authorized
        is False
    )

    assert (
        manifest.holdout_count
        > 0
    )

    assert (
        manifest.holdout_count
        < manifest.scanned_count
    )

    records = (
        load_jsonl_models(
            Path(
                result.records_path
            ),
            DeveloperHoldoutRecord,
        )
    )

    assert (
        len(
            records
        )
        == manifest.holdout_count
    )

    policy = (
        CorpusDecontaminationPolicy(
            source_id=(
                "developer-holdout-test"
            ),

            holdout_modulus=10,

            holdout_buckets=[
                0
            ],
        )
    )

    identities = set()

    for record in records:

        assert (
            record.source_identity
            not in identities
        )

        identities.add(
            record.source_identity
        )

        assert (
            decontamination_reason(
                policy=policy,

                source_record_id=(
                    record
                    .source_record_id
                ),

                repository=(
                    record.repository
                ),

                base_commit=(
                    record.base_commit
                ),

                content_sha256=(
                    record
                    .content_sha256
                ),
            )
            == "reserved_developer_holdout"
        )


def test_holdout_materialization_is_not_a_training_snapshot(
    tmp_path: Path,
):

    corpus = (
        tmp_path
        / "corpus.jsonl"
    )

    registry = (
        tmp_path
        / "registry.json"
    )

    decontamination = (
        tmp_path
        / "decontamination.json"
    )

    _write_corpus(
        corpus
    )

    _write_registry(
        registry,
        corpus,
    )

    _write_decontamination(
        decontamination
    )

    result = (
        materialize_developer_holdout(
            registry_path=(
                registry
            ),

            source_id=(
                "developer-holdout-test"
            ),

            scan_start=0,
            scan_limit=100,

            output_root=(
                tmp_path
                / "holdouts"
            ),

            decontamination_path=(
                decontamination
            ),
        )
    )

    with pytest.raises(
        ValueError
    ):

        materialize_developer_sft_snapshot(
            snapshot_directory=(
                Path(
                    result
                    .manifest
                    .output_directory
                )
            ),

            target_model_key=(
                "hub-main"
            ),

            base_model_path=(
                tmp_path
                / "base"
            ),

            target_contract_path=(
                tmp_path
                / "contract.md"
            ),

            token_length_resolver=(
                lambda _record: (
                    10,
                    20,
                    10,
                )
            ),

            max_sequence_tokens=768,
            max_records=8,

            output_root=(
                tmp_path
                / "training"
            ),
        )
