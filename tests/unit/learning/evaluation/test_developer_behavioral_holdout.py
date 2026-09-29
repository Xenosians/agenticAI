import hashlib
import json

from pathlib import Path

import pytest

from learning.continual.storage import (
    load_jsonl_models,
)

from learning.evaluation.developer_behavioral_holdout import (
    DeveloperBehavioralHoldoutRecord,
    materialize_developer_behavioral_holdout,
)

from learning.evaluation.developer_holdout import (
    DeveloperHoldoutRecord,
    materialize_developer_holdout,
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
                            "developer-behavior-test",

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
                        },
                    }
                ],
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
                            "developer-behavior-test",

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
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _rows() -> list[dict]:

    return [
        {
            "instance_id":
                f"instance-{index}",

            "repo":
                "owner/repo",

            "base_commit":
                f"{index:040x}",

            "license":
                "MIT",

            "language":
                "Python",

            "problem_statement":
                f"Fix bug {index}.",

            "patch":
                (
                    "diff --git a/a.py b/a.py\n"
                    f"+# gold patch {index}"
                ),

            "test_patch":
                (
                    "diff --git a/test_a.py b/test_a.py\n"
                    f"+# evaluator test {index}"
                ),

            "image_name":
                (
                    "docker.io/example/"
                    f"repo:{index}"
                ),

            "FAIL_TO_PASS": [
                f"test_bug_{index}"
            ],

            "PASS_TO_PASS": [
                "test_existing_behavior"
            ],

            "install_config": {
                "base_image_name":
                    "python_3.12",

                "install": [
                    "pip install -e ."
                ],

                "test_cmd":
                    "pytest -q",
            },

            "interface":
                "Function: fix_bug()",
        }

        for index
        in range(
            100
        )
    ]


def _write_rows(
    path: Path,
    rows: list[
        dict
    ],
) -> None:

    path.write_text(
        "".join(
            (
                json.dumps(
                    row
                )
                + "\n"
            )

            for row
            in rows
        ),
        encoding="utf-8",
    )


def _materialize_parent(
    *,
    tmp_path: Path,
    corpus: Path,
    registry: Path,
    decontamination: Path,
):

    return (
        materialize_developer_holdout(
            registry_path=(
                registry
            ),

            source_id=(
                "developer-behavior-test"
            ),

            scan_start=0,

            scan_limit=100,

            output_root=(
                tmp_path
                / "loss"
            ),

            decontamination_path=(
                decontamination
            ),
        )
    )


def test_behavioral_holdout_matches_parent_and_excludes_gold(
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

    rows = (
        _rows()
    )

    _write_rows(
        corpus,
        rows,
    )

    _write_registry(
        registry,
        corpus,
    )

    _write_decontamination(
        decontamination
    )

    parent = (
        _materialize_parent(
            tmp_path=tmp_path,
            corpus=corpus,
            registry=registry,
            decontamination=(
                decontamination
            ),
        )
    )

    behavioral = (
        materialize_developer_behavioral_holdout(
            holdout_directory=(
                Path(
                    parent
                    .manifest
                    .output_directory
                )
            ),

            registry_path=(
                registry
            ),

            decontamination_path=(
                decontamination
            ),

            output_root=(
                tmp_path
                / "behavioral"
            ),
        )
    )

    parent_records = (
        load_jsonl_models(
            Path(
                parent.records_path
            ),

            DeveloperHoldoutRecord,
        )
    )

    behavioral_records = (
        load_jsonl_models(
            Path(
                behavioral.records_path
            ),

            DeveloperBehavioralHoldoutRecord,
        )
    )

    assert [
        item.source_identity

        for item
        in behavioral_records
    ] == [
        item.source_identity

        for item
        in parent_records
    ]

    assert (
        behavioral
        .manifest
        .holdout_count
        == parent
        .manifest
        .holdout_count
    )

    assert (
        behavioral
        .manifest
        .identity_verified
        is True
    )

    assert (
        behavioral
        .manifest
        .gold_patch_included
        is False
    )

    assert (
        behavioral
        .manifest
        .training_authorized
        is False
    )

    assert (
        behavioral
        .manifest
        .promotion_authorized
        is False
    )

    serialized = (
        Path(
            behavioral.records_path
        )
        .read_text(
            encoding="utf-8"
        )
    )

    row_by_id = {
        row[
            "instance_id"
        ]:
            row

        for row
        in rows
    }

    for record in behavioral_records:

        source = (
            row_by_id[
                record.source_record_id
            ]
        )

        assert (
            source[
                "patch"
            ]
            not in serialized
        )

        assert (
            record.reference_patch_sha256
            == hashlib.sha256(
                source[
                    "patch"
                ]
                .strip()
                .encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        assert (
            record.test_patch
            == source[
                "test_patch"
            ]
        )

        assert (
            record.fail_to_pass
            == source[
                "FAIL_TO_PASS"
            ]
        )

        assert (
            record.pass_to_pass
            == source[
                "PASS_TO_PASS"
            ]
        )


def test_behavioral_holdout_fails_closed_when_parent_metadata_missing(
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

    rows = (
        _rows()
    )

    _write_rows(
        corpus,
        rows,
    )

    _write_registry(
        registry,
        corpus,
    )

    _write_decontamination(
        decontamination
    )

    parent = (
        _materialize_parent(
            tmp_path=tmp_path,
            corpus=corpus,
            registry=registry,
            decontamination=(
                decontamination
            ),
        )
    )

    parent_records = (
        load_jsonl_models(
            Path(
                parent.records_path
            ),

            DeveloperHoldoutRecord,
        )
    )

    target_id = (
        parent_records[
            0
        ]
        .source_record_id
    )

    for row in rows:

        if (
            row[
                "instance_id"
            ]
            == target_id
        ):

            row.pop(
                "test_patch"
            )

            break

    _write_rows(
        corpus,
        rows,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Behavioral metadata is incomplete"
        ),
    ):

        materialize_developer_behavioral_holdout(
            holdout_directory=(
                Path(
                    parent
                    .manifest
                    .output_directory
                )
            ),

            registry_path=(
                registry
            ),

            decontamination_path=(
                decontamination
            ),

            output_root=(
                tmp_path
                / "behavioral"
            ),
        )
