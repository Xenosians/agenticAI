import hashlib
import json

from pathlib import (
    Path,
)

import pytest

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.datasets.training_export import (
    PreferenceTrainingSplitExporter,
)

from learning.evidence.types import (
    PreferenceExample,
    PreferenceOption,
)


def example(
    *,
    index: int,
    request: str,
) -> PreferenceExample:

    return (
        PreferenceExample(
            example_id=(
                f"example-{index}"
            ),

            created_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            trajectory_id=(
                f"trajectory-{index}"
            ),

            correction_id=(
                f"correction-{index}"
            ),

            task_id=(
                f"task-{index}"
            ),

            source=(
                "trusted_review"
            ),

            correction_type=(
                "repository_scope"
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
                            "ai",
                    },
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
                            "frontend",
                    },
                )
            ),

            dataset_eligible=False,
        )
    )


def build_dataset(
    tmp_path: Path,
    *,
    requests: list[
        str
    ],
) -> Path:

    root = (
        tmp_path
        / "datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                root
            ),

            # Synthetic unit-test data must not depend on the
            # repository's real held-out suites.
            eval_paths=[],
        )
    )

    builder.promote(
        examples=[
            example(
                index=(
                    index
                ),

                request=(
                    request
                ),
            )

            for (
                index,
                request,
            )
            in enumerate(
                requests,
                start=1,
            )
        ],

        promoted_by=(
            "trusted_review"
        ),

        promotion_reason=(
            "Synthetic verified test data."
        ),
    )

    return root


def load_jsonl(
    path: Path,
) -> list[
    dict
]:

    return [
        json.loads(
            line
        )

        for line
        in path.read_text(
            encoding="utf-8"
        )
        .splitlines()

        if line.strip()
    ]


def test_export_creates_train_validation_and_manifest(
    tmp_path: Path,
):

    dataset_root = (
        build_dataset(
            tmp_path,

            requests=[
                "Synthetic request alpha.",
                "Synthetic request beta.",
                "Synthetic request gamma.",
                "Synthetic request delta.",
                "Synthetic request epsilon.",
            ],
        )
    )

    output_root = (
        tmp_path
        / "exports"
    )

    exporter = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                output_root
            ),
        )
    )

    manifest = (
        exporter.export(
            version="v000001",
            validation_fraction=0.20,
            eval_paths=[],
        )
    )

    target = (
        output_root
        / "preference"
        / "v000001"
        / manifest.split_id
    )

    assert (
        target
        / "train.jsonl"
    ).is_file()

    assert (
        target
        / "validation.jsonl"
    ).is_file()

    assert (
        target
        / "manifest.json"
    ).is_file()

    assert (
        manifest.record_count
        == 5
    )

    assert (
        manifest.train_record_count
        + manifest.validation_record_count
        == 5
    )


def test_split_is_deterministic(
    tmp_path: Path,
):

    requests = [
        "Synthetic request one.",
        "Synthetic request two.",
        "Synthetic request three.",
        "Synthetic request four.",
        "Synthetic request five.",
        "Synthetic request six.",
    ]

    dataset_root = (
        build_dataset(
            tmp_path,
            requests=(
                requests
            ),
        )
    )

    first = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                tmp_path
                / "exports-a"
            ),
        )
        .export(
            version="v000001",
            validation_fraction=0.33,
            split_seed="stable-seed",
            eval_paths=[],
        )
    )

    second = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                tmp_path
                / "exports-b"
            ),
        )
        .export(
            version="v000001",
            validation_fraction=0.33,
            split_seed="stable-seed",
            eval_paths=[],
        )
    )

    assert (
        first.split_id
        == second.split_id
    )

    assert (
        first.train_record_ids
        == second.train_record_ids
    )

    assert (
        first.validation_record_ids
        == second.validation_record_ids
    )

    assert (
        first.train_sha256
        == second.train_sha256
    )

    assert (
        first.validation_sha256
        == second.validation_sha256
    )


def test_equal_normalized_requests_never_cross_splits(
    tmp_path: Path,
):

    dataset_root = (
        build_dataset(
            tmp_path,

            requests=[
                "Synthetic repeated request.",
                "  SYNTHETIC   REPEATED REQUEST. ",
                "Synthetic request beta.",
                "Synthetic request gamma.",
                "Synthetic request delta.",
            ],
        )
    )

    output_root = (
        tmp_path
        / "exports"
    )

    exporter = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                output_root
            ),
        )
    )

    manifest = (
        exporter.export(
            version="v000001",
            validation_fraction=0.40,
            eval_paths=[],
        )
    )

    target = (
        output_root
        / "preference"
        / "v000001"
        / manifest.split_id
    )

    train = (
        load_jsonl(
            target
            / "train.jsonl"
        )
    )

    validation = (
        load_jsonl(
            target
            / "validation.jsonl"
        )
    )

    def normalized(
        value: str,
    ) -> str:

        return (
            " ".join(
                value
                .strip()
                .casefold()
                .split()
            )
        )

    train_requests = {
        normalized(
            item[
                "user_request"
            ]
        )

        for item
        in train
    }

    validation_requests = {
        normalized(
            item[
                "user_request"
            ]
        )

        for item
        in validation
    }

    assert (
        train_requests
        .isdisjoint(
            validation_requests
        )
    )


def test_held_out_intersection_is_rejected_before_export(
    tmp_path: Path,
):

    dataset_root = (
        build_dataset(
            tmp_path,

            requests=[
                "Synthetic safe request.",
                "Synthetic held out request.",
            ],
        )
    )

    eval_path = (
        tmp_path
        / "held-out.jsonl"
    )

    eval_path.write_text(
        json.dumps(
            {
                "schema":
                    "evaluation-case.v1",

                "case_id":
                    "synthetic.heldout",

                "suite":
                    "synthetic",

                "user_request":
                    " SYNTHETIC   HELD OUT REQUEST. ",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = (
        tmp_path
        / "exports"
    )

    exporter = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                output_root
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Held-out evaluation intersection"
        ),
    ):

        exporter.export(
            version="v000001",
            eval_paths=[
                eval_path,
            ],
        )

    assert not (
        output_root
        .exists()
    )


def test_invalid_validation_fraction_is_rejected(
    tmp_path: Path,
):

    dataset_root = (
        build_dataset(
            tmp_path,

            requests=[
                "Synthetic request one.",
                "Synthetic request two.",
            ],
        )
    )

    exporter = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                tmp_path
                / "exports"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "validation_fraction"
        ),
    ):

        exporter.export(
            version="v000001",
            validation_fraction=1.0,
            eval_paths=[],
        )


def test_single_request_group_is_rejected(
    tmp_path: Path,
):

    dataset_root = (
        build_dataset(
            tmp_path,

            requests=[
                "Synthetic same request.",
                " SYNTHETIC   SAME REQUEST. ",
            ],
        )
    )

    exporter = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                tmp_path
                / "exports"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "two unique normalized request groups"
        ),
    ):

        exporter.export(
            version="v000001",
            eval_paths=[],
        )


def test_manifest_hashes_match_exported_files(
    tmp_path: Path,
):

    dataset_root = (
        build_dataset(
            tmp_path,

            requests=[
                "Synthetic request alpha.",
                "Synthetic request beta.",
                "Synthetic request gamma.",
                "Synthetic request delta.",
            ],
        )
    )

    output_root = (
        tmp_path
        / "exports"
    )

    exporter = (
        PreferenceTrainingSplitExporter(
            dataset_root=(
                dataset_root
            ),

            output_root=(
                output_root
            ),
        )
    )

    manifest = (
        exporter.export(
            version="v000001",
            validation_fraction=0.25,
            eval_paths=[],
        )
    )

    target = (
        output_root
        / "preference"
        / "v000001"
        / manifest.split_id
    )

    train_hash = (
        hashlib
        .sha256(
            (
                target
                / "train.jsonl"
            )
            .read_bytes()
        )
        .hexdigest()
    )

    validation_hash = (
        hashlib
        .sha256(
            (
                target
                / "validation.jsonl"
            )
            .read_bytes()
        )
        .hexdigest()
    )

    assert (
        manifest.train_sha256
        == train_hash
    )

    assert (
        manifest.validation_sha256
        == validation_hash
    )