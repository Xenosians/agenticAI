import hashlib
import json

from pathlib import (
    Path,
)

import pytest

from learning.datasets import (
    PreferenceDatasetBuilder,
)

from learning.types import (
    PreferenceExample,
    PreferenceOption,
)


def preference_example(
    *,
    example_id: str = (
        "example-1"
    ),
) -> PreferenceExample:

    return (
        PreferenceExample(
            example_id=(
                example_id
            ),

            created_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            trajectory_id=(
                "trajectory-1"
            ),

            correction_id=(
                "correction-1"
            ),

            task_id=(
                "task-1"
            ),

            source=(
                "explicit_user"
            ),

            correction_type=(
                "repository_scope"
            ),

            user_request=(
                "Show me the frontend git status."
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


def test_promote_creates_immutable_dataset_version(
    tmp_path: Path,
):
    builder = (
        PreferenceDatasetBuilder(
            root=(
                tmp_path
                / "datasets"
            )
        )
    )

    manifest = (
        builder.promote(
            examples=[
                preference_example()
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Verified repository correction."
            ),
        )
    )

    assert (
        manifest.version
        == "v000001"
    )

    version_root = (
        tmp_path
        / "datasets"
        / "preference"
        / "v000001"
    )

    assert (
        (
            version_root
            / "manifest.json"
        )
        .exists()
    )

    assert (
        (
            version_root
            / "records.jsonl"
        )
        .exists()
    )


def test_second_promotion_creates_next_version(
    tmp_path: Path,
):
    builder = (
        PreferenceDatasetBuilder(
            root=(
                tmp_path
                / "datasets"
            )
        )
    )

    first = (
        builder.promote(
            examples=[
                preference_example(
                    example_id="example-1"
                )
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "First review."
            ),
        )
    )

    second = (
        builder.promote(
            examples=[
                preference_example(
                    example_id="example-2"
                )
            ],

            promoted_by=(
                "evaluation"
            ),

            promotion_reason=(
                "Second evaluation."
            ),
        )
    )

    assert (
        first.version
        == "v000001"
    )

    assert (
        second.version
        == "v000002"
    )


def test_promotion_does_not_mutate_source_example(
    tmp_path: Path,
):
    example = (
        preference_example()
    )

    original = (
        example.model_copy(
            deep=True
        )
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                tmp_path
                / "datasets"
            )
        )
    )

    builder.promote(
        examples=[
            example
        ],

        promoted_by=(
            "trusted_review"
        ),

        promotion_reason=(
            "Verified."
        ),
    )

    assert (
        example
        == original
    )

    assert (
        example.dataset_eligible
        is False
    )


def test_dataset_record_is_promoted_and_hashed(
    tmp_path: Path,
):
    root = (
        tmp_path
        / "datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                root
            )
        )
    )

    manifest = (
        builder.promote(
            examples=[
                preference_example()
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Verified correction."
            ),
        )
    )

    records_path = (
        root
        / "preference"
        / "v000001"
        / "records.jsonl"
    )

    records_blob = (
        records_path.read_text(
            encoding="utf-8"
        )
    )

    expected_hash = (
        hashlib
        .sha256(
            records_blob.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    assert (
        manifest.content_sha256
        == expected_hash
    )

    record = (
        json.loads(
            records_blob
            .splitlines()[
                0
            ]
        )
    )

    assert (
        record[
            "schema"
        ]
        == (
            "preference-dataset-record.v1"
        )
    )

    assert (
        record[
            "dataset_eligible"
        ]
        is True
    )

    assert (
        record[
            "rejected"
        ][
            "arguments"
        ][
            "repository"
        ]
        == "ai"
    )

    assert (
        record[
            "chosen"
        ][
            "arguments"
        ][
            "repository"
        ]
        == "frontend"
    )


def test_untrusted_source_cannot_promote(
    tmp_path: Path,
):
    builder = (
        PreferenceDatasetBuilder(
            root=(
                tmp_path
                / "datasets"
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "trusted_review or evaluation"
        ),
    ):
        builder.promote(
            examples=[
                preference_example()
            ],

            promoted_by=(
                "explicit_user"
            ),

            promotion_reason=(
                "User corrected it."
            ),
        )


def test_duplicate_examples_are_rejected(
    tmp_path: Path,
):
    builder = (
        PreferenceDatasetBuilder(
            root=(
                tmp_path
                / "datasets"
            )
        )
    )

    example = (
        preference_example()
    )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate preference examples"
        ),
    ):
        builder.promote(
            examples=[
                example,
                example,
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Review."
            ),
        )