import json

from pathlib import (
    Path,
)

import pytest

from learning.datasets.dataset_loader import (
    PreferenceDatasetLoader,
)

from learning.datasets.dataset_verifier import (
    PreferenceDatasetVerifier,
)

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.evidence.types import (
    PreferenceExample,
    PreferenceOption,
)


def example(
) -> PreferenceExample:

    return (
        PreferenceExample(
            example_id="example-1",

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
                "Check the working tree status for my frontend workspace."
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
        )
    )


def build_dataset(
    tmp_path: Path,
) -> Path:

    root = (
        tmp_path
        / "datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=root
        )
    )

    builder.promote(
        examples=[
            example()
        ],

        promoted_by=(
            "trusted_review"
        ),

        promotion_reason=(
            "Verified."
        ),
    )

    return root


def test_valid_dataset_verifies(
    tmp_path: Path,
):
    root = (
        build_dataset(
            tmp_path
        )
    )

    verifier = (
        PreferenceDatasetVerifier(
            root=root
        )
    )

    result = (
        verifier.verify(
            "v000001"
        )
    )

    assert (
        result.ok
        is True
    )

    assert (
        result.record_count
        == 1
    )

    assert (
        result.errors
        == []
    )


def test_modified_records_fail_hash_verification(
    tmp_path: Path,
):
    root = (
        build_dataset(
            tmp_path
        )
    )

    records = (
        root
        / "preference"
        / "v000001"
        / "records.jsonl"
    )

    records.write_text(
        (
            records.read_text(
                encoding="utf-8"
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    verifier = (
        PreferenceDatasetVerifier(
            root=root
        )
    )

    result = (
        verifier.verify(
            "v000001"
        )
    )

    assert (
        result.ok
        is False
    )

    assert any(
        "SHA-256"
        in error

        for error
        in result.errors
    )


def test_manifest_count_tampering_is_detected(
    tmp_path: Path,
):
    root = (
        build_dataset(
            tmp_path
        )
    )

    manifest_path = (
        root
        / "preference"
        / "v000001"
        / "manifest.json"
    )

    manifest = (
        json.loads(
            manifest_path
            .read_text(
                encoding="utf-8"
            )
        )
    )

    manifest[
        "record_count"
    ] = 99

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    verifier = (
        PreferenceDatasetVerifier(
            root=root
        )
    )

    result = (
        verifier.verify(
            "v000001"
        )
    )

    assert (
        result.ok
        is False
    )

    assert any(
        "record_count"
        in error

        for error
        in result.errors
    )


def test_loader_refuses_corrupt_dataset(
    tmp_path: Path,
):
    root = (
        build_dataset(
            tmp_path
        )
    )

    records = (
        root
        / "preference"
        / "v000001"
        / "records.jsonl"
    )

    records.write_text(
        "garbage\n",
        encoding="utf-8",
    )

    loader = (
        PreferenceDatasetLoader(
            root=root
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Dataset verification failed"
        ),
    ):
        loader.load(
            "v000001"
        )


def test_loader_returns_verified_records(
    tmp_path: Path,
):
    root = (
        build_dataset(
            tmp_path
        )
    )

    loader = (
        PreferenceDatasetLoader(
            root=root
        )
    )

    manifest, records = (
        loader.load(
            "v000001"
        )
    )

    assert (
        manifest.version
        == "v000001"
    )

    assert (
        len(
            records
        )
        == 1
    )

    assert (
        records[
            0
        ]
        .dataset_eligible
        is True
    )