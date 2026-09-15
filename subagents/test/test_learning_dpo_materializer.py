import json

from pathlib import (
    Path,
)

import pytest

from learning.dpo_materializer import (
    SpecialistDpoMaterializer,
)

from learning.types import (
    DatasetPromotion,
    PreferenceDatasetRecord,
    PreferenceOption,
)


def write_agent(
    tmp_path: Path,
) -> Path:

    path = (
        tmp_path
        / "account-specialist.md"
    )

    path.write_text(
        """---
name: account-specialist
description: Handles account operations.
tools:
  - account_status
  - unlock_user
model: qwen2.5-0.5b-funccall
max_steps: 3
---

You are an account specialist.
""",
        encoding="utf-8",
    )

    return path


def record(
    *,
    record_id: str = "record-1",
    rejected_agent: str = "account-specialist",
    chosen_agent: str = "account-specialist",
    rejected_tool: str = "account_status",
    chosen_tool: str = "account_status",
    rejected_arguments: dict | None = None,
    chosen_arguments: dict | None = None,
    rejected_answer: str | None = None,
    chosen_answer: str | None = None,
) -> PreferenceDatasetRecord:

    if rejected_arguments is None:

        rejected_arguments = {
            "user_id":
                "wrong-user"
        }

    if chosen_arguments is None:

        chosen_arguments = {
            "user_id":
                "jdoe"
        }

    return (
        PreferenceDatasetRecord(
            record_id=(
                record_id
            ),

            source_example_id=(
                f"example-{record_id}"
            ),

            trajectory_id=(
                f"trajectory-{record_id}"
            ),

            correction_id=(
                f"correction-{record_id}"
            ),

            task_id=(
                f"task-{record_id}"
            ),

            task_instructions=(
                "Check the requested account."
            ),

            source=(
                "trusted_review"
            ),

            correction_type=(
                "account_identifier"
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            rejected=(
                PreferenceOption(
                    agent=(
                        rejected_agent
                    ),

                    tool=(
                        rejected_tool
                    ),

                    arguments=(
                        rejected_arguments
                    ),

                    answer=(
                        rejected_answer
                    ),
                )
            ),

            chosen=(
                PreferenceOption(
                    agent=(
                        chosen_agent
                    ),

                    tool=(
                        chosen_tool
                    ),

                    arguments=(
                        chosen_arguments
                    ),

                    answer=(
                        chosen_answer
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
                        "Verified synthetic preference."
                    ),
                )
            ),

            dataset_eligible=True,
        )
    )


def materializer(
    tmp_path: Path,
) -> SpecialistDpoMaterializer:

    return (
        SpecialistDpoMaterializer(
            agent_definition_path=(
                write_agent(
                    tmp_path
                )
            ),

            output_root=(
                tmp_path
                / "dpo"
            ),
        )
    )


def test_argument_correction_materializes_for_target_specialist(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    result = (
        builder.build(
            records=[
                record()
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )
    )

    assert (
        result.manifest.record_count
        == 1
    )

    assert (
        result.manifest.target_agent
        == "account-specialist"
    )

    assert (
        result.manifest.target_model_key
        == "qwen2.5-0.5b-funccall"
    )

    payload = (
        json.loads(
            (
                result.output_directory
                / "records.jsonl"
            )
            .read_text(
                encoding="utf-8"
            )
            .splitlines()[
                0
            ]
        )
    )

    assert (
        payload[
            "change_type"
        ]
        == "arguments"
    )

    assert (
        json.loads(
            payload[
                "chosen"
            ]
        )[
            0
        ][
            "arguments"
        ][
            "user_id"
        ]
        == "jdoe"
    )

    assert (
        payload[
            "prompt_messages"
        ][
            1
        ][
            "content"
        ]
        == "Is jdoe locked?"
    )

    assert (
        payload[
            "prompt_messages"
        ][
            2
        ][
            "content"
        ]
        == (
            "Additional task context "
            "from the routing stage:\n"
            "Check the requested account."
        )
    )


def test_tool_correction_materializes(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            rejected_tool=(
                "unlock_user"
            ),

            chosen_tool=(
                "account_status"
            ),

            rejected_arguments={
                "user_id":
                    "jdoe"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },
        )
    )

    result = (
        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )
    )

    payload = (
        json.loads(
            (
                result.output_directory
                / "records.jsonl"
            )
            .read_text(
                encoding="utf-8"
            )
            .splitlines()[
                0
            ]
        )
    )

    assert (
        payload[
            "change_type"
        ]
        == "tool"
    )


def test_router_correction_is_not_specialist_training_data(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            rejected_agent=(
                "access-specialist"
            ),

            chosen_agent=(
                "account-specialist"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_final_answer_correction_is_not_specialist_training_data(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            rejected_arguments={
                "user_id":
                    "jdoe"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },

            rejected_answer=(
                "Wrong answer."
            ),

            chosen_answer=(
                "Correct answer."
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_different_specialist_is_excluded(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            rejected_agent=(
                "access-specialist"
            ),

            chosen_agent=(
                "access-specialist"
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_mixed_tool_and_argument_change_is_refused(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            rejected_tool=(
                "unlock_user"
            ),

            chosen_tool=(
                "account_status"
            ),

            rejected_arguments={
                "user_id":
                    "wrong-user"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_chosen_tool_must_belong_to_target_specialist(
    tmp_path: Path,
):

    builder = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            rejected_tool=(
                "account_status"
            ),

            chosen_tool=(
                "totally_fake_tool"
            ),

            rejected_arguments={
                "user_id":
                    "jdoe"
            },

            chosen_arguments={
                "user_id":
                    "jdoe"
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No target-compatible specialist DPO"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "split-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )