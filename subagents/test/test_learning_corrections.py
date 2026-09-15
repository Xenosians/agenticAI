import json

from pathlib import (
    Path,
)

import pytest

from learning.corrections import (
    CorrectionRecorder,
)


def test_correction_records_rejected_and_chosen_values(
    tmp_path: Path,
):
    output = (
        tmp_path
        / "corrections.jsonl"
    )

    recorder = (
        CorrectionRecorder(
            path=output,
            enabled=True,
        )
    )

    payload = (
        recorder.record(
            trajectory_id=(
                "trajectory-123"
            ),

            correction_type=(
                "repository_scope"
            ),

            values=[
                {
                    "field":
                        "repository",

                    "rejected_value":
                        "ai",

                    "chosen_value":
                        "frontend",
                }
            ],

            note=(
                "User explicitly clarified "
                "that frontend was intended."
            ),
        )
    )

    assert (
        payload
        is not None
    )

    assert (
        payload[
            "schema"
        ]
        == "correction-event.v1"
    )

    assert (
        payload[
            "trajectory_id"
        ]
        == "trajectory-123"
    )

    assert (
        payload[
            "correction_type"
        ]
        == "repository_scope"
    )

    assert (
        payload[
            "source"
        ]
        == "explicit_user"
    )

    assert (
        payload[
            "values"
        ][
            0
        ][
            "rejected_value"
        ]
        == "ai"
    )

    assert (
        payload[
            "values"
        ][
            0
        ][
            "chosen_value"
        ]
        == "frontend"
    )

    assert (
        payload[
            "dataset_eligible"
        ]
        is False
    )


def test_correction_is_written_as_jsonl(
    tmp_path: Path,
):
    output = (
        tmp_path
        / "corrections.jsonl"
    )

    recorder = (
        CorrectionRecorder(
            path=output,
            enabled=True,
        )
    )

    recorder.record(
        trajectory_id="trajectory-1",

        correction_type="argument",

        values=[
            {
                "field":
                    "identifier",

                "rejected_value":
                    "KAN-2",

                "chosen_value":
                    "KAN-1",
            }
        ],
    )

    lines = (
        output
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert (
        len(
            lines
        )
        == 1
    )

    stored = (
        json.loads(
            lines[
                0
            ]
        )
    )

    assert (
        stored[
            "schema"
        ]
        == "correction-event.v1"
    )


def test_correction_is_sanitized(
    tmp_path: Path,
):
    output = (
        tmp_path
        / "corrections.jsonl"
    )

    recorder = (
        CorrectionRecorder(
            path=output,
            enabled=True,
        )
    )

    payload = (
        recorder.record(
            trajectory_id="trajectory-1",

            correction_type="answer",

            values=[
                {
                    "field":
                        "answer",

                    "rejected_value":
                        "email=user@example.com",

                    "chosen_value":
                        "email=other@example.com",
                }
            ],
        )
    )

    assert (
        payload
        is not None
    )

    serialized = (
        json.dumps(
            payload
        )
    )

    assert (
        "user@example.com"
        not in serialized
    )

    assert (
        "other@example.com"
        not in serialized
    )


def test_invalid_correction_type_is_rejected(
    tmp_path: Path,
):
    recorder = (
        CorrectionRecorder(
            path=(
                tmp_path
                / "corrections.jsonl"
            ),

            enabled=True,
        )
    )

    with pytest.raises(
        ValueError
    ):
        recorder.record(
            trajectory_id=(
                "trajectory-1"
            ),

            correction_type=(
                "whatever-the-model-wants"
            ),

            values=[
                {
                    "field":
                        "repository",

                    "rejected_value":
                        "ai",

                    "chosen_value":
                        "frontend",
                }
            ],
        )


def test_empty_correction_values_are_rejected(
    tmp_path: Path,
):
    recorder = (
        CorrectionRecorder(
            path=(
                tmp_path
                / "corrections.jsonl"
            ),

            enabled=True,
        )
    )

    with pytest.raises(
        ValueError
    ):
        recorder.record(
            trajectory_id=(
                "trajectory-1"
            ),

            correction_type=(
                "repository_scope"
            ),

            values=[],
        )


def test_disabled_correction_recorder_writes_nothing(
    tmp_path: Path,
):
    output = (
        tmp_path
        / "corrections.jsonl"
    )

    recorder = (
        CorrectionRecorder(
            path=output,
            enabled=False,
        )
    )

    result = (
        recorder.record(
            trajectory_id=(
                "trajectory-1"
            ),

            correction_type=(
                "repository_scope"
            ),

            values=[
                {
                    "field":
                        "repository",

                    "rejected_value":
                        "ai",

                    "chosen_value":
                        "frontend",
                }
            ],
        )
    )

    assert (
        result
        is None
    )

    assert not (
        output.exists()
    )