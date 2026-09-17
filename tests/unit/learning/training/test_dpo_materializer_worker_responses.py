import json

import pytest

from learning.evidence.types import (
    PreferenceOption,
)

from tests.unit.learning.training.test_dpo_materializer import (
    materializer,
    record,
)


def read_only_materialized_record(
    result,
) -> dict:

    lines = (
        (
            result.output_directory
            / "records.jsonl"
        )
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

    return (
        json.loads(
            lines[
                0
            ]
        )
    )


def test_hallucinated_rejected_tool_is_valid_negative_evidence(
    tmp_path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_tool=(
                "check_account_status"
            ),

            chosen_tool=(
                "account_status"
            ),

            rejected_arguments={
                "user_id":
                    "jdoe",
            },

            chosen_arguments={
                "user_id":
                    "jdoe",
            },
        )
    )

    result = (
        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "hallucinated-tool"
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
        read_only_materialized_record(
            result
        )
    )

    assert (
        payload[
            "change_type"
        ]
        == "tool"
    )

    rejected = (
        json.loads(
            payload[
                "rejected"
            ]
        )
    )

    chosen = (
        json.loads(
            payload[
                "chosen"
            ]
        )
    )

    assert (
        rejected
        == [
            {
                "name":
                    "check_account_status",

                "arguments": {
                    "user_id":
                        "jdoe",
                },
            }
        ]
    )

    assert (
        chosen
        == [
            {
                "name":
                    "account_status",

                "arguments": {
                    "user_id":
                        "jdoe",
                },
            }
        ]
    )


def test_multicall_negative_materializes_exact_worker_call_set(
    tmp_path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),
        )
    )

    source.correction_type = (
        "tool_selection"
    )

    source.rejected = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                },

                {
                    "name":
                        "unlock_user",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ],
        )
    )

    source.chosen = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                }
            ],
        )
    )

    result = (
        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "multi-call-negative"
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
        read_only_materialized_record(
            result
        )
    )

    assert (
        payload[
            "change_type"
        ]
        == "tool_calls"
    )

    rejected = (
        json.loads(
            payload[
                "rejected"
            ]
        )
    )

    chosen = (
        json.loads(
            payload[
                "chosen"
            ]
        )
    )

    assert (
        rejected
        == [
            {
                "name":
                    "account_status",

                "arguments": {
                    "user_id":
                        "jdoe",
                },
            },

            {
                "name":
                    "unlock_user",

                "arguments": {
                    "user_id":
                        "alice",
                },
            },
        ]
    )

    assert (
        chosen
        == [
            {
                "name":
                    "account_status",

                "arguments": {
                    "user_id":
                        "jdoe",
                },
            }
        ]
    )

    assert (
        payload[
            "source_execution_provenance"
        ][
            "provenance_complete"
        ]
        is True
    )


def test_multicall_negative_may_include_hallucinated_tool(
    tmp_path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),
        )
    )

    source.correction_type = (
        "tool_selection"
    )

    source.rejected = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                },

                {
                    "name":
                        "check_account_locked",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ],
        )
    )

    source.chosen = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                }
            ],
        )
    )

    result = (
        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "multi-call-hallucinated"
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
        read_only_materialized_record(
            result
        )
    )

    rejected = (
        json.loads(
            payload[
                "rejected"
            ]
        )
    )

    assert (
        rejected[
            1
        ][
            "name"
        ]
        == "check_account_locked"
    )


def test_chosen_multicall_behavior_fails_closed(
    tmp_path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),
        )
    )

    source.rejected = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                },

                {
                    "name":
                        "unlock_user",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ],
        )
    )

    source.chosen = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                },

                {
                    "name":
                        "unlock_user",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                },
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "chosen_tool_call_count_invalid"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "chosen-multicall"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_single_call_tool_calls_negative_fails_closed(
    tmp_path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),
        )
    )

    source.rejected = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "unlock_user",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                }
            ],
        )
    )

    source.chosen = (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=[
                {
                    "name":
                        "account_status",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                }
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "rejected_tool_call_count_not_invalid"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "single-call-callset"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )


def test_chosen_hallucinated_tool_still_fails_closed(
    tmp_path,
):

    (
        builder,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    source = (
        record(
            builder=(
                builder
            ),

            profile=(
                profile
            ),

            rejected_tool=(
                "account_status"
            ),

            chosen_tool=(
                "invented_tool"
            ),

            rejected_arguments={
                "user_id":
                    "jdoe",
            },

            chosen_arguments={
                "user_id":
                    "jdoe",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "chosen_tool_not_allowed"
        ),
    ):

        builder.build(
            records=[
                source
            ],

            source_split_id=(
                "chosen-invalid"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "source-hash"
            ),
        )
