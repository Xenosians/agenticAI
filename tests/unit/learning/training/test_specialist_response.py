import json

import pytest

from learning.evidence.types import (
    PreferenceOption,
)

from learning.training.specialist_response import (
    preference_option_calls,
    preference_option_shape,
    serialize_preference_option,
    validate_specialist_preference_pair,
)


ALLOWED_TOOLS = {
    "account_status",
    "unlock_user",
    "reset_password",
}


def singular(
    tool: str,
    user_id: str,
) -> PreferenceOption:

    return (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool=(
                tool
            ),

            arguments={
                "user_id":
                    user_id,
            },
        )
    )


def call_set(
    calls,
) -> PreferenceOption:

    return (
        PreferenceOption(
            agent=(
                "account-specialist"
            ),

            tool_calls=(
                calls
            ),
        )
    )


def test_singular_option_serializes_to_worker_contract(
):

    option = (
        singular(
            "account_status",
            "jdoe",
        )
    )

    assert (
        preference_option_shape(
            option,
            label=(
                "chosen"
            ),
        )
        == "singular"
    )

    serialized = (
        serialize_preference_option(
            option,
            label=(
                "chosen"
            ),
        )
    )

    assert (
        json.loads(
            serialized
        )
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


def test_multicall_option_preserves_complete_call_order(
):

    calls = [
        {
            "name":
                "reset_password",

            "arguments": {
                "user_id":
                    "jdoe",
            },
        },

        {
            "name":
                "reset_password",

            "arguments": {
                "user_id":
                    "alice",
            },
        },
    ]

    option = (
        call_set(
            calls
        )
    )

    assert (
        preference_option_shape(
            option,
            label=(
                "rejected"
            ),
        )
        == "tool_calls"
    )

    assert (
        preference_option_calls(
            option,
            label=(
                "rejected"
            ),
        )
        == calls
    )

    assert (
        json.loads(
            serialize_preference_option(
                option,
                label=(
                    "rejected"
                ),
            )
        )
        == calls
    )


def test_valid_multicall_negative_to_single_call_positive(
):

    rejected = (
        call_set(
            [
                {
                    "name":
                        "reset_password",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                },

                {
                    "name":
                        "reset_password",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ]
        )
    )

    chosen = (
        call_set(
            [
                {
                    "name":
                        "reset_password",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                }
            ]
        )
    )

    shape = (
        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )
    )

    assert (
        shape
        == "tool_calls"
    )


def test_rejected_negative_may_contain_hallucinated_tool(
):

    rejected = (
        call_set(
            [
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
            ]
        )
    )

    chosen = (
        call_set(
            [
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
    )

    assert (
        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )
        == "tool_calls"
    )


def test_singular_rejected_hallucinated_tool_is_valid_negative(
):

    rejected = (
        singular(
            "check_account_status",
            "jdoe",
        )
    )

    chosen = (
        singular(
            "account_status",
            "jdoe",
        )
    )

    assert (
        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )
        == "singular"
    )


def test_chosen_tool_must_be_allowed(
):

    rejected = (
        singular(
            "account_status",
            "jdoe",
        )
    )

    chosen = (
        singular(
            "invented_tool",
            "jdoe",
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "not allowed for the target agent"
        ),
    ):

        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )


def test_chosen_multicall_response_is_rejected(
):

    rejected = (
        call_set(
            [
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
                        "account_status",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ]
        )
    )

    chosen = (
        call_set(
            [
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
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "must contain exactly one tool call"
        ),
    ):

        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )


def test_single_call_tool_calls_negative_is_rejected(
):

    rejected = (
        call_set(
            [
                {
                    "name":
                        "unlock_user",

                    "arguments": {
                        "user_id":
                            "jdoe",
                    },
                }
            ]
        )
    )

    chosen = (
        call_set(
            [
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
    )

    with pytest.raises(
        ValueError,
        match=(
            "invalid multi-call worker response"
        ),
    ):

        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )


def test_mixed_representation_is_rejected(
):

    rejected = (
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
                        "account_status",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ],

            tool=(
                "account_status"
            ),

            arguments={
                "user_id":
                    "jdoe",
            },
        )
    )

    chosen = (
        call_set(
            [
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
    )

    with pytest.raises(
        ValueError,
        match=(
            "mixes tool_calls"
        ),
    ):

        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )


def test_singular_and_call_set_shapes_cannot_be_mixed(
):

    rejected = (
        call_set(
            [
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
                        "account_status",

                    "arguments": {
                        "user_id":
                            "alice",
                    },
                },
            ]
        )
    )

    chosen = (
        singular(
            "account_status",
            "jdoe",
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "different representation shapes"
        ),
    ):

        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )


def test_identical_behavior_is_rejected(
):

    rejected = (
        singular(
            "account_status",
            "jdoe",
        )
    )

    chosen = (
        singular(
            "account_status",
            "jdoe",
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "are identical"
        ),
    ):

        validate_specialist_preference_pair(
            rejected=(
                rejected
            ),

            chosen=(
                chosen
            ),

            allowed_tools=(
                ALLOWED_TOOLS
            ),
        )
