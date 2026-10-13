import json

import pytest

from services.extensions.budget import (
    ExternalInferenceBudgetError,
    prepare_external_request,
)

from services.extensions.diagnostics import (
    classify_external_failure,
    sanitize_diagnostic,
)

from services.extensions.retrieval import (
    bound_retrieval_evidence,
    make_hit,
)

from services.extensions.workflow import (
    admit_workflow,
)

from tools.registry import TOOLS


def test_external_budget_caps_output_and_drops_old_context():
    profile = {
        "request_token_budget":
            512,

        "max_output_tokens":
            64,
    }

    messages = [
        {
            "role":
                "system",

            "content":
                "Follow trusted server policy.",
        },
        {
            "role":
                "user",

            "content":
                "old "
                + (
                    "x"
                    * 5000
                ),
        },
        {
            "role":
                "assistant",

            "content":
                "old reply",
        },
        {
            "role":
                "user",

            "content":
                "show git status",
        },
    ]

    budget = (
        prepare_external_request(
            profile,
            messages,
            8192,
            default_request_budget=(
                7000
            ),
        )
    )

    assert (
        budget.max_output_tokens
        == 64
    )

    assert (
        budget.estimated_request_tokens
        <= 512
    )

    assert (
        budget.messages[0]
        == messages[0]
    )

    assert (
        budget.messages[-1]
        == messages[-1]
    )

    assert (
        messages[1]
        not in budget.messages
    )

    assert (
        budget.dropped_messages
        >= 1
    )


def test_external_budget_never_truncates_mandatory_request():
    profile = {
        "request_token_budget":
            512,

        "max_output_tokens":
            64,
    }

    messages = [
        {
            "role":
                "system",

            "content":
                "x"
                * 4000,
        },
        {
            "role":
                "user",

            "content":
                "y"
                * 4000,
        },
    ]

    with pytest.raises(
        ExternalInferenceBudgetError
    ):
        prepare_external_request(
            profile,
            messages,
            64,
            default_request_budget=(
                7000
            ),
        )


def test_groq_tpm_error_is_classified_without_leaking_key():
    raw = (
        '{"error":{"message":'
        '"Request too large on tokens per minute. '
        'Limit 8000, Requested 16516",'
        '"code":"rate_limit_exceeded"}} '
        "Bearer super-secret-provider-key"
    )

    sanitized = (
        sanitize_diagnostic(
            raw,
            secrets=[
                "super-secret-provider-key"
            ],
        )
    )

    assert (
        "super-secret-provider-key"
        not in sanitized
    )

    assert (
        classify_external_failure(
            sanitized,
            status_code=429,
        )
        == (
            "provider_token_budget_exceeded"
        )
    )


def test_retrieval_budget_keeps_complete_versioned_hits():
    resource = {
        "id":
            "a"
            * 32,

        "source":
            "Owned manual",

        "license":
            "Owned",

        "sha256":
            "b"
            * 64,
    }

    first = make_hit(
        resource,
        1,
        "A"
        * 1800,
    )

    second = make_hit(
        resource,
        2,
        "B"
        * 1800,
    )

    third = make_hit(
        resource,
        3,
        "C"
        * 1800,
    )

    selected = (
        bound_retrieval_evidence(
            [
                first,
                second,
                third,
            ],
            max_hits=3,
            max_content_chars=4000,
        )
    )

    assert (
        len(
            selected
        )
        == 2
    )

    assert (
        selected[0][
            "locator"
        ]
        == first[
            "locator"
        ]
    )

    assert (
        selected[0][
            "content_sha256"
        ]
        == first[
            "content_sha256"
        ]
    )


def test_workflow_plan_is_nonexecuting_and_hash_is_deterministic():
    manifest = {
        "version":
            1,

        "name":
            "Read account twice",

        "nodes": [
            {
                "id":
                    "first",

                "capability":
                    "account_status",

                "arguments": {
                    "user_id":
                        "operator"
                },
            },
            {
                "id":
                    "second",

                "capability":
                    "account_status",

                "arguments": {
                    "user_id":
                        "operator"
                },

                "depends_on": [
                    "first"
                ],
            },
        ],
    }

    first = (
        admit_workflow(
            manifest,
            TOOLS,
        )
    )

    second = (
        admit_workflow(
            manifest,
            TOOLS,
        )
    )

    assert (
        first[
            "execution_enabled"
        ]
        is False
    )

    assert (
        first[
            "retry_unknown_outcome"
        ]
        is False
    )

    assert (
        first[
            "plan_sha256"
        ]
        == second[
            "plan_sha256"
        ]
    )

    assert (
        len(
            first[
                "plan_sha256"
            ]
        )
        == 64
    )


def test_workflow_rejects_unknown_dependency():
    manifest = {
        "version":
            1,

        "name":
            "Bad dependency",

        "nodes": [
            {
                "id":
                    "read",

                "capability":
                    "account_status",

                "arguments": {
                    "user_id":
                        "operator"
                },

                "depends_on": [
                    "missing"
                ],
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="unknown dependency",
    ):
        admit_workflow(
            manifest,
            TOOLS,
        )


def test_workflow_rejects_cycle_independent_of_node_order():
    manifest = {
        "version":
            1,

        "name":
            "Cycle",

        "nodes": [
            {
                "id":
                    "one",

                "capability":
                    "account_status",

                "arguments": {
                    "user_id":
                        "operator"
                },

                "depends_on": [
                    "two"
                ],
            },
            {
                "id":
                    "two",

                "capability":
                    "account_status",

                "arguments": {
                    "user_id":
                        "operator"
                },

                "depends_on": [
                    "one"
                ],
            },
        ],
    }

    with pytest.raises(
        ValueError,
        match="cycle",
    ):
        admit_workflow(
            manifest,
            TOOLS,
        )
