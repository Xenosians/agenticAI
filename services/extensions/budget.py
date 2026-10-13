"""Conservative budgeting for external inference.

The caller's requested max token value is not automatically safe for a hosted
provider. Hosted services may apply TPM/account-tier limits that are much lower
than a model's context window.

This module deliberately uses a provider-neutral approximate token estimator.
It is not a tokenizer and must be conservative.

Mandatory model instructions and the newest request are never silently
truncated. Old conversation turns are dropped first.

Retrieved evidence is separately bounded before it reaches this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os


DEFAULT_HTTP_REQUEST_TOKEN_BUDGET = 7000
DEFAULT_HARNESS_REQUEST_TOKEN_BUDGET = 5000

DEFAULT_OUTPUT_TOKEN_CAP = 768

MIN_REQUEST_TOKEN_BUDGET = 512
MAX_REQUEST_TOKEN_BUDGET = 262_144

MIN_OUTPUT_TOKEN_CAP = 32
MAX_OUTPUT_TOKEN_CAP = 16_384


class ExternalInferenceBudgetError(
    ValueError
):
    pass


@dataclass(
    frozen=True
)
class ExternalRequestBudget:
    messages: list[dict[str, str]]
    max_output_tokens: int
    estimated_input_tokens: int
    estimated_request_tokens: int
    request_token_budget: int
    dropped_messages: int


def _configured_int(
    profile: dict,
    *,
    profile_key: str,
    env_key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = profile.get(
        profile_key
    )

    if value is None:
        value = os.environ.get(
            env_key
        )

    if value is None:
        value = default

    try:
        resolved = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        raise ExternalInferenceBudgetError(
            f"Invalid {profile_key}"
        ) from None

    if not (
        minimum
        <= resolved
        <= maximum
    ):
        raise ExternalInferenceBudgetError(
            f"{profile_key} outside "
            "supported range"
        )

    return resolved


def estimate_text_tokens(
    text: str,
) -> int:
    if not isinstance(
        text,
        str,
    ):
        raise ExternalInferenceBudgetError(
            "External inference content "
            "must be text"
        )

    # Four characters per token is only a heuristic.
    # The extra constant accounts for message framing.
    return max(
        1,
        math.ceil(
            len(text) / 4
        ),
    )


def estimate_message_tokens(
    message: dict[str, str],
) -> int:
    if not isinstance(
        message,
        dict,
    ):
        raise ExternalInferenceBudgetError(
            "Invalid inference message"
        )

    role = message.get(
        "role"
    )

    content = message.get(
        "content"
    )

    if (
        role
        not in {
            "system",
            "user",
            "assistant",
        }
        or not isinstance(
            content,
            str,
        )
    ):
        raise ExternalInferenceBudgetError(
            "Invalid inference message"
        )

    return (
        estimate_text_tokens(
            content
        )
        + 8
    )


def estimate_messages_tokens(
    messages: list[dict[str, str]],
) -> int:
    return sum(
        estimate_message_tokens(
            message
        )
        for message in messages
    )


def prepare_external_request(
    profile: dict,
    messages: list[dict[str, str]],
    requested_max_tokens: int,
    *,
    default_request_budget: int,
) -> ExternalRequestBudget:
    request_budget = _configured_int(
        profile,
        profile_key=(
            "request_token_budget"
        ),
        env_key=(
            "EXTERNAL_INFERENCE_REQUEST_TOKEN_BUDGET"
        ),
        default=default_request_budget,
        minimum=MIN_REQUEST_TOKEN_BUDGET,
        maximum=MAX_REQUEST_TOKEN_BUDGET,
    )

    output_cap = _configured_int(
        profile,
        profile_key=(
            "max_output_tokens"
        ),
        env_key=(
            "EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS"
        ),
        default=DEFAULT_OUTPUT_TOKEN_CAP,
        minimum=MIN_OUTPUT_TOKEN_CAP,
        maximum=MAX_OUTPUT_TOKEN_CAP,
    )

    try:
        requested = int(
            requested_max_tokens
        )
    except (
        TypeError,
        ValueError,
    ):
        raise ExternalInferenceBudgetError(
            "Invalid output token request"
        ) from None

    requested = max(
        1,
        requested,
    )

    output_tokens = min(
        requested,
        output_cap,
    )

    input_budget = (
        request_budget
        - output_tokens
    )

    if input_budget <= 0:
        raise ExternalInferenceBudgetError(
            "External request budget leaves "
            "no input capacity"
        )

    canonical: list[
        dict[str, str]
    ] = []

    for message in messages:
        # Validate and detach.
        estimate_message_tokens(
            message
        )

        canonical.append(
            {
                "role":
                    message["role"],

                "content":
                    message["content"],
            }
        )

    if not canonical:
        return ExternalRequestBudget(
            messages=[],
            max_output_tokens=(
                output_tokens
            ),
            estimated_input_tokens=0,
            estimated_request_tokens=(
                output_tokens
            ),
            request_token_budget=(
                request_budget
            ),
            dropped_messages=0,
        )

    mandatory_indices = {
        index
        for index, message
        in enumerate(
            canonical
        )
        if message["role"]
        == "system"
    }

    # Always retain the newest request/turn.
    mandatory_indices.add(
        len(canonical) - 1
    )

    selected = set(
        mandatory_indices
    )

    used = sum(
        estimate_message_tokens(
            canonical[index]
        )
        for index
        in mandatory_indices
    )

    if used > input_budget:
        raise ExternalInferenceBudgetError(
            "Mandatory model instructions "
            "and newest request exceed the "
            "external provider input budget"
        )

    # Prefer recent conversation context.
    for index in range(
        len(canonical) - 1,
        -1,
        -1,
    ):
        if index in selected:
            continue

        size = (
            estimate_message_tokens(
                canonical[index]
            )
        )

        if (
            used
            + size
            <= input_budget
        ):
            selected.add(
                index
            )

            used += size

    bounded = [
        message
        for index, message
        in enumerate(
            canonical
        )
        if index in selected
    ]

    return ExternalRequestBudget(
        messages=bounded,
        max_output_tokens=(
            output_tokens
        ),
        estimated_input_tokens=(
            used
        ),
        estimated_request_tokens=(
            used
            + output_tokens
        ),
        request_token_budget=(
            request_budget
        ),
        dropped_messages=(
            len(canonical)
            - len(bounded)
        ),
    )
