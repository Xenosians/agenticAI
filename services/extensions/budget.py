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
import re


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
    """Fail-closed budget error with content-free accounting metadata."""

    def __init__(
        self,
        message: str,
        *,
        request_token_budget: int | None = None,
        output_tokens: int | None = None,
        input_budget: int | None = None,
        mandatory_tokens: int | None = None,
        system_tokens: int | None = None,
        newest_tokens: int | None = None,
        message_count: int | None = None,
    ) -> None:
        super().__init__(
            message
        )

        self.request_token_budget = (
            request_token_budget
        )

        self.output_tokens = (
            output_tokens
        )

        self.input_budget = (
            input_budget
        )

        self.mandatory_tokens = (
            mandatory_tokens
        )

        self.system_tokens = (
            system_tokens
        )

        self.newest_tokens = (
            newest_tokens
        )

        self.message_count = (
            message_count
        )


# Provider-aware request envelopes.
#
# These are application safety envelopes, NOT claims about model context
# windows. They intentionally leave headroom for provider/harness framing.
#
# Groq's currently exercised account tier has shown an 8k TPM boundary, so
# harness requests remain materially below that value.
_PROVIDER_REQUEST_BUDGETS: dict[
    tuple[str, str],
    int,
] = {
    ("http", "groq"):
        7000,

    ("harness", "groq"):
        6000,

    ("http", "openrouter"):
        9000,

    ("harness", "openrouter"):
        8000,
}


def external_provider(
    profile: dict,
    *,
    backend: str,
) -> str:
    """Return a public provider label without inspecting credentials."""

    if backend == "http":
        try:
            from urllib.parse import urlsplit

            hostname = (
                urlsplit(
                    str(
                        profile.get(
                            "url",
                            "",
                        )
                    )
                )
                .hostname
                or ""
            ).lower()

        except (
            TypeError,
            ValueError,
        ):
            hostname = ""

        if hostname == "api.groq.com":
            return "groq"

        if hostname == "openrouter.ai":
            return "openrouter"

        return "custom"

    if backend in {
        "pi",
        "opencode",
    }:
        model = str(
            profile.get(
                "model",
                "",
            )
        )

        provider, separator, _ = (
            model.partition(
                "/"
            )
        )

        if (
            separator
            and provider
        ):
            return provider

    return "unknown"


def _policy_budget_value(
    value: object,
    *,
    label: str,
) -> int:
    try:
        resolved = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        raise ExternalInferenceBudgetError(
            f"Invalid {label}"
        ) from None

    if not (
        MIN_REQUEST_TOKEN_BUDGET
        <= resolved
        <= MAX_REQUEST_TOKEN_BUDGET
    ):
        raise ExternalInferenceBudgetError(
            f"{label} outside supported range"
        )

    return resolved


def default_external_request_budget(
    profile: dict,
    *,
    backend: str,
) -> int:
    """Resolve a conservative backend/provider-specific default.

    Precedence here is:

        provider/backend environment override
        backend-family environment override
        built-in provider policy
        generic backend default

    `prepare_external_request()` still gives an explicit profile
    `request_token_budget` and the legacy
    EXTERNAL_INFERENCE_REQUEST_TOKEN_BUDGET final precedence.
    """

    provider = external_provider(
        profile,
        backend=backend,
    )

    family = (
        "http"
        if backend == "http"
        else "harness"
    )

    provider_env = (
        "EXTERNAL_"
        + re.sub(
            r"[^A-Z0-9]+",
            "_",
            provider.upper(),
        )
        + "_"
        + family.upper()
        + "_REQUEST_TOKEN_BUDGET"
    )

    family_env = (
        "EXTERNAL_HTTP_REQUEST_TOKEN_BUDGET"
        if family == "http"
        else "EXTERNAL_HARNESS_REQUEST_TOKEN_BUDGET"
    )

    for key in (
        provider_env,
        family_env,
    ):
        value = os.environ.get(
            key
        )

        if value is not None:
            return _policy_budget_value(
                value,
                label=key,
            )

    generic_default = (
        DEFAULT_HTTP_REQUEST_TOKEN_BUDGET
        if family == "http"
        else DEFAULT_HARNESS_REQUEST_TOKEN_BUDGET
    )

    return (
        _PROVIDER_REQUEST_BUDGETS.get(
            (
                family,
                provider,
            ),
            generic_default,
        )
    )


def log_external_budget_rejection(
    exc: "ExternalInferenceBudgetError",
    *,
    backend: str,
    provider: str,
    model: str,
) -> None:
    """Emit counts only. Never emit prompt content or credentials."""

    print(
        "[INFERENCE_BUDGET] Rejected "
        f"backend={backend} "
        f"provider={provider} "
        f"model={model!r} "
        f"request_budget={exc.request_token_budget} "
        f"output_cap={exc.output_tokens} "
        f"input_budget={exc.input_budget} "
        f"mandatory_tokens={exc.mandatory_tokens} "
        f"system_tokens={exc.system_tokens} "
        f"newest_tokens={exc.newest_tokens} "
        f"messages={exc.message_count}"
    )


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
            "no input capacity",
            request_token_budget=(
                request_budget
            ),
            output_tokens=(
                output_tokens
            ),
            input_budget=(
                input_budget
            ),
            mandatory_tokens=0,
            system_tokens=0,
            newest_tokens=0,
            message_count=(
                len(
                    messages
                )
            ),
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

    system_tokens = sum(
        estimate_message_tokens(
            message
        )
        for message
        in canonical
        if message[
            "role"
        ]
        == "system"
    )

    newest_tokens = (
        estimate_message_tokens(
            canonical[
                -1
            ]
        )
    )

    if used > input_budget:
        raise ExternalInferenceBudgetError(
            "Mandatory model instructions "
            "and newest request exceed the "
            "external provider input budget",
            request_token_budget=(
                request_budget
            ),
            output_tokens=(
                output_tokens
            ),
            input_budget=(
                input_budget
            ),
            mandatory_tokens=(
                used
            ),
            system_tokens=(
                system_tokens
            ),
            newest_tokens=(
                newest_tokens
            ),
            message_count=(
                len(
                    canonical
                )
            ),
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
