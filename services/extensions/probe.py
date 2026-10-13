"""Safe connectivity probes for external inference providers.

A probe proves reachability/authentication only.

It MUST NOT weaken the normal inference contract. Runtime inference still
uses services.extensions.transport.remote_generate(), where truncated or
missing visible output remains a hard failure.
"""

from __future__ import annotations

import re

from services.extensions.transport import request_json


PROBE_CONTRACT = "provider-probe.v1"

_MAX_PUBLIC_TEXT = 300
_MAX_PROVIDER_NAME = 120
_MAX_MODEL_NAME = 300


class ProviderProbeError(ValueError):
    """Curated provider error safe to cross the public API boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        provider_status: int | None = None,
    ):
        super().__init__(
            message
        )

        self.code = (
            str(
                code
            )[:80]
        )

        self.public_message = (
            str(
                message
            )[:_MAX_PUBLIC_TEXT]
        )

        self.provider_status = (
            provider_status
        )

    def public_detail(
        self,
    ) -> dict:
        detail = {
            "code":
                self.code,

            "message":
                self.public_message,
        }

        if (
            isinstance(
                self.provider_status,
                int,
            )
            and 100
            <= self.provider_status
            <= 599
        ):
            detail[
                "provider_status"
            ] = self.provider_status

        return detail


def _provider_error(
    exc: ValueError,
) -> ProviderProbeError:
    """Translate internal transport failures into stable public categories.

    Never forwards a raw provider response body, credential, authorization
    header, or arbitrary exception representation.
    """

    message = str(
        exc
    )

    match = re.search(
        r"\bHTTP\s+(\d{3})\b",
        message,
    )

    if match:
        status = int(
            match.group(
                1
            )
        )

        if status in {
            401,
            403,
        }:
            return ProviderProbeError(
                "provider_authentication_failed",
                (
                    "The provider rejected the configured "
                    "credential or account access."
                ),
                provider_status=status,
            )

        if status == 429:
            return ProviderProbeError(
                "provider_rate_limited",
                (
                    "The provider accepted the endpoint but "
                    "rejected the probe because of quota or "
                    "rate limits."
                ),
                provider_status=status,
            )

        if status == 404:
            return ProviderProbeError(
                "provider_endpoint_or_model_not_found",
                (
                    "The provider endpoint or selected model "
                    "was not found."
                ),
                provider_status=status,
            )

        if 500 <= status <= 599:
            return ProviderProbeError(
                "provider_unavailable",
                (
                    "The external provider returned a server "
                    "error while processing the probe."
                ),
                provider_status=status,
            )

        return ProviderProbeError(
            "provider_rejected_request",
            (
                "The external provider rejected the probe "
                "request."
            ),
            provider_status=status,
        )

    lowered = (
        message
        .lower()
    )

    if (
        "connection failed"
        in lowered
    ):
        return ProviderProbeError(
            "provider_connection_failed",
            (
                "The external provider could not be reached "
                "from the AI service."
            ),
        )

    if (
        "invalid json"
        in lowered
    ):
        return ProviderProbeError(
            "provider_invalid_response",
            (
                "The provider was reached but returned an "
                "invalid JSON response."
            ),
        )

    if (
        "endpoint origin"
        in lowered
        or "remote endpoints require https"
        in lowered
        or "endpoint url"
        in lowered
    ):
        return ProviderProbeError(
            "provider_endpoint_rejected",
            (
                "The configured provider endpoint was rejected "
                "by the server's endpoint policy."
            ),
        )

    return ProviderProbeError(
        "provider_probe_failed",
        (
            "The provider probe failed before a valid "
            "completion response was received."
        ),
    )


def _bounded_text(
    value,
    limit: int,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        return ""

    return (
        value
        .strip()
    )[:limit]


async def remote_probe(
    profile: dict,
    *,
    max_tokens: int = 256,
) -> dict:
    """Probe an OpenAI-compatible provider without accepting model authority.

    A successful HTTP completion proves that the provider processed the
    request. A reasoning-heavy model exhausting the small probe budget is
    therefore still "reachable".

    This function never produces a tool proposal and is never used by normal
    agent inference.
    """

    model = (
        profile.get(
            "model",
            "",
        )
    )

    url = (
        profile.get(
            "url",
            "",
        )
    )

    if (
        not isinstance(
            model,
            str,
        )
        or not model.strip()
    ):
        raise ProviderProbeError(
            "provider_model_missing",
            (
                "A provider model identifier is required "
                "before testing the connection."
            ),
        )

    if (
        not isinstance(
            url,
            str,
        )
        or not url.strip()
    ):
        raise ProviderProbeError(
            "provider_endpoint_missing",
            (
                "A provider API base URL is required "
                "before testing the connection."
            ),
        )

    token_budget = max(
        32,
        min(
            int(
                max_tokens
            ),
            512,
        ),
    )

    body = {
        "model":
            model.strip(),

        "messages": [
            {
                "role":
                    "user",

                "content":
                    (
                        "Reply with exactly: "
                        "OPENROUTER_OK"
                    ),
            }
        ],

        "max_tokens":
            token_budget,

        "stream":
            False,
    }

    try:
        data = await request_json(
            url.rstrip(
                "/"
            )
            + "/chat/completions",
            payload=body,
            api_key=profile.get(
                "api_key",
                "",
            ),
        )

    except ValueError as exc:
        raise _provider_error(
            exc
        ) from None

    if not isinstance(
        data,
        dict,
    ):
        raise ProviderProbeError(
            "provider_invalid_response",
            (
                "The provider was reached but did not return "
                "a completion object."
            ),
        )

    choices = data.get(
        "choices"
    )

    if (
        not isinstance(
            choices,
            list,
        )
        or not choices
        or not isinstance(
            choices[0],
            dict,
        )
    ):
        raise ProviderProbeError(
            "provider_invalid_response",
            (
                "The provider was reached but returned an "
                "unsupported completion response."
            ),
        )

    choice = choices[
        0
    ]

    message = choice.get(
        "message"
    )

    if not isinstance(
        message,
        dict,
    ):
        raise ProviderProbeError(
            "provider_invalid_response",
            (
                "The provider was reached but returned no "
                "completion message."
            ),
        )

    finish_reason = (
        _bounded_text(
            choice.get(
                "finish_reason"
            ),
            80,
        )
    )

    content = (
        _bounded_text(
            message.get(
                "content"
            ),
            2_000,
        )
    )

    reasoning = (
        _bounded_text(
            message.get(
                "reasoning"
            ),
            2_000,
        )
    )

    result = {
        "schema":
            PROBE_CONTRACT,

        "state":
            "reachable",

        "requested_model":
            _bounded_text(
                model,
                _MAX_MODEL_NAME,
            ),

        "model":
            _bounded_text(
                data.get(
                    "model"
                ),
                _MAX_MODEL_NAME,
            )
            or _bounded_text(
                model,
                _MAX_MODEL_NAME,
            ),

        "finish_reason":
            finish_reason
            or "unknown",
    }

    provider = (
        _bounded_text(
            data.get(
                "provider"
            ),
            _MAX_PROVIDER_NAME,
        )
    )

    if provider:
        result[
            "provider"
        ] = provider

    if content:
        result[
            "response"
        ] = content

    if (
        finish_reason
        == "length"
    ):
        result[
            "probe_complete"
        ] = False

        result[
            "message"
        ] = (
            "Provider authenticated and processed the probe, "
            "but the small probe token budget was exhausted "
            "before completion. Runtime inference remains "
            "strict and will reject truncated output."
        )

    elif not content and reasoning:
        result[
            "probe_complete"
        ] = False

        result[
            "message"
        ] = (
            "Provider authenticated and processed the probe. "
            "The selected model returned reasoning metadata "
            "without visible text."
        )

    elif not content:
        result[
            "probe_complete"
        ] = False

        result[
            "message"
        ] = (
            "Provider authenticated and returned a completion "
            "without visible text."
        )

    else:
        result[
            "probe_complete"
        ] = True

        result[
            "message"
        ] = (
            "Provider connection is reachable and returned "
            "visible completion text."
        )

    return result
