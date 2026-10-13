#!/usr/bin/env python3
"""Live external-inference preflight.

This is intentionally inference-only.

It does not:
- register an application connection;
- persist credentials;
- execute Agentic ITSM tools;
- call provider-native tools;
- alter the networking specialist;
- mutate Jira, Git, shell or directory state.

Credentials are read from environment variables and are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from services.extensions.providers import (  # noqa: E402
    get_provider_preset,
)
from services.extensions.transport import (  # noqa: E402
    remote_generate,
    validate_url,
)


_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter-free": "OPENROUTER_API_KEY",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an external inference provider "
            "through the same bounded transport used "
            "by Agentic ITSM."
        )
    )

    parser.add_argument(
        "--provider",
        default="groq",
        help=(
            "Built-in provider preset id "
            "(default: groq)."
        ),
    )

    parser.add_argument(
        "--model",
        default="",
        help=(
            "Override the preset model identifier."
        ),
    )

    parser.add_argument(
        "--url",
        default="",
        help=(
            "Override the preset API base URL."
        ),
    )

    parser.add_argument(
        "--key-env",
        default="",
        help=(
            "Environment variable holding the API key. "
            "The key itself is never printed."
        ),
    )

    return parser.parse_args()


async def run(
    args: argparse.Namespace,
) -> int:
    try:
        preset = get_provider_preset(
            args.provider
        )
    except KeyError as exc:
        print(
            f"[FAIL] {exc}",
            file=sys.stderr,
        )
        return 2

    url = (
        args.url.strip()
        or str(preset["url"])
    )

    model = (
        args.model.strip()
        or str(
            preset["recommended_model"]
        ).strip()
    )

    if not model:
        print(
            "[FAIL] A model identifier is required.",
            file=sys.stderr,
        )
        return 2

    key_env = (
        args.key_env.strip()
        or _PROVIDER_KEY_ENV.get(
            args.provider,
            "EXTERNAL_MODEL_API_KEY",
        )
    )

    api_key = os.environ.get(
        key_env,
        "",
    ).strip()

    if (
        bool(preset["requires_api_key"])
        and not api_key
    ):
        print(
            "[FAIL] No API key found in "
            f"{key_env}.",
            file=sys.stderr,
        )
        print(
            "Use a secure shell prompt, for example:",
            file=sys.stderr,
        )
        print(
            f'  read -rsp "API key: " {key_env}; echo; '
            f"export {key_env}",
            file=sys.stderr,
        )
        return 2

    try:
        validated_url = validate_url(
            url
        )

        profile = {
            "backend": "http",
            "model": model,
            "url": validated_url,
            "api_key": api_key,
        }

        response = await remote_generate(
            profile,
            [
                {
                    "role": "system",
                    "content": (
                        "You are being tested only as an "
                        "inference engine. Do not claim to "
                        "have executed tools or changed any "
                        "external system."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Reply briefly with the result of "
                        "17 + 25 and the text PROVIDER_OK."
                    ),
                },
            ],
            128,
        )

    except Exception as exc:
        # Error messages produced by the transport never contain the
        # credential or raw provider response.
        print(
            "[FAIL] External provider preflight failed:",
            str(exc),
            file=sys.stderr,
        )
        return 1

    print("========================================")
    print("External provider preflight")
    print("========================================")
    print(f"provider : {preset['name']}")
    print(f"model    : {model}")
    print(f"url      : {validated_url}")
    print(f"key env  : {key_env} (value hidden)")
    print("----------------------------------------")
    print("response :")
    print(response[:1000])
    print("----------------------------------------")
    print("[PASS] Bounded external inference is reachable.")
    print(
        "[NOTE] This proves inference transport only. "
        "It does not grant or test tool authority."
    )

    return 0


def main() -> int:
    return asyncio.run(
        run(arguments())
    )


if __name__ == "__main__":
    raise SystemExit(main())
