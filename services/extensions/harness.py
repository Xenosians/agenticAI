"""Inference-only harness adapters.

Pi/OpenCode run inside a temporary HOME with native tools, MCP, extensions,
skills, sessions and project context disabled.

The temporary HOME is intentional. Credentials are supplied only through the
isolated subprocess environment for the duration of the request.

The existing specialist parser, SemanticGuard and ToolGateway remain the only
execution path.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import signal
import tempfile

from services.extensions.budget import (
    DEFAULT_HARNESS_REQUEST_TOKEN_BUDGET,
    prepare_external_request,
)

from services.extensions.diagnostics import (
    HarnessFailure,
    classify_external_failure,
    sanitize_diagnostic,
)


PI_CREDENTIAL_ENV = {
    "openrouter":
        "OPENROUTER_API_KEY",

    "anthropic":
        "ANTHROPIC_API_KEY",

    "google":
        "GEMINI_API_KEY",

    "groq":
        "GROQ_API_KEY",

    "mistral":
        "MISTRAL_API_KEY",

    "openai":
        "OPENAI_API_KEY",

    "xai":
        "XAI_API_KEY",

    "cerebras":
        "CEREBRAS_API_KEY",
}


MAX_PROMPT_BYTES = 500_000
MAX_STDOUT_BYTES = 2_000_000
MAX_STDERR_BYTES = 64_000
HARNESS_TIMEOUT_SECONDS = 120


def split_harness_model(
    model: str,
) -> tuple[str, str]:
    if not isinstance(
        model,
        str,
    ):
        raise ValueError(
            "Harness model must be text"
        )

    provider, separator, identifier = (
        model.partition(
            "/"
        )
    )

    if (
        not separator
        or not identifier
        or not re.fullmatch(
            r"[a-z0-9_-]+",
            provider,
        )
    ):
        raise ValueError(
            "Use an explicit "
            "provider/model identifier "
            "for this harness"
        )

    return (
        provider,
        identifier,
    )


def pi_credential_env(
    model: str,
) -> str:
    provider, _ = (
        split_harness_model(
            model
        )
    )

    if provider not in PI_CREDENTIAL_ENV:
        raise ValueError(
            "For a Pi API key, use a "
            "supported explicit "
            "provider/model identifier"
        )

    return (
        PI_CREDENTIAL_ENV[
            provider
        ]
    )


def harness_provider(
    model: str,
) -> str:
    provider, _ = (
        split_harness_model(
            model
        )
    )

    return provider


def harness_binary(
    kind: str,
) -> Path:
    configured = os.environ.get(
        (
            "OPENCODE_BINARY"
            if kind == "opencode"
            else "PI_BINARY"
        )
    )

    default = (
        Path.home()
        / ".local/share/agentic-harnesses"
        / "node_modules"
        / ".bin"
        / (
            "opencode"
            if kind == "opencode"
            else "pi"
        )
    )

    binary = (
        Path(configured)
        if configured
        else default
    )

    if not binary.is_file():
        raise ValueError(
            f"{kind} stable harness "
            "is not installed on the AI host"
        )

    return binary.resolve()


async def _read_bounded(
    stream,
    *,
    limit: int,
    label: str,
) -> bytes:
    if stream is None:
        return b""

    content = bytearray()

    while True:
        chunk = await stream.read(
            65_536
        )

        if not chunk:
            break

        content.extend(
            chunk
        )

        if len(content) > limit:
            raise ValueError(
                f"Harness {label} "
                "exceeds size limit"
            )

    return bytes(
        content
    )


def _write_pi_model_override(
    root: Path,
    *,
    provider: str,
    model_id: str,
    max_output_tokens: int,
) -> None:
    """Bound Pi's provider-side maximum without persisting credentials."""

    config_dir = (
        root
        / "pi"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = {
        "providers": {
            provider: {
                "modelOverrides": {
                    model_id: {
                        "maxTokens":
                            max_output_tokens
                    }
                }
            }
        }
    }

    (
        config_dir
        / "models.json"
    ).write_text(
        json.dumps(
            models,
            separators=(
                ",",
                ":",
            ),
        ),
        encoding="utf-8",
    )


def _log_harness_failure(
    *,
    kind: str,
    provider: str,
    category: str,
    exit_code: int | None,
    diagnostic: str,
) -> None:
    print(
        "[HARNESS] Failure "
        f"backend={kind} "
        f"provider={provider} "
        f"category={category} "
        f"exit_code={exit_code} "
        f"detail={diagnostic!r}"
    )


async def harness_generate(
    profile: dict,
    messages: list[dict],
    max_tokens: int,
) -> str:
    kind = profile[
        "backend"
    ]

    if kind not in {
        "opencode",
        "pi",
    }:
        raise ValueError(
            "Unsupported harness"
        )

    model = profile[
        "model"
    ]

    provider, model_id = (
        split_harness_model(
            model
        )
    )

    binary = harness_binary(
        kind
    )

    # Harnesses have their own system/context overhead.
    # Keep a lower default budget than direct HTTP inference.
    budget = prepare_external_request(
        profile,
        messages,
        max_tokens,
        default_request_budget=(
            int(
                os.environ.get(
                    "EXTERNAL_HARNESS_REQUEST_TOKEN_BUDGET",
                    DEFAULT_HARNESS_REQUEST_TOKEN_BUDGET,
                )
            )
        ),
    )

    print(
        "[HARNESS] Request budget "
        f"backend={kind} "
        f"provider={provider} "
        f"model={model!r} "
        "estimated_input="
        f"{budget.estimated_input_tokens} "
        "output_cap="
        f"{budget.max_output_tokens} "
        "estimated_request="
        f"{budget.estimated_request_tokens} "
        "budget="
        f"{budget.request_token_budget} "
        "dropped_messages="
        f"{budget.dropped_messages}"
    )

    with tempfile.TemporaryDirectory(
        prefix="itsm-harness-"
    ) as temp:
        root = Path(
            temp
        )

        env = {
            "PATH":
                os.environ.get(
                    "PATH",
                    (
                        "/usr/local/bin:"
                        "/usr/bin:"
                        "/bin"
                    ),
                ),

            "HOME":
                temp,

            "XDG_CONFIG_HOME":
                str(
                    root
                    / "config"
                ),

            "XDG_DATA_HOME":
                str(
                    root
                    / "data"
                ),

            "XDG_CACHE_HOME":
                str(
                    root
                    / "cache"
                ),

            "NO_COLOR":
                "1",
        }

        prompt = (
            "Follow this conversation as an inference engine. "
            "Return only the assistant response. "
            "Do not execute tools. Any proposed tool call is "
            "text for the ITSM gateway to validate.\n"
            + json.dumps(
                budget.messages,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        if (
            len(
                prompt.encode(
                    "utf-8"
                )
            )
            > MAX_PROMPT_BYTES
        ):
            raise ValueError(
                "Harness context "
                "exceeds byte limit"
            )

        api_key = (
            profile.get(
                "api_key",
                "",
            )
            or ""
        )

        if kind == "opencode":
            config = {
                "$schema":
                    (
                        "https://opencode.ai/"
                        "config.json"
                    ),

                "permission": {
                    "*":
                        "deny"
                },

                "share":
                    "disabled",

                "autoupdate":
                    False,

                "plugin":
                    [],

                "mcp":
                    {},
            }

            env.update(
                {
                    "OPENCODE_DISABLE_AUTOUPDATE":
                        "1",

                    "OPENCODE_DISABLE_DEFAULT_PLUGINS":
                        "1",

                    "OPENCODE_DISABLE_CLAUDE_CODE":
                        "1",

                    "OPENCODE_DISABLE_LSP_DOWNLOAD":
                        "1",

                    "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX":
                        str(
                            budget
                            .max_output_tokens
                        ),
                }
            )

            if api_key:
                env[
                    "ITSM_HARNESS_API_KEY"
                ] = api_key

                config[
                    "provider"
                ] = {
                    provider: {
                        "options": {
                            "apiKey":
                                (
                                    "{env:"
                                    "ITSM_HARNESS_API_KEY"
                                    "}"
                                )
                        }
                    }
                }

            env[
                "OPENCODE_CONFIG_CONTENT"
            ] = json.dumps(
                config,
                separators=(
                    ",",
                    ":",
                ),
            )

            args = [
                str(
                    binary
                ),
                "run",
                "--format",
                "json",
                "--model",
                model,
            ]

        else:
            env[
                "PI_CODING_AGENT_DIR"
            ] = str(
                root
                / "pi"
            )

            env[
                "PI_SKIP_VERSION_CHECK"
            ] = "1"

            env[
                "PI_TELEMETRY"
            ] = "0"

            _write_pi_model_override(
                root,
                provider=provider,
                model_id=model_id,
                max_output_tokens=(
                    budget
                    .max_output_tokens
                ),
            )

            args = [
                str(
                    binary
                ),
                "--print",
                "--mode",
                "json",
                "--no-tools",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-mcp",
                "--no-context-files",
                "--no-session",
                "--thinking",
                "off",
                "--provider",
                provider,
                "--model",
                model_id,
            ]

            # The provider key exists only in this isolated process
            # environment and is never written to models.json.
            if api_key:
                env[
                    pi_credential_env(
                        model
                    )
                ] = api_key

        process = await (
            asyncio
            .create_subprocess_exec(
                *args,
                cwd=temp,
                env=env,
                stdin=(
                    asyncio
                    .subprocess
                    .PIPE
                ),
                stdout=(
                    asyncio
                    .subprocess
                    .PIPE
                ),
                stderr=(
                    asyncio
                    .subprocess
                    .PIPE
                ),
                start_new_session=(
                    os.name
                    != "nt"
                ),
            )
        )

        async def exchange():
            if process.stdin is None:
                raise ValueError(
                    "Harness stdin "
                    "is unavailable"
                )

            async def feed():
                process.stdin.write(
                    prompt.encode(
                        "utf-8"
                    )
                )

                await (
                    process
                    .stdin
                    .drain()
                )

                process.stdin.close()

            feed_task = (
                asyncio
                .create_task(
                    feed()
                )
            )

            stdout_task = (
                asyncio
                .create_task(
                    _read_bounded(
                        getattr(
                            process,
                            "stdout",
                            None,
                        ),
                        limit=(
                            MAX_STDOUT_BYTES
                        ),
                        label="stdout",
                    )
                )
            )

            stderr_task = (
                asyncio
                .create_task(
                    _read_bounded(
                        getattr(
                            process,
                            "stderr",
                            None,
                        ),
                        limit=(
                            MAX_STDERR_BYTES
                        ),
                        label="stderr",
                    )
                )
            )

            try:
                await feed_task

                stdout, stderr = (
                    await asyncio.gather(
                        stdout_task,
                        stderr_task,
                    )
                )

                await process.wait()

                return (
                    stdout,
                    stderr,
                )

            finally:
                for task in (
                    feed_task,
                    stdout_task,
                    stderr_task,
                ):
                    if not task.done():
                        task.cancel()

        try:
            try:
                output, stderr = (
                    await asyncio.wait_for(
                        exchange(),
                        HARNESS_TIMEOUT_SECONDS,
                    )
                )

            except asyncio.TimeoutError:
                diagnostic = (
                    "Harness subprocess timed out"
                )

                _log_harness_failure(
                    kind=kind,
                    provider=provider,
                    category=(
                        "provider_timeout"
                    ),
                    exit_code=None,
                    diagnostic=diagnostic,
                )

                raise HarnessFailure(
                    kind,
                    provider,
                    "provider_timeout",
                ) from None

        finally:
            if (
                process.returncode
                is None
            ):
                if (
                    os.name
                    != "nt"
                ):
                    os.killpg(
                        process.pid,
                        signal.SIGKILL,
                    )

                else:
                    process.kill()

                await process.wait()

        if process.returncode != 0:
            diagnostic_raw = (
                stderr.decode(
                    "utf-8",
                    errors="replace",
                )
                + "\n"
                + output[
                    -MAX_STDERR_BYTES:
                ].decode(
                    "utf-8",
                    errors="replace",
                )
            )

            diagnostic = (
                sanitize_diagnostic(
                    diagnostic_raw,
                    secrets=[
                        api_key,
                    ],
                )
            )

            category = (
                classify_external_failure(
                    diagnostic
                )
            )

            _log_harness_failure(
                kind=kind,
                provider=provider,
                category=category,
                exit_code=(
                    process.returncode
                ),
                diagnostic=diagnostic,
            )

            raise HarnessFailure(
                kind,
                provider,
                category,
                exit_code=(
                    process.returncode
                ),
            )

        texts: list[str] = []

        for line in (
            output
            .decode(
                "utf-8",
                errors="replace",
            )
            .splitlines()
        ):
            try:
                event = json.loads(
                    line
                )

            except ValueError:
                continue

            if (
                kind == "opencode"
                and event.get(
                    "type"
                )
                == "text"
            ):
                part = (
                    event.get(
                        "part",
                        {},
                    )
                )

                if isinstance(
                    part,
                    dict,
                ):
                    text = part.get(
                        "text",
                        "",
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        texts.append(
                            text
                        )

            if (
                kind == "pi"
                and event.get(
                    "type"
                )
                == "message_end"
            ):
                msg = (
                    event.get(
                        "message",
                        {},
                    )
                )

                if (
                    isinstance(
                        msg,
                        dict,
                    )
                    and msg.get(
                        "role"
                    )
                    == "assistant"
                ):
                    if (
                        msg.get(
                            "stopReason"
                        )
                        in {
                            "length",
                            "error",
                            "aborted",
                            "toolUse",
                        }
                    ):
                        raise ValueError(
                            "Harness response "
                            "did not finish normally"
                        )

                    for part in msg.get(
                        "content",
                        [],
                    ):
                        if (
                            isinstance(
                                part,
                                dict,
                            )
                            and part.get(
                                "type"
                            )
                            == "text"
                        ):
                            text = part.get(
                                "text",
                                "",
                            )

                            if isinstance(
                                text,
                                str,
                            ):
                                texts.append(
                                    text
                                )

            if event.get(
                "type"
            ) in {
                "tool_use",
                "tool_execution_start",
            }:
                raise ValueError(
                    "Harness attempted "
                    "native tool execution"
                )

            if event.get(
                "type"
            ) == "error":
                diagnostic = (
                    sanitize_diagnostic(
                        json.dumps(
                            event
                        ),
                        secrets=[
                            api_key,
                        ],
                    )
                )

                category = (
                    classify_external_failure(
                        diagnostic
                    )
                )

                _log_harness_failure(
                    kind=kind,
                    provider=provider,
                    category=category,
                    exit_code=(
                        process.returncode
                    ),
                    diagnostic=diagnostic,
                )

                raise HarnessFailure(
                    kind,
                    provider,
                    category,
                    exit_code=(
                        process.returncode
                    ),
                )

        result = (
            "".join(
                texts
            )
            .strip()
        )

        if not result:
            raise ValueError(
                "Harness returned "
                "no assistant text"
            )

        return result
