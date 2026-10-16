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

from services.extensions.admission import admit_external_request, record_external_usage
from services.extensions.options import reasoning_effort

from services.extensions.budget import (
    ExternalInferenceBudgetError,
    default_external_request_budget,
    log_external_budget_rejection,
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


def _resolve_node_runtime() -> Path:
    """Resolve Pi's real Node executable before HOME isolation.

    `node` may itself be an asdf/mise/etc. shim. Such shims can stop working
    after the harness switches HOME and cwd to an empty temporary directory.

    Ask the currently working host Node process for process.execPath, then use
    that absolute executable path for the isolated Pi child.
    """

    configured = os.environ.get(
        "PI_NODE_BINARY",
        "",
    ).strip()

    if configured:
        candidate = Path(
            configured
        ).expanduser()

        if (
            not candidate.is_absolute()
            or not candidate.is_file()
            or not os.access(
                candidate,
                os.X_OK,
            )
        ):
            raise ValueError(
                "PI_NODE_BINARY must point to "
                "an executable Node binary"
            )

        return candidate.resolve()

    import subprocess

    try:
        completed = subprocess.run(
            [
                "node",
                "-p",
                "process.execPath",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
            env=os.environ.copy(),
        )

    except (
        OSError,
        subprocess.TimeoutExpired,
    ):
        raise ValueError(
            "Pi requires a working Node runtime "
            "on the AI host"
        ) from None

    if completed.returncode != 0:
        raise ValueError(
            "Pi requires a working Node runtime "
            "on the AI host"
        )

    value = (
        completed.stdout
        .strip()
    )

    if not value:
        raise ValueError(
            "Unable to resolve Pi Node runtime"
        )

    candidate = Path(
        value
    )

    if (
        not candidate.is_absolute()
        or not candidate.is_file()
        or not os.access(
            candidate,
            os.X_OK,
        )
    ):
        raise ValueError(
            "Resolved Pi Node runtime is invalid"
        )

    return candidate.resolve()


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


def _write_pi_model_config(
    root: Path,
    *,
    provider: str,
    model_id: str,
    max_output_tokens: int,
    reasoning: bool = False,
) -> None:
    """Create the minimum isolated Pi model catalog for one request.

    modelOverrides cannot register an unknown model. Pi explicitly ignores
    override IDs that are not already present in its model registry.

    The Agentic harness uses a fresh PI_CODING_AGENT_DIR for every invocation,
    so dynamic/cached provider catalogs such as OpenRouter cannot be assumed to
    exist.

    Register exactly the selected model with `models` instead. Credentials
    remain exclusively in the subprocess environment and are never persisted
    in models.json.
    """

    config_dir = (
        root
        / "pi"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "$schema":
            "https://pi.dev/schemas/models.schema.json",

        "providers": {
            provider: {
                "models": [
                    {
                        "id":
                            model_id,

                        "maxTokens":
                            max_output_tokens,
                    }
                ]
            }
        },
    }

    if reasoning:
        # A models[] entry replaces Pi's registry metadata. Its default for
        # omitted `reasoning` is false, which would silently ignore --thinking.
        config["providers"][provider]["models"][0]["reasoning"] = True

    (
        config_dir
        / "models.json"
    ).write_text(
        json.dumps(
            config,
            separators=(
                ",",
                ":",
            ),
        ),
        encoding="utf-8",
    )

    # Admission reserves one provider attempt. Pi enables up to three agent
    # retries by default; provider SDK retries are a separate setting.
    (config_dir / "settings.json").write_text(
        json.dumps({"retry": {"enabled": False, "maxRetries": 0,
                              "provider": {"maxRetries": 0}}}, separators=(",", ":")),
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
    effort = reasoning_effort(profile)
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

    # Harness/provider envelopes remain conservative and are
    # resolved before any provider request is attempted.
    try:
        budget = prepare_external_request(
            profile,
            messages,
            max_tokens,
            default_request_budget=(
                default_external_request_budget(
                    profile,
                    backend=kind,
                )
            ),
        )

    except ExternalInferenceBudgetError as exc:
        log_external_budget_rejection(
            exc,
            backend=kind,
            provider=provider,
            model=model,
        )

        raise

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
            # Resolve Node before entering the isolated HOME. The host's
            # `node` command may be an asdf/mise shim that cannot resolve
            # a version after HOME/cwd are replaced.
            node_runtime = (
                _resolve_node_runtime()
            )

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

            _write_pi_model_config(
                root,
                provider=provider,
                model_id=model_id,
                max_output_tokens=(
                    budget
                    .max_output_tokens
                ),
                reasoning=effort is not None,
            )

            args = [
                str(
                    node_runtime
                ),
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
                effort or "off",
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

        admission = await admit_external_request(profile, budget.estimated_request_tokens)

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
                    usage = msg.get("usage")
                    if isinstance(usage, dict):
                        record_external_usage(admission, usage.get("totalTokens"))
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
                        stop_reason = msg["stopReason"]
                        # Pi can exit successfully while the provider reports
                        # an unsuccessful assistant turn. Never accept its
                        # partial content or expose provider diagnostics to
                        # the public exception/job result.
                        error_message = msg.get("errorMessage", "")
                        diagnostic = sanitize_diagnostic(
                            f"Pi assistant stop_reason={stop_reason}; "
                            + (error_message if isinstance(error_message, str) else ""),
                            secrets=[api_key],
                        )
                        category = (
                            classify_external_failure(diagnostic)
                            if stop_reason == "error"
                            else {
                                "length": "provider_output_truncated",
                                "aborted": "provider_aborted",
                                "toolUse": "harness_native_tools_rejected",
                            }[stop_reason]
                        )
                        _log_harness_failure(
                            kind=kind,
                            provider=provider,
                            category=category,
                            exit_code=process.returncode,
                            diagnostic=diagnostic,
                        )
                        raise HarnessFailure(
                            kind,
                            provider,
                            category,
                            exit_code=process.returncode,
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
