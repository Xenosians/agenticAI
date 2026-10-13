"""Inference-only harness adapters. Native tool execution is disabled.

The existing specialist parser, SemanticGuard and ToolGateway remain the only
execution path. No shell command, filesystem tool or provider key is exposed.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import tempfile
import re

PI_CREDENTIAL_ENV = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY", "cerebras": "CEREBRAS_API_KEY"}


def pi_credential_env(model: str) -> str:
    provider, separator, _ = model.partition("/")
    if not separator or provider not in PI_CREDENTIAL_ENV:
        raise ValueError("For a Pi API key, use a supported explicit provider/model identifier")
    return PI_CREDENTIAL_ENV[provider]


def harness_provider(model: str) -> str:
    provider, separator, identifier = model.partition("/")
    if not separator or not identifier or not re.fullmatch(r"[a-z0-9_-]+", provider):
        raise ValueError("Use an explicit provider/model identifier for this harness")
    return provider


def harness_binary(kind: str) -> Path:
    configured = os.environ.get("OPENCODE_BINARY" if kind == "opencode" else "PI_BINARY")
    default = Path.home() / ".local/share/agentic-harnesses/node_modules/.bin" / ("opencode" if kind == "opencode" else "pi")
    binary = Path(configured) if configured else default
    if not binary.is_file():
        raise ValueError(f"{kind} stable harness is not installed on the AI host")
    return binary.resolve()


async def harness_generate(profile: dict, messages: list[dict], max_tokens: int) -> str:
    kind = profile["backend"]
    binary = harness_binary(kind)
    # Private empty home/cwd prevents discovery of project instructions, MCP
    # servers, extensions, sessions, credentials or .env from the host workspace.
    with tempfile.TemporaryDirectory(prefix="itsm-harness-") as temp:
        root = Path(temp)
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": temp,
               "XDG_CONFIG_HOME": str(root/"config"), "XDG_DATA_HOME": str(root/"data"),
               "XDG_CACHE_HOME": str(root/"cache"), "NO_COLOR": "1"}
        prompt = "Follow this conversation as an inference engine. Return only the assistant response. " \
                 "Do not execute tools. Any proposed tool call is text for the ITSM gateway to validate.\n" + json.dumps(messages)
        if len(prompt.encode()) > 500_000:
            raise ValueError("Harness context exceeds limit")
        if kind == "opencode":
            config = {"$schema": "https://opencode.ai/config.json",
                "permission": {"*": "deny"}, "share": "disabled", "autoupdate": False,
                "plugin": [], "mcp": {}}
            if profile.get("api_key"):
                env["ITSM_HARNESS_API_KEY"] = profile["api_key"]
                config["provider"] = {harness_provider(profile["model"]): {
                    "options": {"apiKey": "{env:ITSM_HARNESS_API_KEY}"}}}
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
            args = [str(binary), "run", "--format", "json", "--model", profile["model"]]
        elif kind == "pi":
            env["PI_CODING_AGENT_DIR"] = str(root/"pi")
            args = [str(binary), "--print", "--mode", "json", "--no-tools", "--no-extensions",
                    "--no-skills", "--no-prompt-templates", "--no-mcp", "--no-context-files",
                    "--no-session", "--model", profile["model"]]
            # Pi provider credentials must be configured explicitly, never
            # inherited from the app/provider environment.
            if profile.get("api_key"):
                env[pi_credential_env(profile["model"])] = profile["api_key"]
        else:
            raise ValueError("Unsupported harness")
        process = await asyncio.create_subprocess_exec(*args, cwd=temp, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=(os.name != "nt"))
        async def exchange():
            async def feed():
                process.stdin.write(prompt.encode())
                await process.stdin.drain()
                process.stdin.close()
            writer = asyncio.create_task(feed())
            output = bytearray()
            try:
                while chunk := await process.stdout.read(65536):
                    output.extend(chunk)
                    if len(output) > 2_000_000:
                        raise ValueError("Harness output exceeds limit")
                await writer
                await process.wait()
                return bytes(output)
            finally:
                writer.cancel()
        try:
            output = await asyncio.wait_for(exchange(), 120)
        finally:
            if process.returncode is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()
        if process.returncode != 0:
            raise ValueError("Harness failed; check provider access and model availability")
        texts = []
        for line in output.decode().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if kind == "opencode" and event.get("type") == "text":
                texts.append(event.get("part", {}).get("text", ""))
            if kind == "pi" and event.get("type") == "message_end":
                msg = event.get("message", {})
                if msg.get("role") == "assistant":
                    if msg.get("stopReason") in {"length", "error", "aborted", "toolUse"}:
                        raise ValueError("Harness response did not finish normally")
                    texts.extend(p.get("text", "") for p in msg.get("content", []) if p.get("type") == "text")
            if event.get("type") in {"tool_use", "tool_execution_start"}:
                raise ValueError("Harness attempted native tool execution")
            if event.get("type") == "error":
                raise ValueError("Harness reported a provider error")
        result = "".join(texts).strip()
        if not result:
            raise ValueError("Harness returned no assistant text")
        return result
