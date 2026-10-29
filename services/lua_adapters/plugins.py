"""Reviewed file-based Lua plugins and typed codec guards.

Protocol behavior lives in the sibling .lua files. Python selects immutable
artifacts, serializes bounded JSON, and validates outputs against the admitted
request before the existing host performs network/process/model I/O.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
from typing import Any

from services.lua_adapters.runtime import AdapterResult, LuaAdapterError, run_adapter

PLUGIN_DIRECTORY = Path(__file__).with_name("plugins")
PLUGIN_FILES = {
    "model.local": "local_model.lua",
    "model.openai": "openai_chat.lua",
    "model.anthropic": "anthropic_messages.lua",
    "model.harness": "harness_prompt.lua",
    "connection.default": "connection.lua",
    "source.default": "source.lua",
    "tool.default": "tool.lua",
}
MAX_PLUGIN_BYTES = 20_000
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginArtifact:
    plugin_id: str
    path: Path
    sha256: str
    source: str
    packaged: bool


def _invalid() -> LuaAdapterError:
    return LuaAdapterError("lua_adapter_configuration_invalid")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        raise LuaAdapterError("lua_adapter_invalid_input") from None


def _read_regular(path: Path, digest: str | None) -> tuple[str, str]:
    """Read one pinned regular file; neither symlinks nor silent fallback apply."""
    if not path.is_absolute() or path.suffix != ".lua" or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise _invalid()
    if digest is not None and (type(digest) is not str or not re.fullmatch(r"[a-f0-9]{64}", digest)):
        raise _invalid()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PLUGIN_BYTES:
                raise _invalid()
            content = handle.read(MAX_PLUGIN_BYTES + 1)
        if len(content) > MAX_PLUGIN_BYTES:
            raise _invalid()
        actual = hashlib.sha256(content).hexdigest()
        if digest is not None and actual != digest:
            raise _invalid()
        return content.decode("utf-8"), actual
    except (OSError, UnicodeError):
        raise _invalid() from None


def _profile_artifact(plugin_id: str) -> dict | None:
    # Lazy import keeps module import inert and avoids a profile/loader cycle.
    from services.lua_adapters.profile import plugin_artifact
    return plugin_artifact(plugin_id)


def resolve_plugin(plugin_id: str, *, source_file: str | Path | None = None,
                   source_sha256: str | None = None) -> PluginArtifact:
    if type(plugin_id) is not str or plugin_id not in PLUGIN_FILES:
        raise _invalid()
    if source_file is not None or source_sha256 is not None:
        if source_file is None or source_sha256 is None:
            raise _invalid()
        artifact = {"file": source_file, "sha256": source_sha256}
    else:
        artifact = _profile_artifact(plugin_id)
    if artifact is None:
        path = PLUGIN_DIRECTORY / PLUGIN_FILES[plugin_id]
        source, actual = _read_regular(path, None)
        return PluginArtifact(plugin_id, path, actual, source, True)
    if type(artifact) is not dict or set(artifact) != {"file", "sha256"} or not isinstance(artifact["file"], (str, Path)):
        raise _invalid()
    path = Path(artifact["file"])
    source, actual = _read_regular(path, artifact["sha256"])
    return PluginArtifact(plugin_id, path, actual, source, False)


def run_plugin(plugin_id: str, payload: Any, *, binding: dict, phase: str = "request",
               source_file: str | Path | None = None, source_sha256: str | None = None) -> AdapterResult:
    artifact = resolve_plugin(plugin_id, source_file=source_file, source_sha256=source_sha256)
    if type(binding) is not dict or not binding or any(key in binding for key in ("plugin_id", "plugin_source_sha256")):
        raise LuaAdapterError("lua_adapter_binding_invalid")
    pinned = {**binding, "plugin_id": plugin_id, "plugin_source_sha256": artifact.sha256}
    result = run_adapter(kind=plugin_id.partition(".")[0], phase=phase, payload=payload,
                         binding=pinned, source=artifact.source, require_mapping=True)
    receipt = {**result.receipt, "plugin_id": plugin_id, "packaged_plugin": artifact.packaged}
    LOG.info("lua_plugin id=%s source=%s input=%s output=%s elapsed_ms=%s", plugin_id,
             receipt["script_sha256"], receipt["input_sha256"], receipt["output_sha256"], receipt["elapsed_ms"])
    return AdapterResult(result.payload, receipt)


async def run_plugin_async(plugin_id: str, payload: Any, **kwargs) -> AdapterResult:
    return await asyncio.to_thread(run_plugin, plugin_id, payload, **kwargs)


def _openai_input(profile, budget, effort, response_format):
    if type(profile) is not dict or type(profile.get("model")) is not str or not profile["model"].strip():
        raise LuaAdapterError("lua_adapter_invalid_input")
    messages, output_cap = budget.messages, budget.max_output_tokens
    if type(messages) is not list or type(output_cap) is not int or output_cap < 1:
        raise LuaAdapterError("lua_adapter_invalid_input")
    response_format = getattr(response_format, "value", response_format)
    if effort is not None and (type(effort) is not str or effort not in {"low", "medium", "high"}):
        raise LuaAdapterError("lua_adapter_invalid_input")
    if response_format is not None and (type(response_format) is not str or not response_format):
        raise LuaAdapterError("lua_adapter_invalid_input")
    return {"model": profile["model"], "messages": messages, "max_tokens": output_cap,
            "options": {"reasoning_effort": effort, "response_format": response_format}}


def _openai_guard(body, original):
    required = {"model", "messages", "max_tokens", "stream"}
    allowed = required | {"reasoning_effort", "response_format"}
    if type(body) is not dict or not required <= set(body) <= allowed:
        raise LuaAdapterError("lua_adapter_output_invalid")
    if body["model"] != original["model"] or _canonical(body["messages"]) != _canonical(original["messages"]):
        raise LuaAdapterError("lua_adapter_binding_changed")
    if type(body["max_tokens"]) is not int or not 1 <= body["max_tokens"] <= original["max_tokens"] or body["stream"] is not False:
        raise LuaAdapterError("lua_adapter_output_invalid")
    options = original["options"]
    expected_fields = required | ({"reasoning_effort"} if options["reasoning_effort"] is not None else set()) | ({"response_format"} if options["response_format"] is not None else set())
    if set(body) != expected_fields:
        raise LuaAdapterError("lua_adapter_output_invalid")
    if (body.get("reasoning_effort") != options["reasoning_effort"]
            or body.get("response_format") != ({"type": options["response_format"]} if options["response_format"] is not None else None)):
        raise LuaAdapterError("lua_adapter_output_invalid")
    return body


def openai_request(profile, budget, *, effort=None, response_format=None, binding=None):
    """Execute the selected Lua chat codec, then validate its admitted wire shape."""
    original = _openai_input(profile, budget, effort, response_format)
    result = run_plugin("model.openai", original,
                        binding=binding if binding is not None else {"model": original["model"], "backend": "http"})
    return _openai_guard(result.payload, original)


async def openai_request_async(profile, budget, **kwargs):
    return await asyncio.to_thread(openai_request, profile, budget, **kwargs)


def harness_prompt(messages, *, binding=None, max_prompt_bytes=500_000):
    """Lua owns prompt framing; Python only serializes and bounds the conversation."""
    if type(messages) is not list or type(max_prompt_bytes) is not int or max_prompt_bytes < 1:
        raise LuaAdapterError("lua_adapter_invalid_input")
    try:
        serialized = json.dumps(messages, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise LuaAdapterError("lua_adapter_invalid_input") from None
    result = run_plugin("model.harness", {"messages_json": serialized},
                        binding=binding if binding is not None else {"protocol": "inference_only_harness"})
    prompt = result.payload
    if type(prompt) is not str or not prompt.strip() or serialized not in prompt:
        raise LuaAdapterError("lua_adapter_output_invalid")
    if len(prompt.encode("utf-8")) > max_prompt_bytes:
        raise LuaAdapterError("lua_adapter_output_too_large")
    return prompt


async def harness_prompt_async(messages, **kwargs):
    return await asyncio.to_thread(harness_prompt, messages, **kwargs)


def _anthropic_input(profile, messages, max_tokens, effort, thinking, max_tokens_limit):
    if (type(profile) is not dict or type(profile.get("model")) is not str or not profile["model"].strip()
            or type(messages) is not list or type(max_tokens) is not int or max_tokens < 1):
        raise LuaAdapterError("lua_adapter_invalid_input")
    if max_tokens_limit is not None and (type(max_tokens_limit) is not int or max_tokens_limit < 1):
        raise LuaAdapterError("lua_adapter_invalid_input")
    if effort is not None and (type(effort) is not str or effort not in {"low", "medium", "high"}):
        raise LuaAdapterError("lua_adapter_invalid_input")
    if thinking is not None and (type(thinking) is not str or thinking not in {"adaptive", "disabled"}):
        raise LuaAdapterError("lua_adapter_invalid_input")
    for row in messages:
        if type(row) is not dict or type(row.get("role")) is not str or row["role"] not in {"system", "user", "assistant"} or type(row.get("content")) is not str:
            raise LuaAdapterError("lua_adapter_invalid_input")
    return {"model": profile["model"], "messages": messages, "max_tokens": max_tokens,
            "max_tokens_limit": max_tokens_limit, "effort": effort, "thinking": thinking}


def _verify_turn_content(turns, originals):
    """Verify ordered content without constructing the provider's merged turns."""
    if type(turns) is not list or not turns or any(
            type(row) is not dict or set(row) != {"role", "content"}
            or type(row["role"]) is not str or row["role"] not in {"user", "assistant"} or type(row["content"]) is not str or not row["content"].strip()
            for row in turns):
        raise LuaAdapterError("lua_adapter_output_invalid")
    if turns[-1]["role"] != "user":
        raise LuaAdapterError("lua_adapter_output_invalid")
    expected = [row for row in originals if row["role"] != "system" and row["content"].strip()]
    index, offset = 0, 0
    if not expected or expected[0]["role"] != "user":
        # The Messages protocol needs an initial user turn when only system or
        # assistant context exists. Its small prelude carries no tool authority.
        if turns[0]["role"] != "user" or len(turns[0]["content"]) > 256:
            raise LuaAdapterError("lua_adapter_binding_changed")
        index = 1
    for row in expected:
        if index >= len(turns) or turns[index]["role"] != row["role"]:
            raise LuaAdapterError("lua_adapter_binding_changed")
        text = turns[index]["content"]
        if offset:
            if text[offset:offset + 2] != "\n\n":
                raise LuaAdapterError("lua_adapter_binding_changed")
            offset += 2
        value = row["content"]
        if text[offset:offset + len(value)] != value:
            raise LuaAdapterError("lua_adapter_binding_changed")
        offset += len(value)
        if offset == len(text):
            index += 1
            offset = 0
    if index != len(turns) or offset:
        raise LuaAdapterError("lua_adapter_binding_changed")


def _anthropic_guard(body, original):
    required = {"model", "max_tokens", "messages", "stream"}
    allowed = required | {"system", "thinking", "output_config"}
    if type(body) is not dict or not required <= set(body) <= allowed:
        raise LuaAdapterError("lua_adapter_output_invalid")
    cap = min(original["max_tokens"], original["max_tokens_limit"]) if original["max_tokens_limit"] is not None else original["max_tokens"]
    if (body["model"] != original["model"] or type(body["max_tokens"]) is not int
            or not 1 <= body["max_tokens"] <= cap or body["stream"] is not False):
        raise LuaAdapterError("lua_adapter_binding_changed")
    systems = "\n\n".join(row["content"] for row in original["messages"] if row["role"] == "system" and row["content"].strip())
    system = body.get("system", "")
    if type(system) is list:
        if (len(system) != 1 or type(system[0]) is not dict
                or set(system[0]) != {"type", "text", "cache_control"} or system[0]["type"] != "text"
                or system[0]["cache_control"] != {"type": "ephemeral"}):
            raise LuaAdapterError("lua_adapter_output_invalid")
        system = system[0]["text"]
    if type(system) is not str or system != systems:
        raise LuaAdapterError("lua_adapter_binding_changed")
    _verify_turn_content(body["messages"], original["messages"])
    thinking = {"type": original["thinking"]} if original["thinking"] is not None else None
    effort = {"effort": original["effort"]} if original["effort"] is not None else None
    if (body.get("thinking") != thinking or body.get("output_config") != effort
            or (thinking is None and "thinking" in body) or (effort is None and "output_config" in body)):
        raise LuaAdapterError("lua_adapter_output_invalid")
    return body


def anthropic_request(profile, messages, max_tokens, *, effort=None, thinking=None,
                      max_tokens_limit=None, binding=None):
    """Execute the Messages Lua codec; capability/HTTP decisions remain host-owned."""
    original = _anthropic_input(profile, messages, max_tokens, effort, thinking, max_tokens_limit)
    result = run_plugin("model.anthropic", original,
                        binding=binding if binding is not None else {"model": original["model"], "api_format": "anthropic"})
    return _anthropic_guard(result.payload, original)


async def anthropic_request_async(profile, messages, max_tokens, **kwargs):
    return await asyncio.to_thread(anthropic_request, profile, messages, max_tokens, **kwargs)


def _visible_openai(data):
    try:
        choices = data["choices"]
        if type(choices) is not list or not choices or type(choices[0]) is not dict:
            raise ValueError
        content = choices[0]["message"]["content"]
        if type(content) is not str or not content.strip():
            raise ValueError
        return {"content": content}
    except (KeyError, IndexError, TypeError, ValueError):
        raise LuaAdapterError("lua_adapter_output_invalid") from None


def _visible_anthropic(data):
    blocks = data.get("content") if type(data) is dict else None
    if type(blocks) is not list:
        raise LuaAdapterError("lua_adapter_output_invalid")
    visible = []
    for block in blocks:
        if type(block) is not dict:
            raise LuaAdapterError("lua_adapter_output_invalid")
        kind = block.get("type")
        if kind == "text" and type(block.get("text")) is str:
            visible.append({"type": "text", "text": block["text"]})
        elif type(kind) is not str or kind not in {"thinking", "redacted_thinking"}:
            raise LuaAdapterError("lua_adapter_output_invalid")
    if not any(row["text"].strip() for row in visible):
        raise LuaAdapterError("lua_adapter_output_invalid")
    return {"content": visible}


def _response(plugin_id, projection, binding):
    result = run_plugin(plugin_id, projection, phase="response",
                        binding=binding if binding is not None else {"protocol": plugin_id})
    if type(result.payload) is not str or not result.payload.strip():
        raise LuaAdapterError("lua_adapter_output_invalid")
    return result.payload


def openai_response(data, *, binding=None):
    """Visible-text decoding only; caller checks actual status/finish reason first."""
    return _response("model.openai", _visible_openai(data), binding)


async def openai_response_async(data, **kwargs):
    return await asyncio.to_thread(openai_response, data, **kwargs)


def anthropic_response(data, *, binding=None):
    """Decode visible blocks after host stop-reason checks; never reveal thinking."""
    return _response("model.anthropic", _visible_anthropic(data), binding)


async def anthropic_response_async(data, **kwargs):
    return await asyncio.to_thread(anthropic_response, data, **kwargs)
