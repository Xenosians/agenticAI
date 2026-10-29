"""Real file-based Lua codecs, pinned selection and host protocol guards."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.lua_adapters import plugins
from services.lua_adapters.runtime import LuaAdapterError

BOUND = {"owner": "fixture-owner", "resource_id": "fixture-connection", "stage": "SPECIALIST_REASONING"}
REAL_PROFILE_ARTIFACT = plugins._profile_artifact


@pytest.fixture(autouse=True)
def no_deployment_profile(monkeypatch):
    monkeypatch.setattr(plugins, "_profile_artifact", lambda _: None)


def select_source(tmp_path, monkeypatch, plugin_id, source):
    path = tmp_path / "chosen.lua"
    path.write_bytes(source.encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(plugins, "_profile_artifact", lambda selected: {"file": str(path), "sha256": digest}
                        if selected == plugin_id else None)
    return path, digest


def budget(messages=None, cap=64):
    return SimpleNamespace(messages=messages if messages is not None else [{"role": "user", "content": "status"}],
                           max_output_tokens=cap)


def refuses(code, call):
    with pytest.raises(LuaAdapterError) as caught:
        call()
    assert caught.value.code == code


def test_packaged_artifacts_are_named_lua_files_with_actual_pinned_execution():
    for plugin_id in plugins.PLUGIN_FILES:
        artifact = plugins.resolve_plugin(plugin_id)
        assert artifact.path.suffix == ".lua" and artifact.packaged is True
        assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        assert "return function(event)" in artifact.source


@pytest.mark.parametrize("plugin_id,kind", [("connection.default", "connection"), ("source.default", "source"), ("tool.default", "tool")])
@pytest.mark.parametrize("phase,payload", [("request", {"arguments": {}, "items": [], "optional": None}),
                                         ("response", {"ok": False, "status": "outcome_unknown", "result": None})])
def test_default_protocol_plugins_preserve_real_failure_and_json_types(plugin_id, kind, phase, payload):
    result = plugins.run_plugin(plugin_id, payload, phase=phase, binding=BOUND)
    assert result.payload == payload
    assert result.receipt["kind"] == kind and result.receipt["plugin_id"] == plugin_id
    assert result.receipt["script_sha256"] == plugins.resolve_plugin(plugin_id).sha256
    assert {"process", "no_new_privileges", "resource_limits", "no_new_descriptors"} <= set(result.receipt["isolation"])


@pytest.mark.parametrize("phase,limit", [("request", 64), ("warm", 0)])
def test_local_model_codec_constructs_the_canonical_native_request(phase, limit):
    payload = {"messages": [] if phase == "warm" else [{"role": "user", "content": "look", "reference": None}],
               "max_new_tokens": limit}
    result = plugins.run_plugin("model.local", payload, phase=phase, binding=BOUND)
    assert result.payload == payload
    assert type(result.payload["messages"]) is list


def test_local_model_loader_and_profile_admission_keep_exact_artifact_data():
    data = {"artifacts": ["/models/example"], "options": {"local_files_only": True, "adapter": None}}
    assert plugins.run_plugin("model.local", data, phase="warm", binding={"loader": "AutoModel", "module": "transformers"}).payload == data
    profile = {"backend": "hf-causal", "model_path": "/models/example", "adapter_path": None}
    assert plugins.run_plugin("model.local", profile, phase="admission", binding=BOUND).payload == profile
    assert plugins.run_plugin("model.local", {"text": "actual generated text"}, phase="response", binding=BOUND).payload == {"text": "actual generated text"}


def test_openai_codec_builds_the_real_inference_only_body_in_lua():
    original = budget()
    result = plugins.openai_request({"model": "selected-model", "api_key": "must-not-enter-lua"}, original, binding=BOUND)
    assert result == {"model": "selected-model", "messages": original.messages, "max_tokens": 64, "stream": False}
    assert not {"tools", "functions", "tool_choice", "api_key"} & set(result)


def test_openai_codec_maps_explicit_format_and_reasoning_options():
    from services.extensions.response_format import HttpResponseFormat
    result = plugins.openai_request({"model": "selected"}, budget(), effort="medium", response_format=HttpResponseFormat.JSON_OBJECT)
    assert result["reasoning_effort"] == "medium"
    assert result["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("change,code", [
    ('body.model="different"', "lua_adapter_binding_changed"),
    ('body.messages[1].content="changed target"', "lua_adapter_binding_changed"),
    ('body.max_tokens=65', "lua_adapter_output_invalid"),
    ('body.max_tokens=true', "lua_adapter_output_invalid"),
    ('body.stream=true', "lua_adapter_output_invalid"),
    ('body.tools=adapter.array()', "lua_adapter_output_invalid"),
    ('body.reasoning_effort="high"', "lua_adapter_output_invalid"),
    ('body.response_format=adapter.null', "lua_adapter_output_invalid"),
])
def test_custom_openai_plugin_cannot_raise_budgets_change_targets_or_add_native_tools(tmp_path, monkeypatch, change, code):
    source = f"""return function(event)
      local body={{model=event.payload.model,messages=event.payload.messages,max_tokens=event.payload.max_tokens,stream=false}}
      {change}
      event.payload=body; return event
    end"""
    select_source(tmp_path, monkeypatch, "model.openai", source)
    refuses(code, lambda: plugins.openai_request({"model": "selected"}, budget()))


def test_a_reviewed_openai_file_can_lower_the_output_cap_without_python_changes(tmp_path, monkeypatch):
    source = """return function(event)
      event.payload={model=event.payload.model,messages=event.payload.messages,max_tokens=16,stream=false}
      return event
    end"""
    select_source(tmp_path, monkeypatch, "model.openai", source)
    assert plugins.openai_request({"model": "selected"}, budget())["max_tokens"] == 16


def test_harness_prompt_is_the_existing_protocol_built_by_its_lua_file():
    messages = [{"role": "user", "content": "é vpn"}]
    prompt = plugins.harness_prompt(messages, binding=BOUND)
    assert prompt == ("Follow this conversation as an inference engine. Return only the assistant response. "
                      "Do not execute tools. Any proposed tool call is text for the ITSM gateway to validate.\n"
                      + json.dumps(messages, separators=(",", ":")))


def test_custom_harness_framing_comes_from_a_reviewed_file(tmp_path, monkeypatch):
    select_source(tmp_path, monkeypatch, "model.harness", "return function(event) event.payload='CUSTOM PROTOCOL\n' .. event.payload.messages_json; return event end".replace("PROTOCOL\n", "PROTOCOL\\n"))
    assert plugins.harness_prompt([]) == "CUSTOM PROTOCOL\n[]"
    refuses("lua_adapter_output_too_large", lambda: plugins.harness_prompt([], max_prompt_bytes=2))


def test_harness_cannot_erase_the_original_conversation(tmp_path, monkeypatch):
    select_source(tmp_path, monkeypatch, "model.harness", "return function(event) event.payload='pretend success'; return event end")
    refuses("lua_adapter_output_invalid", lambda: plugins.harness_prompt([{ "role": "user", "content": "real request"}]))


def test_anthropic_lua_splits_system_merges_turns_and_keeps_explicit_options():
    messages = [{"role": "system", "content": "System A"}, {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"}, {"role": "system", "content": "System B"},
                {"role": "user", "content": "reference"}, {"role": "user", "content": "question"}]
    result = plugins.anthropic_request({"model": "selected"}, messages, 128,
                                       effort="low", thinking="adaptive", max_tokens_limit=100, binding=BOUND)
    assert result == {"model": "selected", "max_tokens": 100, "stream": False,
                      "system": "System A\n\nSystem B", "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"},
                      "messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"},
                                   {"role": "user", "content": "reference\n\nquestion"}]}


@pytest.mark.parametrize("count,cacheable", [(4095, False), (4096, True)])
def test_anthropic_cache_threshold_counts_unicode_characters_not_utf8_bytes(count, cacheable):
    system = "网" * count
    result = plugins.anthropic_request({"model": "selected"}, [{"role": "system", "content": system},
                                                               {"role": "user", "content": "question"}], 32)
    assert isinstance(result["system"], list) is cacheable
    if cacheable:
        assert result["system"] == [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        assert result["system"] == system


def test_anthropic_lua_skips_unicode_whitespace_and_supplies_only_required_initial_turn():
    result = plugins.anthropic_request({"model": "selected"}, [{"role": "system", "content": "system"},
                                                               {"role": "user", "content": "\u2003\u00a0"}], 32)
    assert result["messages"] == [{"role": "user", "content": "Continue with the task described in the system instructions."}]
    refuses("lua_adapter_failed", lambda: plugins.anthropic_request({"model": "selected"}, [{"role": "assistant", "content": "prefill"}], 32))


@pytest.mark.parametrize("change,code", [
    ('body.model="other"', "lua_adapter_binding_changed"),
    ('body.messages[1].content="invented"', "lua_adapter_binding_changed"),
    ('body.max_tokens=1000', "lua_adapter_binding_changed"),
    ('body.tools=adapter.array()', "lua_adapter_output_invalid"),
    ('body.system="invented policy"', "lua_adapter_binding_changed"),
    ('body.thinking={type="adaptive"}', "lua_adapter_output_invalid"),
])
def test_anthropic_guard_rejects_mapped_identity_context_and_permission_changes(tmp_path, monkeypatch, change, code):
    source = f"""return function(event)
      local body={{model=event.payload.model,messages=event.payload.messages,max_tokens=event.payload.max_tokens,stream=false}}
      {change}
      event.payload=body;return event
    end"""
    select_source(tmp_path, monkeypatch, "model.anthropic", source)
    refuses(code, lambda: plugins.anthropic_request({"model": "selected"}, budget().messages, 64))


def test_response_codecs_select_actual_visible_text_and_never_expose_hidden_thinking(monkeypatch):
    from services.lua_adapters import runtime
    original = runtime.subprocess.run
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs["input"])
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime.subprocess, "run", capture)
    assert plugins.openai_response({"choices": [{"message": {"content": "actual answer", "reasoning_content": "HIDDEN_OPENAI"}}]}) == "actual answer"
    assert plugins.anthropic_response({"content": [{"type": "thinking", "thinking": "HIDDEN_ANTHROPIC"},
                                                   {"type": "redacted_thinking", "data": "HIDDEN_REDACTED"},
                                                   {"type": "text", "text": "actual"}, {"type": "text", "text": " answer"}]}) == "actual answer"
    assert all(b"HIDDEN_" not in body for body in seen)


@pytest.mark.parametrize("kind,data", [("openai", {}), ("openai", {"choices": []}),
                                      ("openai", {"choices": [{"message": {"content": " "}}]}),
                                      ("anthropic", {"content": [{"type": "thinking", "thinking": "hidden"}]}),
                                      ("anthropic", {"content": [{"type": "tool_use", "name": "run_shell"}]})])
def test_response_codec_cannot_manufacture_success_from_empty_or_unrequested_provider_data(monkeypatch, kind, data):
    monkeypatch.setattr(plugins, "run_plugin", lambda *a, **k: pytest.fail("Malformed provider data reached Lua"))
    refuses("lua_adapter_output_invalid", lambda: getattr(plugins, kind + "_response")(data))


def test_explicit_and_profile_artifact_overrides_require_matching_bytes_and_never_fall_back(tmp_path, monkeypatch):
    path = tmp_path / "custom.lua"
    path.write_bytes(b"return function(event) event.payload={label='CUSTOM'}; return event end")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = plugins.run_plugin("source.default", {}, binding=BOUND, source_file=path, source_sha256=digest)
    assert result.payload == {"label": "CUSTOM"} and result.receipt["packaged_plugin"] is False
    path.write_bytes(b"return function(event) return event end")
    refuses("lua_adapter_configuration_invalid", lambda: plugins.run_plugin("source.default", {}, binding=BOUND, source_file=path, source_sha256=digest))
    monkeypatch.setattr(plugins, "_profile_artifact", lambda _: {"file": str(tmp_path / "missing.lua"), "sha256": "1" * 64})
    refuses("lua_adapter_configuration_invalid", lambda: plugins.run_plugin("source.default", {}, binding=BOUND))


@pytest.mark.parametrize("artifact", [{"file": "relative.lua", "sha256": "1" * 64},
                                     {"file": "/missing.lua", "sha256": "wrong"},
                                     {"file": "/missing.lua", "sha256": "1" * 64, "unknown": True}])
def test_malformed_artifact_metadata_is_refused(monkeypatch, artifact):
    monkeypatch.setattr(plugins, "_profile_artifact", lambda _: artifact)
    refuses("lua_adapter_configuration_invalid", lambda: plugins.resolve_plugin("model.local"))


def test_missing_packaged_file_or_unknown_plugin_is_not_a_python_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins, "PLUGIN_DIRECTORY", tmp_path)
    refuses("lua_adapter_configuration_invalid", lambda: plugins.resolve_plugin("model.local"))
    refuses("lua_adapter_configuration_invalid", lambda: plugins.resolve_plugin("../other"))


def test_symlink_and_oversized_artifacts_are_refused(tmp_path):
    path = tmp_path / "actual.lua"
    path.write_bytes(b"return function(event) return event end")
    link = tmp_path / "link.lua"
    link.symlink_to(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    refuses("lua_adapter_configuration_invalid", lambda: plugins.resolve_plugin("model.local", source_file=link, source_sha256=digest))
    path.write_bytes(b" " * 20_001)
    refuses("lua_adapter_configuration_invalid", lambda: plugins.resolve_plugin("model.local", source_file=path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_async_codecs_use_actual_selected_lua_files():
    async def exercise():
        request = await plugins.openai_request_async({"model": "selected"}, budget())
        prompt = await plugins.harness_prompt_async([])
        anthropic = await plugins.anthropic_request_async({"model": "selected"}, [{"role": "user", "content": "hi"}], 32)
        response = await plugins.openai_response_async({"choices": [{"message": {"content": "real"}}]})
        return request, prompt, anthropic, response
    request, prompt, anthropic, response = asyncio.run(exercise())
    assert request["max_tokens"] == 64 and prompt.endswith("[]")
    assert anthropic["messages"] == [{"role": "user", "content": "hi"}] and response == "real"


@pytest.mark.parametrize("change", [{"effort": []}, {"thinking": {}}, {"max_tokens_limit": True}])
def test_malformed_anthropic_options_are_typed_refusals(change):
    refuses("lua_adapter_invalid_input", lambda: plugins.anthropic_request({"model": "selected"}, budget().messages, 64, **change))


def test_malformed_roles_and_response_block_types_do_not_escape_as_untyped_errors():
    refuses("lua_adapter_invalid_input", lambda: plugins.anthropic_request({"model": "selected"}, [{"role": [], "content": "hi"}], 64))
    refuses("lua_adapter_output_invalid", lambda: plugins.anthropic_response({"content": [{"type": [], "text": "bad"}]}))


def test_real_machine_profile_selects_a_relative_pinned_plugin_file(tmp_path, monkeypatch):
    path = tmp_path / "chosen.lua"
    path.write_bytes(b"return function(event) event.payload='FILE PROFILE\n' .. event.payload.messages_json; return event end".replace(b"PROFILE\n", b"PROFILE\\n"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    profile = tmp_path / "machine.lua"
    profile.write_text('return {plugins={["model.harness"]={file="chosen.lua",sha256="' + digest + '"}}}')
    monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", str(profile))
    monkeypatch.setenv("ITSM_LUA_ADAPTER_SHA256", hashlib.sha256(profile.read_bytes()).hexdigest())
    monkeypatch.delenv("ITSM_LINK_SECRETS_FILE", raising=False)
    monkeypatch.setattr(plugins, "_profile_artifact", REAL_PROFILE_ARTIFACT)
    assert plugins.harness_prompt([]) == "FILE PROFILE\n[]"
    # An edited plugin is not the reviewed source; the packaged file is not a fallback.
    path.write_text("return function(event) return event end")
    refuses("lua_adapter_configuration_invalid", lambda: plugins.harness_prompt([]))


def test_documented_executable_examples_work_as_selected_files(monkeypatch):
    root = Path(plugins.__file__).resolve().parents[2]
    harness = root / "docs/examples/lua/json_harness.lua"
    openai = root / "docs/examples/lua/bounded_openai.lua"
    mapping = {"model.harness": harness, "model.openai": openai}
    monkeypatch.setattr(plugins, "_profile_artifact", lambda identity: {
        "file": str(mapping[identity]), "sha256": hashlib.sha256(mapping[identity].read_bytes()).hexdigest()})
    assert plugins.harness_prompt([]).endswith("[]")
    assert plugins.openai_request({"model": "selected"}, budget(cap=256))["max_tokens"] == 128
    assert plugins.openai_request({"model": "selected"}, budget(cap=64))["max_tokens"] == 64


def test_a_selected_named_plugin_requires_actual_mapping_logic(tmp_path, monkeypatch):
    select_source(tmp_path, monkeypatch, "source.default", "return {}")
    refuses("lua_adapter_contract_invalid", lambda: plugins.run_plugin("source.default", {"text": "actual"}, binding=BOUND))
