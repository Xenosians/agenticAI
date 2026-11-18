"""The worker's copies of Janus's prompt and scoring rules stay identical to the originals, and every weight
load in the worker goes through an admission check."""
import ast
from pathlib import Path

import pytest

from learning.evaluation.config_specialist_gate import _arguments_match
from learning.training.specialist_sft import SpecialistEvalCase
from learning.worker import prompts, scoring
from subagents.core.tooling.parser import parse_tool_calls
from subagents.llm.runtime.hf_prompt import HF_CAUSAL_FALLBACK_PROMPT_SCHEMA, render_hf_causal_fallback_prompt

WORKER = Path(__file__).resolve().parents[4] / "learning" / "worker"

MESSAGES = [
    [{"role": "system", "content": "  rules \n"}, {"role": "user", "content": "do it"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b "}, {"role": "tool", "content": "c"}],
    [{"content": "no role"}],
]
RESPONSES = [
    '[{"name": "ping", "arguments": {"host": "a"}}]',
    '[{"arguments": {}, "name": "ping"}',
    '{"name": "ping", "arguments": {}}',
    "[]",
    "not json",
    '[{"name": "", "arguments": {}}]',
    '[{"name": "ping", "arguments": []}]',
    '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]',
]


def test_the_fallback_prompt_is_the_one_janus_serves_with():
    assert prompts.FALLBACK_SCHEMA == HF_CAUSAL_FALLBACK_PROMPT_SCHEMA
    for messages in MESSAGES:
        assert prompts.render_fallback(messages) == render_hf_causal_fallback_prompt(messages)


def test_every_serving_backend_has_a_known_prompt_rendering():
    """A new backend must declare how it renders prompts before a worker job can target it."""
    import re

    from learning.worker_jobs import SERVING_TEMPLATE_OPTIONS

    factory = (WORKER.parents[1] / "subagents/llm/runtime/factory.py").read_text(encoding="utf-8")
    backends = set(re.findall(r'backend_type\s*==\s*"([a-z0-9-]+)"', factory)) | {"hf-causal"}
    assert len(backends) >= 5 and backends <= set(SERVING_TEMPLATE_OPTIONS)


def test_the_qwen3_option_matches_how_the_qwen3_backend_serves():
    from learning.worker_jobs import SERVING_TEMPLATE_OPTIONS

    source = (WORKER.parents[1] / "subagents/llm/backends/qwen3_worker.py").read_text(encoding="utf-8")
    assert "enable_thinking=False" in source
    assert SERVING_TEMPLATE_OPTIONS["qwen3"] == {"enable_thinking": False}


@pytest.mark.parametrize("response", RESPONSES)
def test_tool_calls_parse_exactly_as_janus_parses_them(response):
    try:
        expected = parse_tool_calls(response)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error)):
            scoring.parse_calls(response)
    else:
        assert scoring.parse_calls(response) == expected


@pytest.mark.parametrize("observed", [
    {"host": "a"}, {"host": "b"}, {"host": "a", "port": "22"}, {"host": "a", "port": "23"}, {"host": "a", "x": 1}, {},
    [],
])
def test_arguments_match_exactly_as_the_janus_gate_matches_them(observed):
    case = {"name": "n", "user": "u", "allowed_tools": ["ping"], "expected_tool": "ping",
            "expected_arguments": {"host": "a"}, "allowed_extra_arguments": ["port"],
            "allowed_arguments": {"port": ["22"]}}
    assert scoring.arguments_match(case, observed) is _arguments_match(SpecialistEvalCase(**case), observed)


def test_no_tool_sentinels_match_the_gate():
    from learning.evaluation.config_specialist_gate import NO_TOOL_SENTINELS

    assert scoring.NO_TOOL == NO_TOOL_SENTINELS


def test_the_workers_adapter_digest_is_the_one_janus_registers_by(tmp_path):
    from learning.continual.storage import fingerprint_directory as janus_fingerprint
    from learning.worker.stages import fingerprint_directory

    (tmp_path / "nested").mkdir()
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "nested" / "adapter_model.safetensors").write_bytes(b"\x00\x01weights")
    assert fingerprint_directory(tmp_path) == janus_fingerprint(tmp_path)


def test_the_worker_imports_nothing_from_janus():
    for path in WORKER.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] not in {"learning", "services", "subagents", "config", "tools",
                                                         "api", "agent"}, f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in {"learning", "services", "subagents", "config"}


def test_every_weight_load_in_the_worker_is_admitted_or_its_own_output():
    loads = []
    for path in WORKER.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in {"admitted_load", "own_output"}:
                # The load runs inside the callable handed to the guard.
                guarded.update(id(inner) for argument in node.args for inner in ast.walk(argument))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"from_pretrained", "load_adapter"}:
                loads.append((path.name, node.lineno, id(node) in guarded))
    assert loads, "the worker loads weights somewhere"
    assert [load for load in loads if not load[2]] == []
