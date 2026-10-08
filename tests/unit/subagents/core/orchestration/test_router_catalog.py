from copy import deepcopy
from pathlib import Path
import pytest
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.orchestration.intent_contract import build_router_semantic_agent_spec
from subagents.core.orchestration.router_catalog import compact_router_catalog
from learning.training.hub_training_contracts import build_hub_training_environment
from subagents.core.orchestration.router import LLMRouter


def test_all_capabilities_and_constraints_survive_compaction():
    root = Path(__file__).resolve().parents[5]
    specs = [build_router_semantic_agent_spec(a) for a in load_agent_directory(root / "subagents/agents")]
    original = deepcopy(specs)
    compact = compact_router_catalog(specs)
    assert specs == original
    for before, after in zip(specs, compact["specialists"], strict=True):
        assert before["name"] == after["name"]
        assert after["capabilities"] == [c["name"] for c in before["capabilities"]]
        for capability in before["capabilities"]:
            stored = compact["capabilities"][capability["name"]]
            for key, value in capability.items():
                assert value == (stored[key] if key in stored else stored["intent_metadata"][key])


def test_conflicting_shared_definitions_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        compact_router_catalog([{"name":"a", "capabilities":[{"name":"read", "description":"first"}]},
                                {"name":"b", "capabilities":[{"name":"read", "description":"second"}]}])


def test_training_and_runtime_share_identical_router_prompt():
    root = Path(__file__).resolve().parents[5]
    environment = build_hub_training_environment(agent_directory=root / "subagents/agents")
    router = LLMRouter(registry=environment["registry"], inference=None, model_key="hub-main")
    assert router._build_system_prompt() == environment["system_prompt"]
