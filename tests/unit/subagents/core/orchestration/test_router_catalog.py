from copy import deepcopy
from pathlib import Path
import pytest
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.orchestration.intent_contract import build_router_semantic_agent_spec
from subagents.core.orchestration.router_catalog import METADATA_DEFAULTS, compact_router_catalog, render_router_catalog
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


def _production_compact_specs():
    from mcp_server import register_capability_signatures
    from subagents.core.definitions.registry import AgentRegistry
    from subagents.core.orchestration.candidate_selector import (
        Candidate, CandidateSet, build_candidate_agent_specs)

    register_capability_signatures()
    registry = AgentRegistry()
    registry.register_many(load_agent_directory(Path(__file__).resolve().parents[5] / "subagents/agents"))
    everything = CandidateSet(tuple(Candidate(agent.name, tuple(agent.tools))
                                    for agent in registry.list_agents() if agent.tools), confidence=1.0)
    return build_candidate_agent_specs(everything, registry)


def test_the_compact_catalog_states_shared_defaults_once_and_loses_no_constraint():
    # The full catalog repeats the same empty lists and false flags for most of
    # its capabilities. Stated once, they leave room for the whole catalog on a
    # selected connection; every capability's metadata is still recoverable.
    specs = _production_compact_specs()
    full = compact_router_catalog(specs)
    factored = compact_router_catalog(specs, factor_defaults=True)
    defaults = factored["intent_metadata_defaults"]
    assert defaults == METADATA_DEFAULTS
    assert factored["specialists"] == full["specialists"]
    assert factored["capabilities"].keys() == full["capabilities"].keys()
    for name, capability in full["capabilities"].items():
        stored = factored["capabilities"][name]
        assert {key: value for key, value in stored.items() if key != "intent_metadata"} == \
            {key: value for key, value in capability.items() if key != "intent_metadata"}
        shared = factored["argument_descriptions"]
        metadata = {**defaults, **stored["intent_metadata"]}
        own = metadata.get("grounded_argument_descriptions", {})
        if metadata["grounded_arguments"]:
            metadata["grounded_argument_descriptions"] = {
                arg: own.get(arg, shared.get(arg)) for arg in metadata["grounded_arguments"]}
        assert metadata == capability["intent_metadata"], name
        for key, value in stored["intent_metadata"].items():
            assert key not in defaults or value != defaults[key], (name, key)


def test_a_default_is_never_applied_to_a_field_the_capability_lacks():
    full = {"name": "x", "description": "d", "intent_metadata": dict(METADATA_DEFAULTS)}
    factored = compact_router_catalog([{"name": "a", "capabilities": [full]}], factor_defaults=True)
    assert factored["capabilities"]["x"]["intent_metadata"] == {}
    # Absent means "the default"; a capability that lacks a defaulted field
    # would silently claim it, so it is refused instead.
    partial = {"name": "x", "description": "d", "intent_metadata": {"effect": "read"}}
    with pytest.raises(ValueError, match="default"):
        compact_router_catalog([{"name": "a", "capabilities": [partial]}], factor_defaults=True)


def test_the_compact_prompt_explains_the_defaults_and_the_standard_prompt_is_unchanged():
    specs = _production_compact_specs()
    compact = render_router_catalog(specs, factor_defaults=True)
    assert "intent_metadata_defaults" in compact
    assert "absent" in compact.split("\n", 1)[0]
    assert render_router_catalog(specs) == render_router_catalog(specs, factor_defaults=False)
    assert "intent_metadata_defaults" not in render_router_catalog(specs)


def test_a_shared_argument_description_is_stated_once_and_every_capability_keeps_its_own():
    # 20 Nov 2026: the broad catalog had 87 tokens left before its reserve, while
    # "Exact VMID of the guest, as listed by proxmox_guests." alone was repeated
    # nine times. A description several capabilities share is stated once under
    # argument_descriptions; a capability lists only descriptions that differ,
    # and every capability's full description map is still recoverable.
    specs = _production_compact_specs()
    full = compact_router_catalog(specs)
    factored = compact_router_catalog(specs, factor_defaults=True)
    shared = factored["argument_descriptions"]
    assert shared and all(isinstance(text, str) and text for text in shared.values())
    for name, capability in full["capabilities"].items():
        own = factored["capabilities"][name]["intent_metadata"].get("grounded_argument_descriptions", {})
        grounded = capability["intent_metadata"]["grounded_arguments"]
        recovered = {arg: own.get(arg, shared.get(arg)) for arg in grounded}
        assert recovered == capability["intent_metadata"].get("grounded_argument_descriptions", {}), name
        for arg, text in own.items():
            assert shared.get(arg) != text, (name, arg)
    # Only descriptions used by more than one capability are shared.
    uses = {}
    for capability in full["capabilities"].values():
        for arg, text in capability["intent_metadata"].get("grounded_argument_descriptions", {}).items():
            uses[(arg, text)] = uses.get((arg, text), 0) + 1
    assert all(uses[(arg, text)] > 1 for arg, text in shared.items())
    rendered = render_router_catalog(specs, factor_defaults=True)
    assert "argument_descriptions" in rendered.split("\n", 1)[0]


def test_an_argument_without_any_description_is_never_given_a_shared_one():
    # A grounded argument whose capability did not describe it must not
    # silently borrow another capability's wording.
    described = {"name": "a", "description": "d", "intent_metadata": {
        **METADATA_DEFAULTS, "grounded_arguments": ["vmid"], "required_arguments": ["vmid"],
        "grounded_argument_descriptions": {"vmid": "Exact VMID."}}}
    twin = deepcopy(described) | {"name": "b"}
    bare = {"name": "c", "description": "d", "intent_metadata": {
        **METADATA_DEFAULTS, "grounded_arguments": ["vmid"], "required_arguments": ["vmid"]}}
    with pytest.raises(ValueError, match="description"):
        compact_router_catalog([{"name": "x", "capabilities": [described, twin, bare]}], factor_defaults=True)


def test_the_whole_compact_catalog_fits_the_broad_routing_budget_with_room_to_grow():
    from services.extensions.budget import estimate_text_tokens
    from subagents.core.orchestration.router_prompt import build_compact_router_system_prompt
    from subagents.llm.runtime.stages import InferenceStage, resolve_stage_policy

    policy = resolve_stage_policy(InferenceStage.SEMANTIC_ROUTING_BROAD)
    assert policy.request_token_ceiling is not None and policy.output_token_ceiling is not None
    system = build_compact_router_system_prompt(_production_compact_specs())
    # The output reserve plus room for about six more capabilities, so the
    # next additions fail here, not as an unexplained routing fallback.
    assert estimate_text_tokens(system) + policy.output_token_ceiling + 1000 <= policy.request_token_ceiling


def test_training_and_runtime_share_identical_router_prompt():
    root = Path(__file__).resolve().parents[5]
    environment = build_hub_training_environment(agent_directory=root / "subagents/agents")
    router = LLMRouter(registry=environment["registry"], inference=None, model_key="hub-main")
    assert router._build_system_prompt() == environment["system_prompt"]
