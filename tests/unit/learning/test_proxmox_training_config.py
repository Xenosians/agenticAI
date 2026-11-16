"""The Proxmox specialist's authored examples and held-out cases cover every tool it owns."""
import json
from pathlib import Path

from learning.training.specialist_sft import load_training_config, semantic_context_for
from scripts.tool_acceptance_matrix import evaluate_proposal, load_inventory, load_proposal_cases
from subagents.core.definitions.loader import load_agent_definition

CONFIG = Path("learning/config/specialists/proxmox.json")


def test_every_proxmox_tool_has_training_examples_and_a_held_out_case():
    config = load_training_config(CONFIG)
    agent = load_agent_definition(Path(config.agent_definition))
    example_tools = [example.tool_name for example in config.examples]
    held_out_tools = {case.expected_tool for case in config.eval_cases}
    assert set(example_tools) <= set(agent.tools)
    assert held_out_tools == set(agent.tools)
    assert all(example_tools.count(tool) >= 3 for tool in agent.tools)
    requests = [example.user_request.strip() for example in config.examples]
    assert len(requests) == len(set(requests))
    assert not set(requests) & {case.user.strip() for case in config.eval_cases}


def test_force_stop_is_taught_only_for_explicit_force_wording():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in payload["examples"]:
        request = item["user_request"].lower()
        forceful = any(word in request for word in ("force", "kill", "hard stop"))
        assert (item["tool_name"] == "proxmox_guest_force_stop") == forceful, item["user_request"]


def test_guest_examples_name_the_exact_integer_vmid_and_node_in_the_request():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    guest_examples = [item for item in payload["examples"]
                      if item["tool_name"] in {"proxmox_guest_status", "proxmox_guest_config"}]
    assert guest_examples
    for item in guest_examples:
        vmid = item["arguments"]["vmid"]
        assert type(vmid) is int
        assert str(vmid) in item["user_request"]
        assert item["arguments"]["node"] in item["user_request"]


def test_training_context_grounds_an_integer_vmid_as_its_decimal_text():
    context = semantic_context_for("proxmox_guest_status", {"node": "pve2", "vmid": 104})
    assert context["allowed_arguments"] == {"node": ["pve2"], "vmid": ["104"]}
    assert semantic_context_for("proxmox_guest_status", {"vmid": True})["allowed_arguments"] == {}


def test_training_context_binds_only_grounded_arguments_as_the_runtime_router_does():
    # The runtime intent parser rejects a bound argument the capability does not ground ("untrusted or
    # irrelevant grounded argument 'timeframe'"). Training must show the context the router can produce:
    # an optional, service-bounded window stays in the call, not in allowed_arguments.
    context = semantic_context_for("proxmox_node_load", {"node": "pve1", "timeframe": "week"})
    assert context["allowed_arguments"] == {"node": ["pve1"]}


def test_every_reviewed_example_gives_a_context_the_runtime_intent_parser_accepts():
    from subagents.core.orchestration.intent_contract import parse_semantic_intent

    contracts, agents = load_inventory()
    owner = {tool: name for name, agent in agents.items() for tool in agent.tools}
    rejected = []
    for path in sorted(Path("learning/config/specialists").glob("*.json")):
        for example in json.loads(path.read_text(encoding="utf-8")).get("examples", []):
            tool = example["tool_name"]
            if tool not in owner:
                continue
            context = semantic_context_for(tool, example["arguments"])
            try:
                parse_semantic_intent({"summary": example["user_request"], "effect": contracts[tool].effect,
                                       "allowed_tools": [tool], "allowed_arguments": context["allowed_arguments"],
                                       "forbidden_tools": [], "forbidden_arguments": {}, "max_tool_calls": 1,
                                       "clarification_required": False},
                                      agent=agents[owner[tool]], user_request=example["user_request"])
            except ValueError as error:
                rejected.append((path.stem, tool, str(error)[:80]))
    assert rejected == []


def test_held_out_guest_cases_pass_the_semantic_guard_with_their_gold_call():
    contracts, agents = load_inventory()
    cases = [case for case in load_proposal_cases() if case.case_id.startswith("proxmox:")]
    assert {case.expected_tool for case in cases} >= {"proxmox_guest_status", "proxmox_guest_config"}
    for case in cases:
        raw = json.dumps([{"name": case.expected_tool, "arguments": case.expected_arguments}])
        result = evaluate_proposal(case, raw, contracts=contracts, agents=agents)
        assert result["passed"], (case.case_id, result)
        assert not case.training_overlap, case.case_id
