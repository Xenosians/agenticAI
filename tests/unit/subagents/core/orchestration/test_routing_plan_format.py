"""Routing plans with a slightly wrong format: repaired only where nothing can change, and shown in the trace.

The recorded plans come from the local Hub (tmp/hermes/local-routing-eval-failures.json, compact prompt):
the right capability in each, rejected only because of the closing brackets or an omitted constant field.
"""
import asyncio
import json
from pathlib import Path

import pytest

from learning.evidence.hub_routing import consume_hub_routing_trace
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.candidate_selector import Candidate, CandidateSet
from subagents.core.orchestration.plan_format import parse_routing_plan
from subagents.core.orchestration.router import LLMRouter, RoutingContractError

# Recorded: every container closed, but "}]}}" where "}}]}" was needed; intent constants omitted.
WIFI_MISNESTED = (
    '{"delegations":[{"agent":"networking-specialist","instructions":"list current wireless clients '
    'connected to the specified interface","intent":{"summary":"retrieve wireless station list",'
    '"effect":"read","allowed_tools":["network_wireless_clients"],"forbidden_arguments":{},'
    '"allowed_arguments":{}}]}}'
)
# Recorded: one closing brace short. It may be a cut-off plan, so it is never completed.
WIFI_ONE_SHORT = (
    '{"delegations":[{"agent":"networking-specialist","instructions":"list current wireless clients '
    'connected to the router\'s wireless interface","intent":{"summary":"retrieve wireless clients",'
    '"effect":"read","allowed_tools":["network_wireless_clients"],"forbidden_arguments":{},'
    '"allowed_arguments":{}}]}'
)
# Recorded: valid JSON, both intents without max_tool_calls and clarification_required.
TWO_READS_WITHOUT_CONSTANTS = (
    '{"delegations":[{"agent":"networking-specialist","instructions":"read OpenWrt router uptime",'
    '"intent":{"summary":"list OpenWrt router uptime","effect":"read","allowed_tools":'
    '["network_router_health"],"forbidden_arguments":{},"allowed_arguments":{}}},{"agent":'
    '"jira-specialist","instructions":"list Jira projects","intent":{"summary":"list Jira projects",'
    '"effect":"read","allowed_tools":["jira_project_list"],"forbidden_arguments":{},'
    '"allowed_arguments":{}}}]}'
)


def test_a_valid_complete_plan_is_parsed_without_any_repair():
    plan = '{"delegations":[]}'
    assert parse_routing_plan(plan) == ({"delegations": []}, ())


def test_closers_in_the_wrong_order_are_put_in_the_only_valid_order():
    plan, repairs = parse_routing_plan(WIFI_MISNESTED)
    assert repairs == ("closing_order", "max_tool_calls", "clarification_required")
    [delegation] = plan["delegations"]
    assert delegation["agent"] == "networking-specialist"
    assert delegation["intent"]["allowed_tools"] == ["network_wireless_clients"]
    assert delegation["intent"]["max_tool_calls"] == 1
    assert delegation["intent"]["clarification_required"] is False


def test_extra_closers_after_a_complete_plan_are_dropped():
    assert parse_routing_plan('{"delegations":[]}}') == ({"delegations": []}, ("closing_order",))


@pytest.mark.parametrize("response", [
    WIFI_ONE_SHORT,  # fewer closers than open containers: possibly cut off
    '{"delegations":[{"agent":"proxmox-specialist","instructions":"Check whether',  # cut inside a string
    '{"delegations":[{"agent":"a","intent":{"x":1}]},{"agent":"b"}]}',  # wrong closer mid-plan
    '{"delegations":[{"agent":"a" "x":1}]}',  # missing comma: not a closing slip
    '}]}',
])
def test_any_other_malformed_plan_is_left_to_the_trusted_repair(response):
    with pytest.raises(json.JSONDecodeError):
        parse_routing_plan(response)


def test_omitted_intent_constants_are_added_and_present_values_never_replaced():
    plan, repairs = parse_routing_plan(TWO_READS_WITHOUT_CONSTANTS)
    assert repairs == ("max_tool_calls", "clarification_required")
    assert [d["intent"]["max_tool_calls"] for d in plan["delegations"]] == [1, 1]

    explicit = json.dumps({"delegations": [{"agent": "a", "instructions": "b", "intent": {
        "summary": "c", "effect": "read", "allowed_tools": ["x"], "max_tool_calls": 2,
        "clarification_required": True}}]})
    plan, repairs = parse_routing_plan(explicit)
    assert repairs == ()
    assert plan["delegations"][0]["intent"]["max_tool_calls"] == 2
    assert plan["delegations"][0]["intent"]["clarification_required"] is True


def test_a_lone_argument_value_becomes_its_one_element_list_and_nothing_else_is_coerced():
    # Recorded repair output (17 Nov, node-maintenance): "node":"hummingbird" without the list.
    plan = json.dumps({"delegations": [{"agent": "a", "instructions": "b", "intent": {
        "summary": "c", "effect": "read", "allowed_tools": ["x"], "max_tool_calls": 1,
        "clarification_required": False, "allowed_arguments": {"node": "hummingbird", "kept": ["v"]},
        "forbidden_arguments": {"user_id": "mallory"}}}]})
    parsed, repairs = parse_routing_plan(plan)
    intent = parsed["delegations"][0]["intent"]
    assert repairs == ("argument_list",)
    assert intent["allowed_arguments"] == {"node": ["hummingbird"], "kept": ["v"]}
    assert intent["forbidden_arguments"] == {"user_id": ["mallory"]}

    for value in (7, None, "", "  ", {"a": 1}, True):
        odd = json.dumps({"delegations": [{"agent": "a", "instructions": "b", "intent": {
            "summary": "c", "effect": "read", "allowed_tools": ["x"], "max_tool_calls": 1,
            "clarification_required": False, "allowed_arguments": {"node": value}}}]})
        parsed, repairs = parse_routing_plan(odd)
        assert repairs == ()
        assert parsed["delegations"][0]["intent"]["allowed_arguments"] == {"node": value}


def test_an_empty_object_binding_is_absence_like_null_and_an_empty_list(registry):
    # Recorded (17 Nov, account-then-ticket): the follow-up ticket bound "summary":{}.
    from subagents.core.orchestration.intent_contract import parse_semantic_intent
    intent = {"summary": "Create ticket in project SUP", "effect": "mutation", "allowed_tools": ["ticket_create"],
              "forbidden_tools": [], "forbidden_arguments": {}, "max_tool_calls": 1,
              "clarification_required": False}
    bound = {}
    for empty in (None, [], {}):
        parsed = parse_semantic_intent({**intent, "allowed_arguments": {"project_key": ["SUP"], "summary": empty}},
                                       agent=registry.get("jira-specialist"),
                                       user_request="Create a Jira ticket in project SUP for that created account.")
        bound[repr(empty)] = parsed.allowed_arguments
    assert bound["None"] == bound["[]"] == bound["{}"] == {"project_key": ["SUP"]}

    with pytest.raises(ValueError, match="must be a list"):
        parse_semantic_intent({**intent, "allowed_arguments": {"project_key": {"key": "SUP"}}},
                              agent=registry.get("jira-specialist"), user_request="project SUP")


def test_clarification_is_not_assumed_away_when_no_capability_is_named():
    asking = json.dumps({"delegations": [{"agent": "a", "instructions": "b", "intent": {
        "summary": "c", "effect": "unknown", "allowed_tools": [], "max_tool_calls": 1}}]})
    plan, repairs = parse_routing_plan(asking)
    assert repairs == ()
    assert "clarification_required" not in plan["delegations"][0]["intent"]


class RecordingInference:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FixedSelector:
    def __init__(self, selection):
        self.selection = selection

    def select(self, request, registry):
        return self.selection


@pytest.fixture
def registry(monkeypatch):
    from mcp_server import register_capability_signatures
    from tools import runtime_schema
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    register_capability_signatures()
    result = AgentRegistry()
    result.register_many(load_agent_directory(Path(__file__).resolve().parents[5] / "subagents/agents"))
    return result


def local_router(registry, selection, *responses):
    inference = RecordingInference(*responses)
    router = LLMRouter(registry, inference, "hub-main", strict_contract=True, max_new_tokens=512,
                       router_prompt_profile="compact", candidate_selector=FixedSelector(selection))
    return router, inference


def test_recorded_misnested_plan_routes_on_the_first_generation_and_the_trace_says_why(registry):
    selection = CandidateSet((Candidate("networking-specialist", ("network_wireless_clients",)),), confidence=1)
    router, inference = local_router(registry, selection, WIFI_MISNESTED)

    async def scenario():
        routes = await router.route_with_repair("who is connected to our wifi right now?")
        return routes, consume_hub_routing_trace()

    routes, trace = asyncio.run(scenario())
    assert len(inference.calls) == 1
    assert [route.semantic_intent.allowed_tools for route in routes] == [["network_wireless_clients"]]
    [attempt] = trace
    assert attempt.validation_status == "accepted"
    assert attempt.format_repairs == ["closing_order", "max_tool_calls", "clarification_required"]
    assert attempt.raw_response == WIFI_MISNESTED  # the model's own output stays on record


def test_recorded_two_read_plan_without_constants_routes_both_reads(registry):
    selection = CandidateSet((Candidate("networking-specialist", ("network_router_health",)),
                              Candidate("jira-specialist", ("jira_project_list",))), confidence=1)
    router, inference = local_router(registry, selection, TWO_READS_WITHOUT_CONSTANTS)
    routes = asyncio.run(router.route_with_repair(
        "Show the OpenWrt router uptime and list the Jira projects I can see."))
    assert len(inference.calls) == 1
    assert sorted(tool for route in routes for tool in route.semantic_intent.allowed_tools) == [
        "jira_project_list", "network_router_health"]


def test_a_possibly_cut_off_plan_still_takes_the_one_trusted_repair(registry):
    selection = CandidateSet((Candidate("networking-specialist", ("network_wireless_clients",)),), confidence=1)
    router, inference = local_router(registry, selection, WIFI_ONE_SHORT, WIFI_ONE_SHORT)
    with pytest.raises(RoutingContractError, match="invalid JSON"):
        asyncio.run(router.route_with_repair("who is connected to our wifi right now?"))
    assert len(inference.calls) == 2


def test_an_intent_with_several_capabilities_is_rejected_and_the_repair_is_told_which(registry):
    request = "Does Proxmox node hummingbird need a reboot, and is any Proxmox service down?"
    selection = CandidateSet((Candidate("proxmox-specialist", ("proxmox_nodes", "proxmox_node_maintenance")),),
                             confidence=1)
    both = json.dumps({"delegations": [{"agent": "proxmox-specialist", "instructions": "Check node hummingbird",
                                        "intent": {"summary": "maintenance state", "effect": "read",
                                                   "allowed_tools": ["proxmox_nodes", "proxmox_node_maintenance"],
                                                   "forbidden_tools": [], "allowed_arguments": {},
                                                   "forbidden_arguments": {}, "max_tool_calls": 1,
                                                   "clarification_required": False}}]})
    router, inference = local_router(registry, selection, both, both)
    with pytest.raises(RoutingContractError, match="exactly one allowed tool"):
        asyncio.run(router.route_with_repair(request))
    assert len(inference.calls) == 2
    feedback = next(message["content"] for message in inference.calls[1]["messages"]
                    if message["role"] == "system" and "VALIDATION FEEDBACK" in message["content"])
    assert "it listed 2: proxmox_nodes, proxmox_node_maintenance" in feedback
    assert "Keep only the one capability that directly performs the requested operation" in feedback


# Recorded (17 Nov, compact): the right single capability, but the node the user named was not bound.
MAINTENANCE_UNBOUND = (
    '{"delegations":[{"agent":"proxmox-specialist","instructions":"Check if Proxmox node hummingbird needs a '
    'reboot due to a newer kernel and whether any Proxmox services are down (not running)","intent":{"summary":'
    '"Check maintenance status of Proxmox node hummingbird","effect":"read","allowed_tools":'
    '["proxmox_node_maintenance"],"forbidden_arguments":{},"forbidden_tools":[]}}]}'
)
MAINTENANCE_BOUND = json.dumps({"delegations": [{
    "agent": "proxmox-specialist", "instructions": "Check node hummingbird's maintenance state",
    "intent": {"summary": "maintenance state", "effect": "read", "allowed_tools": ["proxmox_node_maintenance"],
               "forbidden_tools": [], "allowed_arguments": {"node": ["hummingbird"]}, "forbidden_arguments": {},
               "max_tool_calls": 1, "clarification_required": False}}]})
MAINTENANCE_REQUEST = "Does Proxmox node hummingbird need a reboot, and is any Proxmox service down?"


def maintenance_router(registry, *responses):
    selection = CandidateSet((Candidate("proxmox-specialist", ("proxmox_node_maintenance", "proxmox_nodes")),),
                             confidence=1)
    return local_router(registry, selection, *responses)


def test_a_plan_that_asks_nothing_but_leaves_a_required_value_unbound_gets_the_one_repair(registry):
    router, inference = maintenance_router(registry, MAINTENANCE_UNBOUND, MAINTENANCE_BOUND)
    routes = asyncio.run(router.route_with_repair(MAINTENANCE_REQUEST))
    assert len(inference.calls) == 2
    [route] = routes
    assert not route.semantic_intent.clarification_required
    assert route.semantic_intent.allowed_arguments == {"node": ["hummingbird"]}
    feedback = next(message["content"] for message in inference.calls[1]["messages"]
                    if message["role"] == "system" and "VALIDATION FEEDBACK" in message["content"])
    assert "required argument node of proxmox_node_maintenance" in feedback
    assert "Check if Proxmox node" not in feedback  # model text never enters trusted feedback


def test_if_the_repair_still_leaves_it_unbound_the_owner_is_asked_instead_of_failing(registry):
    router, inference = maintenance_router(registry, MAINTENANCE_UNBOUND, MAINTENANCE_UNBOUND)
    [route] = asyncio.run(router.route_with_repair(MAINTENANCE_REQUEST))
    assert len(inference.calls) == 2
    assert route.semantic_intent.clarification_required
    assert route.semantic_intent.allowed_tools == ["proxmox_node_maintenance"]


def test_a_plan_that_itself_asks_for_the_missing_value_is_not_repaired(registry):
    asking = json.loads(MAINTENANCE_UNBOUND)
    asking["delegations"][0]["intent"].update(clarification_required=True, max_tool_calls=1)
    router, inference = maintenance_router(registry, json.dumps(asking))
    [route] = asyncio.run(router.route_with_repair("Does my Proxmox node need a reboot?"))
    assert len(inference.calls) == 1
    assert route.semantic_intent.clarification_required


def test_an_empty_repair_never_turns_the_question_into_a_conversational_answer(registry):
    # Broad routing accepts an empty plan as "no capability needed"; here it must not replace the question.
    router, inference = maintenance_router(registry, MAINTENANCE_UNBOUND, '{"delegations":[]}')
    [route] = asyncio.run(router.route_with_repair(MAINTENANCE_REQUEST))
    assert len(inference.calls) == 2
    assert route.semantic_intent.clarification_required


def test_concurrent_jobs_do_not_share_which_questions_the_contract_imposed(registry):
    # One router serves every job. Job A's contract-imposed question must not leak into job B.
    async def scenario():
        router_a, calls_a = maintenance_router(registry, MAINTENANCE_UNBOUND, MAINTENANCE_BOUND)
        asked = json.loads(MAINTENANCE_UNBOUND)
        asked["delegations"][0]["intent"].update(clarification_required=True, max_tool_calls=1)
        router_b = router_a
        router_b.inference = calls_a  # same router and inference: responses interleave in call order
        calls_a.responses = iter([MAINTENANCE_UNBOUND, json.dumps(asked), MAINTENANCE_BOUND])
        first = asyncio.create_task(router_a.route_with_repair(MAINTENANCE_REQUEST))
        second = asyncio.create_task(router_b.route_with_repair("Does my Proxmox node need a reboot?"))
        return await asyncio.gather(first, second), calls_a

    (plan_a, plan_b), inference = asyncio.run(scenario())
    assert not plan_a[0].semantic_intent.clarification_required  # A was repaired
    assert plan_b[0].semantic_intent.clarification_required       # B asked itself; never repaired
    assert len(inference.calls) == 3


def test_a_value_the_model_bound_but_the_contract_refused_is_asked_not_repaired(registry):
    # Not a slip: the model bound a node the request never named; the trust layer drops it and asks.
    ungrounded = json.loads(MAINTENANCE_BOUND)
    ungrounded["delegations"][0]["intent"]["allowed_arguments"] = {"node": ["pve1"]}
    router, inference = maintenance_router(registry, json.dumps(ungrounded), MAINTENANCE_BOUND)
    [route] = asyncio.run(router.route_with_repair(MAINTENANCE_REQUEST))
    assert len(inference.calls) == 1
    assert route.semantic_intent.clarification_required
    assert route.semantic_intent.allowed_arguments == {}
