"""The networking specialist owns the governed tailnet grant changes (D2) and is taught them.

Each grant tool has at least three authored examples and one held-out case; requests name the
tags, groups or autogroups and ports exactly, the arguments carry exactly those values, and no
example names a user or an address.
"""
import json
from pathlib import Path

from learning.training.specialist_sft import load_training_config
from subagents.core.definitions.loader import load_agent_definition

CONFIG = Path("learning/config/specialists/networking.json")
GRANT_TOOLS = {"network_tailnet_grant_add", "network_tailnet_grant_remove"}


def test_the_networking_specialist_owns_and_is_taught_both_grant_tools():
    config = load_training_config(CONFIG)
    agent = load_agent_definition(Path(config.agent_definition))
    assert GRANT_TOOLS <= set(agent.tools)
    example_tools = [example.tool_name for example in config.examples]
    assert all(example_tools.count(tool) >= 3 for tool in GRANT_TOOLS)
    assert GRANT_TOOLS <= {case.expected_tool for case in config.eval_cases}
    requests = [example.user_request.strip() for example in config.examples]
    assert len(requests) == len(set(requests))
    assert not set(requests) & {case.user.strip() for case in config.eval_cases}


def test_grant_examples_name_every_selector_and_port_exactly_and_never_a_user_or_address():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = [item for item in payload["examples"] if item["tool_name"] in GRANT_TOOLS]
    rows += [{"user_request": case["user"], "arguments": case["expected_arguments"]}
             for case in payload["eval_cases"] if case["expected_tool"] in GRANT_TOOLS]
    assert len(rows) >= 8
    for item in rows:
        arguments = item["arguments"]
        assert set(arguments) == {"sources", "destinations", "ports"}, item
        for value in arguments["sources"] + arguments["destinations"] + arguments["ports"]:
            assert value in item["user_request"] or (value == "*" and "every" in item["user_request"]), (value, item)
        assert "@" not in item["user_request"]
        assert not any(part.count(".") == 3 for part in item["user_request"].split())


def test_the_agent_definition_says_how_grant_changes_are_governed():
    text = " ".join(Path("subagents/agents/networking-specialist.md").read_text(encoding="utf-8").split())
    assert "network_tailnet_grant_add" in text and "network_tailnet_grant_remove" in text
    assert "approval" in text and "the owner types it when approving a removal" in text
    # Live, a removal proposal that added the grant text itself was refused as an unknown argument:
    # the model proposes only the three named lists; Janus writes the grant.
    assert "only sources, destinations and ports" in text
    assert "never add a grant, etag or digest yourself" in text
