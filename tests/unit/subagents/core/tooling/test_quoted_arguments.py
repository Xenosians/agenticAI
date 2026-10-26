import asyncio

import pytest

from subagents.core.definitions.types import AgentDefinition
from subagents.core.tooling.gateway import ToolGateway, wording_appears_in_request


AGENT = AgentDefinition(
    name="jira-specialist",
    description="Jira worker",
    model="test-model",
    tools=["ticket_search"],
)


class RecordingProvider:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {"ok": True, "status": "success", "tickets": [], "count": 0}


def refuse_approval(*args, **kwargs):
    pytest.fail("A read-only search must not create an approval.")


def search(request, arguments, **gateway_options):
    provider = RecordingProvider()
    gateway = ToolGateway(approval_creator=refuse_approval, mcp=provider, **gateway_options)
    result = asyncio.run(gateway.execute(AGENT, request, "ticket_search", arguments))
    return result, provider.calls


@pytest.mark.parametrize("arguments", [
    {"project_key": "NET", "text": "port 8443"},
    # Only the case differs: the same words, and Jira text search ignores case.
    {"project_key": "NET", "text": "Port 8443"},
    # Quoting is checked, never forced: a search without words still runs.
    {"project_key": "NET"},
])
def test_search_words_from_the_request_reach_the_provider(arguments):
    result, calls = search("List NET tickets mentioning port 8443", arguments)
    assert result["decision_code"] == "success"
    assert calls == [("ticket_search", arguments)]


def test_a_filter_word_joined_by_a_hyphen_is_still_the_users_word():
    arguments = {"priority": "High", "text": "printer jams"}
    result, calls = search("Find high-priority tickets about printer jams", arguments)
    assert result["decision_code"] == "success"
    assert calls == [("ticket_search", arguments)]


@pytest.mark.parametrize("wording,request_text,expected", [
    ("port 8443", "List NET tickets mentioning port 8443?", True),
    ("badge reader", "Any issues with the badge  reader in DESK", True),
    ("High", "Find high-priority tickets", True),
    ("port 8433", "List NET tickets mentioning port 8443", False),
    # Whole words only: part of a longer word is not what the user asked for.
    ("print", "Find printer issues", False),
    ("", "Find printer issues", False),
])
def test_wording_matches_whole_words_ignoring_case_and_spacing(wording, request_text, expected):
    assert wording_appears_in_request(wording, request_text) is expected


@pytest.mark.parametrize("request_text,arguments", [
    # A served candidate searched for a different port than the one requested.
    ("List NET tickets mentioning port 8443", {"project_key": "NET", "text": "port 8433"}),
    ("Which In Progress tickets are about laptop overheating?", {"status": "Done", "text": "laptop overheating"}),
    ("Any Highest priority tickets that mention payroll mismatch?", {"priority": "High", "text": "payroll mismatch"}),
])
def test_a_changed_or_invented_search_filter_is_refused_before_the_provider(request_text, arguments):
    result, calls = search(request_text, arguments)
    assert (result["status"], result["decision_code"]) == ("denied", "grounding_failed")
    assert calls == []


def test_malformed_quoted_metadata_fails_closed():
    tool = {
        "risk": "read",
        "requires_approval": False,
        "quoted_arguments": "text",
        "parameters": {"text": {"type": "str"}},
    }
    result, calls = search("Find tickets about VPN", {"text": "VPN"}, tool_lookup=lambda name: tool)
    assert result["decision_code"] == "grounding_policy_invalid"
    assert calls == []
