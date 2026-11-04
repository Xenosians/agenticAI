"""A reply to Janus's own clarification completes the request it asked about.

Janus binds an authority-bearing value (a name, a project key) only when the
user wrote it. When Janus answers a request with a clarification and the user
replies with the missing detail ("Doe"), the reply alone does not contain the
earlier values ("John", "software developer"), so without this rule the same
clarification repeats forever.

The chain: consecutive user-authored requests, each answered by a saved
clarification, ending with the current reply. Only user text joins it, never an
assistant answer or a tool result; a turn that was not a clarification ends it;
it is bounded. A change still needs approval of its exact values.
"""
import asyncio

from subagents.core.definitions.types import SemanticIntent, SpecialistRequest
from subagents.core.orchestration.clarification import CHAIN_LIMIT, clarification_chain
from subagents.core.orchestration.orchestrator import Orchestrator

ASKED = "Please provide the required family name before I continue."


def clarification(content=ASKED, agent="account-specialist", outcome="semantic_clarification_required"):
    """An assistant turn as Phoenix sends it: the answer plus its saved result record."""
    return {"role": "assistant", "content": content, "result_evidence": {
        "job_id": "job-1", "job_status": "completed", "results_truncated": False,
        "results": [{"agent_name": agent, "status": "clarification_required", "outcome_code": outcome}],
    }}


def test_a_reply_joins_the_request_its_clarification_asked_about():
    context = [{"role": "user", "content": "Create an account for John as a software developer."},
               clarification()]

    assert clarification_chain("Doe", context) == [
        "Create an account for John as a software developer.", "Doe"]


def test_a_chain_spans_several_clarifications_in_a_row():
    context = [{"role": "user", "content": "Create an account for John."}, clarification(),
               {"role": "user", "content": "His family name is Doe."},
               clarification("Please provide the required department before I continue.")]

    assert clarification_chain("IT", context) == ["Create an account for John.", "His family name is Doe.", "IT"]


def test_without_a_clarification_the_current_request_stands_alone():
    answered = {"role": "assistant", "content": "Account jdoe is enabled.", "result_evidence": {
        "job_id": "job-1", "job_status": "completed", "results_truncated": False,
        "results": [{"agent_name": "account-specialist", "status": "success", "outcome_code": "success"}]}}
    context = [{"role": "user", "content": "Is jdoe enabled?"}, answered]

    assert clarification_chain("Disable jdoe", context) == ["Disable jdoe"]
    assert clarification_chain("Doe", []) == ["Doe"]


def test_assistant_text_never_joins_the_chain():
    # Text that merely looks like a clarification, without its saved record,
    # and the assistant's own words never become user authority.
    unrecorded = {"role": "assistant", "content": ASKED}
    context = [{"role": "user", "content": "Create an account for John."}, unrecorded]
    assert clarification_chain("Doe", context) == ["Doe"]

    context = [{"role": "user", "content": "Create an account."},
               clarification("Please provide the required given name. For example: Mallory.")]
    assert "Mallory" not in " ".join(clarification_chain("Doe", context)[:-1] + [""])
    assert clarification_chain("Doe", context) == ["Create an account.", "Doe"]


def test_a_turn_that_was_not_a_clarification_ends_the_chain():
    context = [{"role": "user", "content": "Create an account for Mallory."}, clarification(),
               {"role": "user", "content": "Is jdoe enabled?"},
               {"role": "assistant", "content": "Account jdoe is enabled."},
               {"role": "user", "content": "Create an account for John."}, clarification()]

    assert clarification_chain("Doe", context) == ["Create an account for John.", "Doe"]


def test_the_chain_is_bounded():
    context = []
    for index in range(CHAIN_LIMIT + 3):
        context += [{"role": "user", "content": f"detail {index}"}, clarification()]

    chain = clarification_chain("last", context)
    assert len(chain) == CHAIN_LIMIT
    assert chain[-1] == "last"


class RecordingRouter:
    def __init__(self):
        self.requests = []

    async def route_with_repair(self, user_request, *, context=None):
        self.requests.append(user_request)
        return [SpecialistRequest(
            agent_name="account-specialist", instructions="audit-only prose",
            semantic_intent=SemanticIntent(
                summary="Create an account.", effect="mutation", allowed_tools=["account_create"],
                forbidden_tools=[], allowed_arguments={}, forbidden_arguments={}, max_tool_calls=1,
                clarification_required=True, missing_required_arguments=["family_name"]),
        )]


class NeverRuntime:
    async def run(self, task):
        raise AssertionError("specialist must not run during clarification")


class NeverPrimary:
    async def respond(self, *args, **kwargs):
        raise AssertionError("direct response not expected")

    async def synthesize(self, *args, **kwargs):
        raise AssertionError("synthesis not expected")


def test_routing_and_grounding_see_the_whole_chain_but_the_result_keeps_the_reply():
    router = RecordingRouter()
    orchestrator = Orchestrator(router=router, runtime=NeverRuntime(), primary_assistant=NeverPrimary(),
                                require_semantic_intent=True)
    context = [{"role": "user", "content": "Create an account for John as a software developer."},
               clarification()]

    result = asyncio.run(orchestrator.run("Doe", context=context))

    routed = router.requests[0]
    assert "Create an account for John as a software developer." in routed and routed.endswith("Doe")
    assert result.user_request == "Doe"
    assert "family name" in result.answer
