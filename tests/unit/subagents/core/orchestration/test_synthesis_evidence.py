import asyncio
import json

from subagents.core.definitions.types import AgentResult
from subagents.core.orchestration.primary_assistant import PrimaryAssistant, synthesis_evidence


def worker_result(**overrides):
    values = dict(
        task_id="task-1",
        agent_name="developer-specialist",
        status="success",
        proposed_tool="workspace_process_snapshot",
        proposed_arguments={"relative_path": "."},
        tool_result={"ok": True, "processes": [{"pid": 1, "name": "init"}]},
        answer="Formatted answer that repeats the tool result.",
        presentation={"kind": "table", "rows": [["1", "init"]] * 50},
        execution_provenance={"model_sha256": "a" * 64, "prompt_sha256": "b" * 64},
        task_instructions="List processes.",
    )
    values.update(overrides)
    return AgentResult(**values)


def test_evidence_keeps_the_structured_result_and_drops_audit_and_ui_data():
    evidence = synthesis_evidence(worker_result())
    assert evidence["tool_result"] == {"ok": True, "processes": [{"pid": 1, "name": "init"}]}
    assert evidence["proposed_tool"] == "workspace_process_snapshot"
    for dropped in ("presentation", "execution_provenance", "task_instructions", "answer", "task_id"):
        assert dropped not in evidence


def test_answer_is_the_evidence_when_there_is_no_structured_result():
    evidence = synthesis_evidence(worker_result(tool_result=None, answer="Only prose."))
    assert evidence["answer"] == "Only prose."


class RecordingInference:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "  Summary.  "


def test_direct_response_states_a_length_limit_that_fits_its_output_cap():
    # A provider that reaches the output cap fails the job as truncated, so the
    # model must know the limit, exactly as synthesis does.
    inference = RecordingInference()
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=384, synthesis_max_new_tokens=500)

    assert asyncio.run(assistant.respond("What can you do?")) == "Summary."

    call = inference.calls[0]
    assert call["max_new_tokens"] == 384
    assert "at most 96 words" in call["messages"][0]["content"]
    assert call["messages"][-1] == {"role": "user", "content": "What can you do?"}


def test_synthesis_prompt_is_compact_and_states_the_length_limit():
    inference = RecordingInference()
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    answer = asyncio.run(assistant.synthesize("List processes.", [worker_result()]))
    assert answer == "Summary."
    request = inference.calls[0]["messages"][-1]["content"]
    assert "execution_provenance" not in request and "presentation" not in request
    assert "at most 125 words and no tables" in request
    payload = request.split("TRUSTED SPECIALIST RESULTS:\n", 1)[1].split("\n\n", 1)[0]
    assert json.loads(payload)[0]["tool_result"]["processes"][0]["name"] == "init"
