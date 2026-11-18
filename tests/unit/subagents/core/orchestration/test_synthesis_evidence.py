import asyncio
import json

from subagents.core.definitions.types import AgentResult
from subagents.core.orchestration.primary_assistant import (
    PrimaryAssistant, plain_summary, states_an_unknown, synthesis_evidence, unknown_subjects,
)


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


def test_a_tool_whose_answer_is_its_complete_reading_gives_synthesis_that_reading():
    # Live, the synthesis model misread the database inventory's raw fields: it
    # took backup_coverage_known=true for "no visibility" and invented missing
    # backup jobs. The tool's own formatter states the same facts in words.
    reading = 'No Proxmox guest is tagged as a database host. Tag a VM or container "db" in Proxmox.'
    evidence = synthesis_evidence(worker_result(
        agent_name="proxmox-specialist", proposed_tool="proxmox_database_inventory", proposed_arguments={},
        tool_result={"ok": True, "data": {"databases": [], "backup_coverage_known": True}}, answer=reading))
    assert evidence["answer"] == reading
    assert "tool_result" not in evidence
    assert evidence["proposed_tool"] == "proxmox_database_inventory"


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


def test_synthesis_is_told_not_to_add_facts_or_steps_beyond_the_results():
    # Live, a correct inventory reading ("no guest is tagged") was extended with
    # invented advice ("configure backup policies separately", "tag a guest to
    # check backups"). The model may only restate what the results say.
    inference = RecordingInference()
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    asyncio.run(assistant.synthesize("List processes.", [worker_result()]))
    system = inference.calls[0]["messages"][0]["content"]
    assert "add no facts, causes, requirements or next steps that the results do not state" in system
    assert "if the results do not answer part of the request, say that part was not read" in system


def test_synthesis_is_told_to_answer_without_narrating_its_own_process():
    # Live, two of three correct answers after the rule above described how
    # they were made ("answered directly by the Proxmox specialist ... No
    # additional details or inferences were drawn"; "Key details from the
    # Proxmox specialist"). The user talks to one assistant; how the answer
    # was produced is not part of it.
    inference = RecordingInference()
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    asyncio.run(assistant.synthesize("List processes.", [worker_result()]))
    system = inference.calls[0]["messages"][0]["content"]
    assert ("never mention specialists, results, tools or these rules, and never describe how the answer was "
            "produced; state the answer itself") in system


HA_READING = "High availability is not in use: no VM or container is an HA resource. The cluster has quorum."


def ha_result(**overrides):
    values = dict(agent_name="proxmox-specialist", proposed_tool="proxmox_ha_status", proposed_arguments={},
                  tool_result={"ok": True, "data": {"in_use": False, "quorate": True}}, answer=HA_READING,
                  presentation=None, execution_provenance=None, task_instructions=None)
    values.update(overrides)
    return worker_result(**values)


def test_a_one_line_complete_reading_is_the_answer_without_a_synthesis_model_call():
    # Live, synthesis turned this exact reading into "quorum means all required
    # nodes are online" plus a step to check the web UI, against its rules. A
    # reading its tool declares complete, already one statement, is returned
    # as written: nothing can be added to it.
    inference = RecordingInference()
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    assert asyncio.run(assistant.synthesize("Is HA healthy?", [ha_result()])) == HA_READING
    assert inference.calls == []


def test_listings_other_tools_and_several_results_still_go_through_synthesis():
    # A multi-line reading without figures is a listing the model narrows to
    # the question; other tools' raw results and several results need combining.
    cases = [
        [ha_result(answer=HA_READING + "\nHA manager: active.")],
        [worker_result(answer="One line from a tool without a complete reading.")],
        [ha_result(), ha_result()],
        [ha_result(status="failed")],
    ]
    for results in cases:
        inference = RecordingInference()
        assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
        assert asyncio.run(assistant.synthesize("Is HA healthy?", results)) == "Summary."
        assert len(inference.calls) == 1


LOAD_READING = (
    "Node hummingbird over the last hour (60 samples, one a minute):\n"
    "CPU: average 7.4 %, peak 31.0 % at 03:12.\n"
    "Memory: average 3.1 GiB of 5.7 GiB used; least available 2.2 GiB at 03:12.\n"
    "Network: in 13.0 KiB/s average, out 313 B/s average; peak in 77.0 KiB/s."
)


def load_result(**overrides):
    values = dict(agent_name="proxmox-specialist", proposed_tool="proxmox_node_load",
                  proposed_arguments={"node": "hummingbird", "timeframe": "hour"},
                  tool_result={"ok": True, "data": {"samples": 60}}, answer=LOAD_READING,
                  presentation=None, execution_provenance=None, task_instructions=None)
    values.update(overrides)
    return worker_result(**values)


class SummaryInference(RecordingInference):
    def __init__(self, summary):
        super().__init__()
        self.summary = summary

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.summary


def test_a_reading_with_many_figures_is_shown_exactly_after_a_summary_without_figures():
    # Owner's rule (16 November): live, synthesis restated the node-load reading
    # with a peak given as an average and the least available memory as an
    # average. A reading full of figures is shown exactly as the tool wrote it;
    # the model only adds one or two sentences that contain no figures.
    inference = SummaryInference("The node was lightly loaded over the hour, with one short CPU peak.")
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    answer = asyncio.run(assistant.synthesize("How busy was hummingbird in the last hour?", [load_result()]))
    assert answer == ("The node was lightly loaded over the hour, with one short CPU peak.\n\n" + LOAD_READING)
    request = inference.calls[0]["messages"][-1]["content"]
    assert "one or two sentences" in request and "no numbers" in request
    assert LOAD_READING in request


def test_a_summary_that_states_a_figure_is_dropped_and_the_exact_reading_stands_alone():
    for summary in ("CPU averaged 7.4 % with a peak of 31 %.", "Memory peaked at three GiB.", "", "   "):
        inference = SummaryInference(summary)
        assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
        assert asyncio.run(assistant.synthesize("How busy was it?", [load_result()])) == LOAD_READING


def test_a_failed_summary_call_still_gives_the_exact_reading():
    class Failing(RecordingInference):
        async def generate(self, **kwargs):
            raise RuntimeError("provider down")

    assistant = PrimaryAssistant(Failing(), "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    assert asyncio.run(assistant.synthesize("How busy was it?", [load_result()])) == LOAD_READING


def test_a_summary_is_one_or_two_plain_sentences_without_any_figure():
    assert plain_summary("  The node was lightly loaded,\n with one short CPU peak. ") == (
        "The node was lightly loaded, with one short CPU peak.")
    assert plain_summary("Backups are overdue for every guest. Nothing else needs attention.") == (
        "Backups are overdue for every guest. Nothing else needs attention.")
    refused = ("Peak 31 %.", "About seven percent.", "Three of four devices.", "Twenty services run.",
               "It used half the memory.", "A dozen guests.", "Its uplink is nic0.", "Zero errors.",
               "One. Two sentences. Three sentences.", "x" * 400, "", "   ")
    for text in refused:
        assert plain_summary(text) is None, text


def test_readings_with_few_figures_still_go_through_synthesis():
    # Only a reading full of figures is shown exactly; a multi-line listing
    # with a figure or two is still narrowed to the question by synthesis.
    inference = RecordingInference()
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    reading = HA_READING + "\n2 node(s) in HA; manager active."
    assert asyncio.run(assistant.synthesize("Is HA healthy?", [ha_result(answer=reading)])) == "Summary."
    assert "TRUSTED SPECIALIST RESULTS" in inference.calls[0]["messages"][-1]["content"]


# The live maintenance reading of 18 November (hummingbird), verbatim.
MAINTENANCE_READING = (
    "Node hummingbird runs Proxmox VE 9.2.27 on kernel 7.0.14-29-pve; boots EFI with Secure Boot on.\n"
    "Nothing needs attention.\n"
    "Services: 20 of 21 running; corosync stopped as expected (the node is not in a cluster).\n"
    "Certificates: pve-ssl.pem (served to browsers, signed by the Proxmox cluster CA, RSA 2048) expires in 722 days; "
    "pve-root-ca.pem (cluster CA, self-signed, RSA 4096) expires in 3642 days.\n"
    "Updates come from: no-subscription, ceph-squid-no-subscription; turned off: enterprise; no subscription.\n"
    "Pending package updates are not read: Proxmox shows them only with Sys.Modify on the node, which can also "
    "change node settings, so Janus does not use it."
)


def maintenance_result(**overrides):
    values = dict(agent_name="proxmox-specialist", proposed_tool="proxmox_node_maintenance",
                  proposed_arguments={"node": "hummingbird"}, tool_result={"ok": True, "data": {}},
                  answer=MAINTENANCE_READING, presentation=None, execution_provenance=None, task_instructions=None)
    values.update(overrides)
    return worker_result(**values)


def test_a_summary_that_answers_what_the_reading_could_not_read_is_dropped():
    # Live (18 November): asked whether there are pending package updates, the
    # summary said "There are no pending package updates available." above a
    # reading that says they are not read. A summary may name such a subject
    # only to say it is not known; otherwise the exact reading stands alone.
    question = "Does Proxmox node hummingbird need a reboot, and are there pending package updates?"
    invented = "No, the node hummingbird does not require a reboot. There are no pending package updates available."
    inference = SummaryInference(invented)
    assistant = PrimaryAssistant(inference, "hub-main", response_max_new_tokens=64, synthesis_max_new_tokens=500)
    assert asyncio.run(assistant.synthesize(question, [maintenance_result()])) == MAINTENANCE_READING
    request = inference.calls[0]["messages"][-1]["content"]
    assert "never guess" in request
    honest = "The node does not need a reboot; pending package updates cannot be checked."
    assistant = PrimaryAssistant(SummaryInference(honest), "hub-main", response_max_new_tokens=64,
                                 synthesis_max_new_tokens=500)
    assert asyncio.run(assistant.synthesize(question, [maintenance_result()])) == honest + "\n\n" + MAINTENANCE_READING


def test_unknown_subjects_are_read_from_what_the_reading_says_it_does_not_know():
    cases = [
        (MAINTENANCE_READING, "There are no pending package updates.", "Pending updates are not known."),
        ("Not readable with Janus's access: services, certificates.", "All services are running.",
         "Services could not be read."),
        ("Node pve1 has no sample in the last hour (60 expected), so its load is unknown.",
         "The node was lightly loaded.", "Its load is unknown for that hour."),
        ("Storage backup-nas could not be read, so missing or old backups cannot be confirmed.",
         "No backups are missing.", "Backups on that storage cannot be confirmed."),
        ("pve-a: posture could not be read.", "Every device meets the posture rules.",
         "The posture of pve-a is not known."),
        ("/dev/sda: SMART health unknown.", "All disks are healthy.", "The health of /dev/sda is unknown."),
        # One stem in two forms is the same subject.
        ("laptop: online state unknown; key expiry unknown.", "Its key never expires.",
         "Its key expiry is unknown."),
        ("No guest is tagged; backup coverage unknown (Proxmox did not report backup jobs).",
         "Every guest is covered by a backup job.", "Backup coverage is unknown."),
    ]
    for reading, guessed, hedged in cases:
        subjects = unknown_subjects(reading)
        assert subjects, reading
        assert states_an_unknown(guessed, subjects), (reading, guessed)
        assert not states_an_unknown(hedged, subjects), (reading, hedged)
    # Facts the reading does know stay free to state.
    assert not states_an_unknown("The node does not need a reboot.", unknown_subjects(MAINTENANCE_READING))
    assert unknown_subjects("High availability is not in use: no VM or container is an HA resource.") == []


def test_a_synthesized_answer_that_guesses_an_unknown_falls_back_to_the_exact_reading():
    # A multi-line reading with few figures still goes through synthesis; if the
    # answer states what the reading says is unknown, the reading is shown instead.
    reading = HA_READING + "\nHA manager state unknown."
    guessed = PrimaryAssistant(SummaryInference("HA is not in use and the HA manager state is active."), "hub-main",
                               response_max_new_tokens=64, synthesis_max_new_tokens=500)
    assert asyncio.run(guessed.synthesize("Is HA healthy?", [ha_result(answer=reading)])) == reading
    honest_text = "HA is not in use; the HA manager state is unknown."
    honest = PrimaryAssistant(SummaryInference(honest_text), "hub-main", response_max_new_tokens=64,
                              synthesis_max_new_tokens=500)
    assert asyncio.run(honest.synthesize("Is HA healthy?", [ha_result(answer=reading)])) == honest_text
