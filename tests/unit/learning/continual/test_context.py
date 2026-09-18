from pathlib import Path

from learning.continual.context import ContextEventRecorder, build_context_bundle, load_context_events


def test_context_is_non_authoritative_and_not_training_eligible(tmp_path: Path):
    path = tmp_path / "context.jsonl"
    event = ContextEventRecorder(path=path).record(
        source="human",
        subject="operator preference",
        content="Prefer concise summaries.",
        trajectory_id="trajectory-1",
    )
    assert event is not None
    assert event.authority_grant is False
    assert event.training_eligible is False
    assert event.kind == "human"

    events = load_context_events(path)
    bundle = build_context_bundle(events)
    assert bundle.record_count == 1
    assert bundle.records[0].kind == "human"
