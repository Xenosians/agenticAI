from pathlib import Path

from learning.context import load_context_records
from learning.integrations.runtime_hooks import ContinualLearningRuntimeHooks


def test_runtime_hooks_capture_guard_and_gateway_without_authority(tmp_path: Path):
    path = tmp_path / "context.jsonl"
    hooks = ContinualLearningRuntimeHooks(path=path)
    hooks.record_semantic_guard_outcome(allowed=False, code="semantic_argument_not_allowed", trajectory_id="t1", task_id="task1")
    hooks.record_gateway_outcome(tool_name="check_access", gateway_result={"ok": False, "status": "denied", "decision_code": "grounding_failed"}, trajectory_id="t1", task_id="task1")
    records = load_context_records(path)
    assert len(records) == 2
    assert all(item.authority_grant is False for item in records)
    assert all(item.training_eligible is False for item in records)
