"""Runtime capture must use the selected persistent path without loading models."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.runtime import ApplicationRuntime
from config import Settings
from learning.context import load_context_records
from learning.integrations.runtime_hooks import ContinualLearningRuntimeHooks


@pytest.mark.parametrize("enabled", [True, False])
def test_runtime_uses_configured_context_ledger(tmp_path: Path, enabled: bool, monkeypatch):
    # The old constructor's default must also stay inside the isolated fixture.
    monkeypatch.setitem(ContinualLearningRuntimeHooks.__init__.__kwdefaults__, "path", tmp_path / "old-default.jsonl")
    target = tmp_path / "persistent" / "context-records.jsonl"
    settings = SimpleNamespace(
        resolve_runtime_path=lambda value: Path(value).resolve(),
        learning_context_path=target,
        learning_trajectory_path=tmp_path / "trajectories.jsonl",
        learning_capture_enabled=enabled,
        hub_model_key="isolated-fixture",
    )
    runtime = ApplicationRuntime(settings=settings, completion_outbox=None,
        outbox_wakeup=asyncio.Event(), phoenix_client=None, mcp=None, approvals=None,
        model_manager=None, gpu_scheduler=None, inference=None, tool_gateway=None, hub=None)
    runtime.learning_hooks.record_gateway_outcome(
        tool_name="check_access", gateway_result={"ok": False, "status": "denied", "decision_code": "grounding_failed"},
        trajectory_id="fixture-trajectory", task_id="fixture-task")

    assert runtime.learning_hooks.recorder.path == target
    assert target.exists() is enabled
    if enabled:
        records = load_context_records(target)
        assert len(records) == 1
        assert records[0].authority_grant is False
        assert records[0].training_eligible is False


def test_context_path_is_a_supported_environment_setting(tmp_path: Path, monkeypatch):
    target = tmp_path / "context.jsonl"
    monkeypatch.setenv("LEARNING_CONTEXT_PATH", str(target))
    assert Settings(_env_file=None).learning_context_path == target
