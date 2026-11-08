"""A continual cycle gives way to the owner: no room means no cycle, and a request stops training."""
import asyncio
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from learning.continual.automation import service as service_module
from learning.continual.automation.config import ContinualAutomationSettings
from learning.continual.automation.limiter import TrainingYielded
from learning.continual.automation.service import ContinualAutomationService
from learning.paths import REPOSITORY_ROOT

TRAJECTORY = {
    "trajectory_id": "trajectory-limiter", "job_id": "job-limiter", "attempt": 1, "hub_model": "hub-main",
    "signals": {"overall_success": True, "had_error": False, "waiting_approval": False}, "steps": [],
}


class Scheduler:
    active = False
    queue_depth = 0

    async def run(self, *, priority, work):
        return await asyncio.to_thread(work)


class Inference:
    def __init__(self):
        self.warmed = 0

    async def warm(self, model_key, *, stage):
        self.warmed += 1


def build(tmp_path: Path, monkeypatch):
    sources = tmp_path / "corpus-sources.json"
    sources.write_text("[]", encoding="utf-8")
    runtime = SimpleNamespace(
        settings=SimpleNamespace(hub_model_key="hub-main", agents_dir=REPOSITORY_ROOT / "subagents" / "agents"),
        active_jobs={}, gpu_scheduler=Scheduler(), inference=Inference(),
        model_manager=SimpleNamespace(unload_all=lambda: []),
    )
    settings = ContinualAutomationSettings(
        enabled=True, auto_train=True, min_events_per_cycle=1, single_gpu_idle_only=False,
        step_pause_seconds=0.0, state_db_path=tmp_path / "continuous.sqlite3", corpus_sources_path=sources,
    )
    service = ContinualAutomationService(runtime=runtime, settings=settings)
    monkeypatch.setattr(service, "_select_pages", lambda **_: [])
    monkeypatch.setattr(service, "_behavior_materialization", lambda: tmp_path / "materialized")
    monkeypatch.setattr(service_module, "seed_adapter_directory", lambda: (None, None))
    monkeypatch.setattr(service_module, "room_to_train", lambda **_: None)
    monkeypatch.setattr(service_module, "on_mains_power", lambda: True)
    monkeypatch.setattr(service_module, "gpu_budget", lambda **_: nullcontext())
    assert service.enqueue_trajectory(TRAJECTORY)
    return service, runtime


def test_a_request_during_training_stops_it_and_keeps_its_inputs(tmp_path, monkeypatch):
    service, runtime = build(tmp_path, monkeypatch)

    def train(*, materialization_directory, recipe, seed_adapter, between_steps):
        between_steps()
        runtime.active_jobs["chat"] = object()
        between_steps()
        raise AssertionError("training must stop for the chat")

    monkeypatch.setattr(service_module, "train_continual_candidate", train)
    result = asyncio.run(service.run_cycle())
    assert (result.state, result.training_executed, result.guard_signal) == ("interrupted", False, "request_waiting")
    assert result.blocked_reasons == [] and result.candidate_checkpoint_id is None
    assert service.store.pending_event_counts()["total"] == 1
    assert runtime.inference.warmed == 1


def test_a_request_waiting_at_handover_keeps_the_serving_model(tmp_path, monkeypatch):
    service, runtime = build(tmp_path, monkeypatch)
    unloaded = []
    runtime.model_manager = SimpleNamespace(unload_all=lambda: unloaded.append(1))
    runtime.gpu_scheduler.queue_depth = 1
    monkeypatch.setattr(service_module, "train_continual_candidate", lambda **_: pytest.fail("must not train"))
    result = asyncio.run(service.run_cycle())
    assert (result.state, result.guard_signal) == ("interrupted", "request_waiting")
    assert unloaded == []


def test_no_room_means_no_cycle_and_nothing_claimed(tmp_path, monkeypatch):
    service, _runtime = build(tmp_path, monkeypatch)
    monkeypatch.setattr(service_module, "room_to_train", lambda **_: "low_memory")
    monkeypatch.setattr(service_module, "train_continual_candidate", lambda **_: pytest.fail("must not train"))
    assert asyncio.run(service.run_cycle()) is None
    assert service.store.latest_cycle() is None
    assert service.store.pending_event_counts()["total"] == 1


def test_training_data_is_prepared_off_the_event_loop(tmp_path, monkeypatch):
    service, _runtime = build(tmp_path, monkeypatch)
    threads = {}

    def prepare():
        threads["prepare"] = threading.get_ident()
        return tmp_path / "materialized"

    def stop(**_):
        raise TrainingYielded("stopping")

    monkeypatch.setattr(service, "_behavior_materialization", prepare)
    monkeypatch.setattr(service_module, "train_continual_candidate", stop)

    async def cycle():
        threads["loop"] = threading.get_ident()
        return await service.run_cycle()

    assert asyncio.run(cycle()).state == "interrupted"
    assert threads["prepare"] != threads["loop"]


def test_training_gives_way_to_shutdown_and_battery(tmp_path, monkeypatch):
    service, _runtime = build(tmp_path, monkeypatch)
    assert service._yield_reason() is None
    monkeypatch.setattr(service_module, "on_mains_power", lambda: False)
    assert service._yield_reason() == "on_battery"
    monkeypatch.setattr(service_module, "on_mains_power", lambda: True)
    service.request_stop()
    assert service._yield_reason() == "stopping"


def test_an_interrupted_cycle_waits_a_full_idle_window_before_trying_again(tmp_path, monkeypatch):
    service, runtime = build(tmp_path, monkeypatch)
    service.settings = service.settings.model_copy(update={"single_gpu_idle_only": True,
                                                           "idle_seconds_before_training": 60.0})
    service._last_activity -= 120
    assert service._idle_ready()

    def chat_arrives(**_):
        runtime.active_jobs["chat"] = object()
        raise TrainingYielded("request_waiting")

    monkeypatch.setattr(service_module, "train_continual_candidate", chat_arrives)
    assert asyncio.run(service.run_cycle()).state == "interrupted"
    runtime.active_jobs.clear()
    assert not service._idle_ready()
