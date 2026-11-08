"""The continual service trains small specialists when idle, before (or instead of) the Hub.

Specialist training needs no queued learning events: its material is the
reviewed specialist training sets. It shares the Hub cycle's breathing-room
checks and gives way to requests the same way.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

from learning.continual.automation import service as service_module
from learning.continual.automation.config import ContinualAutomationSettings
from learning.continual.automation.service import ContinualAutomationService
from learning.continual.automation.specialists import SpecialistCycleResult
from learning.paths import REPOSITORY_ROOT


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


class Specialists:
    def __init__(self, results):
        self.results = list(results)
        self.cycles = 0

    async def run_cycle(self):
        self.cycles += 1
        return self.results.pop(0) if self.results else None


def build(tmp_path: Path, monkeypatch, *, specialist_training=True, results=()):
    sources = tmp_path / "corpus-sources.json"
    sources.write_text("[]", encoding="utf-8")
    runtime = SimpleNamespace(
        settings=SimpleNamespace(hub_model_key="hub-main", agents_dir=REPOSITORY_ROOT / "subagents" / "agents"),
        active_jobs={}, gpu_scheduler=Scheduler(), inference=Inference(),
        model_manager=SimpleNamespace(unload_all=lambda: []),
    )
    settings = ContinualAutomationSettings(
        enabled=True, auto_train=True, specialist_training=specialist_training, single_gpu_idle_only=False,
        step_pause_seconds=0.0, state_db_path=tmp_path / "continuous.sqlite3", corpus_sources_path=sources,
    )
    specialists = Specialists(results)
    monkeypatch.setattr(service_module, "room_to_train", lambda **_: None)
    service = ContinualAutomationService(runtime=runtime, settings=settings, specialist_trainer=specialists)
    return service, runtime, specialists


def finished(name="account"):
    return SpecialistCycleResult(specialist=name, outcome="candidate", gate_passed=True, passed=9, total=9)


def test_an_idle_service_trains_a_specialist_without_waiting_for_learning_events(tmp_path, monkeypatch):
    service, runtime, specialists = build(tmp_path, monkeypatch, results=[finished()])
    assert service.store.pending_event_counts()["total"] == 0
    result = asyncio.run(service.run_cycle())
    assert specialists.cycles == 1 and result.specialist == "account"
    # The serving Hub is warmed again after training released the GPU.
    assert runtime.inference.warmed == 1


def test_specialist_training_waits_for_room_like_the_hub_cycle(tmp_path, monkeypatch):
    service, _runtime, specialists = build(tmp_path, monkeypatch, results=[finished()])
    monkeypatch.setattr(service_module, "room_to_train", lambda **_: "on_battery")
    assert asyncio.run(service.run_cycle()) is None
    assert specialists.cycles == 0


def test_specialist_training_waits_for_an_idle_machine(tmp_path, monkeypatch):
    service, runtime, specialists = build(tmp_path, monkeypatch, results=[finished()])
    service.settings = service.settings.model_copy(update={"single_gpu_idle_only": True,
                                                           "idle_seconds_before_training": 60.0})
    assert asyncio.run(service.run_cycle()) is None
    service._last_activity -= 120
    runtime.active_jobs["chat"] = object()
    assert asyncio.run(service.run_cycle()) is None
    assert specialists.cycles == 0


def test_an_interrupted_specialist_run_waits_a_full_idle_window(tmp_path, monkeypatch):
    stopped = SpecialistCycleResult(specialist="account", outcome="interrupted", reason="request_waiting")
    service, _runtime, _specialists = build(tmp_path, monkeypatch, results=[stopped])
    service.settings = service.settings.model_copy(update={"single_gpu_idle_only": True,
                                                           "idle_seconds_before_training": 60.0})
    service._last_activity -= 120
    assert asyncio.run(service.run_cycle()).outcome == "interrupted"
    assert not service._idle_ready()


def test_specialist_training_is_off_unless_switched_on(tmp_path, monkeypatch):
    service, _runtime, specialists = build(tmp_path, monkeypatch, specialist_training=False, results=[finished()])
    assert asyncio.run(service.run_cycle()) is None
    assert specialists.cycles == 0


def test_with_nothing_new_for_specialists_the_hub_cycle_runs_as_before(tmp_path, monkeypatch):
    service, _runtime, specialists = build(tmp_path, monkeypatch, results=[])
    # No learning events queued: the Hub cycle has nothing to train either.
    assert asyncio.run(service.run_cycle()) is None
    assert specialists.cycles == 1


def test_specialist_training_is_controlled_from_the_environment(monkeypatch):
    monkeypatch.setenv("CONTINUAL_SPECIALIST_TRAINING", "true")
    monkeypatch.setenv("CONTINUAL_HUB_TRAINING", "false")
    monkeypatch.setenv("CONTINUAL_SPECIALIST_MAX_WEIGHTS_GIB", "1.5")
    settings = ContinualAutomationSettings.from_env()
    assert (settings.specialist_training, settings.hub_training, settings.specialist_max_weights_gib) == (
        True, False, 1.5)


def test_the_hub_cycle_can_be_left_off_while_specialists_train(tmp_path, monkeypatch):
    service, _runtime, _specialists = build(tmp_path, monkeypatch, results=[])
    # One queued event is enough to start a Hub cycle here, so only the switch keeps it off.
    service.settings = service.settings.model_copy(update={"hub_training": False, "min_events_per_cycle": 1})
    monkeypatch.setattr(service_module, "train_continual_candidate",
                        lambda **_: (_ for _ in ()).throw(AssertionError("the Hub cycle must not train")))
    service.enqueue_trajectory({
        "trajectory_id": "t", "job_id": "j", "attempt": 1, "hub_model": "hub-main",
        "signals": {"overall_success": True, "had_error": False, "waiting_approval": False}, "steps": []})
    assert asyncio.run(service.run_cycle()) is None
    assert service.store.latest_cycle() is None
    assert service.store.pending_event_counts()["total"] == 1


def test_status_reports_the_last_specialist_runs(tmp_path, monkeypatch):
    service, _runtime, _specialists = build(tmp_path, monkeypatch, results=[finished()])
    asyncio.run(service.run_cycle())
    status = service.status()
    assert status["controls"]["specialist_training"] is True
    assert status["latest_specialist_cycle"]["specialist"] == "account"
