"""Idle training keeps out of the owner's way: it starts only with room to spare and stops for any request."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from learning.continual.automation import trainer
from learning.continual.automation.limiter import (
    StepLimiter,
    TrainingYielded,
    available_ram_gib,
    gpu_budget,
    on_mains_power,
    room_to_train,
    trainable_gpu_gib,
)
from learning.continual.automation.types import AdaptiveRecipe


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def supply(root: Path, name: str, kind: str, *, online: int | None = None, status: str | None = None) -> None:
    write(root / name / "type", kind + "\n")
    if online is not None:
        write(root / name / "online", f"{online}\n")
    if status is not None:
        write(root / name / "status", status + "\n")


def meminfo(tmp_path: Path, available_kib: int) -> Path:
    path = tmp_path / "meminfo"
    write(path, f"MemTotal:        7733248 kB\nMemFree:          215040 kB\nMemAvailable:    {available_kib} kB\n")
    return path


def test_mains_power_is_required_only_where_the_host_reports_it(tmp_path):
    plugged = tmp_path / "plugged"
    supply(plugged, "AC1", "Mains", online=1)
    supply(plugged, "BAT1", "Battery", status="Full")
    unplugged = tmp_path / "unplugged"
    supply(unplugged, "AC1", "Mains", online=0)
    supply(unplugged, "BAT1", "Battery", status="Discharging")
    battery_only = tmp_path / "battery"
    supply(battery_only, "BAT0", "Battery", status="Discharging")
    assert on_mains_power(plugged) is True
    assert on_mains_power(unplugged) is False
    assert on_mains_power(battery_only) is False
    # No power supply reported at all: a desktop, server or container.
    assert on_mains_power(tmp_path / "desktop") is True


def test_available_ram_is_read_from_memavailable(tmp_path):
    assert available_ram_gib(meminfo(tmp_path, 4456448)) == pytest.approx(4.25, abs=0.01)
    assert available_ram_gib(tmp_path / "missing") is None


def test_a_cycle_needs_mains_power_free_memory_and_a_free_gpu(tmp_path):
    plugged = tmp_path / "plugged"
    supply(plugged, "AC1", "Mains", online=1)
    unplugged = tmp_path / "unplugged"
    supply(unplugged, "AC1", "Mains", online=0)
    roomy, tight = meminfo(tmp_path / "a", 4456448), meminfo(tmp_path / "b", 1048576)
    check = dict(require_mains_power=True, min_free_ram_gib=3.0, min_free_gpu_gib=4.0)
    gpu = lambda: 5.4  # noqa: E731
    assert room_to_train(**check, power_root=plugged, meminfo=roomy, gpu_room_gib=gpu) is None
    assert room_to_train(**check, power_root=unplugged, meminfo=roomy, gpu_room_gib=gpu) == "on_battery"
    assert room_to_train(**check, power_root=plugged, meminfo=tight, gpu_room_gib=gpu) == "low_memory"
    # Unknown memory is not room: fail closed.
    assert room_to_train(**check, power_root=plugged, meminfo=tmp_path / "missing", gpu_room_gib=gpu) == "memory_unknown"
    # Another program holding the GPU is found before anything is prepared.
    assert room_to_train(**check, power_root=plugged, meminfo=roomy, gpu_room_gib=lambda: 2.0) == "gpu_busy"
    assert room_to_train(**check, power_root=plugged, meminfo=roomy, gpu_room_gib=lambda: None) == "gpu_unavailable"
    assert room_to_train(require_mains_power=False, min_free_ram_gib=3.0, min_free_gpu_gib=4.0,
                         power_root=unplugged, meminfo=roomy, gpu_room_gib=gpu) is None


def test_gpu_room_counts_memory_this_process_would_release():
    # Free device memory plus what this process holds (its serving models,
    # which a cycle unloads first); another program's memory is not counted.
    cuda = FakeCuda(free_gib=3.4)
    cuda.reserved_gib = 2.6
    assert trainable_gpu_gib(cuda) == pytest.approx(6.0)
    assert trainable_gpu_gib(SimpleNamespace(is_available=lambda: False)) is None


def test_the_limiter_pauses_between_steps_and_stops_for_a_request():
    slept = []
    waiting = {"reason": None}
    limiter = StepLimiter(should_yield=lambda: waiting["reason"], pause_seconds=1.0, sleep=slept.append,
                          slice_seconds=0.25)
    limiter()
    assert sum(slept) == pytest.approx(1.0)
    waiting["reason"] = "request_waiting"
    with pytest.raises(TrainingYielded) as stopped:
        limiter()
    assert stopped.value.reason == "request_waiting"


def test_a_request_during_the_pause_stops_training_at_once():
    checks = []

    def should_yield():
        checks.append(1)
        return "request_waiting" if len(checks) > 2 else None

    slept = []
    limiter = StepLimiter(should_yield=should_yield, pause_seconds=10.0, sleep=slept.append, slice_seconds=0.5)
    with pytest.raises(TrainingYielded):
        limiter()
    assert sum(slept) < 10.0


class FakeCuda:
    def __init__(self, free_gib: float):
        self.free_gib = free_gib
        self.reserved_gib = 0.0
        self.fractions: list[float] = []
        self.events: list[str] = []

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        # Released cache becomes free device memory.
        self.events.append("empty_cache")
        self.free_gib += self.reserved_gib
        self.reserved_gib = 0.0

    def memory_reserved(self) -> int:
        return int(self.reserved_gib * 2**30)

    def mem_get_info(self):
        self.events.append("mem_get_info")
        return int(self.free_gib * 2**30), int(6 * 2**30)

    def set_per_process_memory_fraction(self, fraction: float) -> None:
        self.fractions.append(fraction)


def test_training_takes_only_its_share_of_the_gpu_and_gives_it_back():
    cuda = FakeCuda(free_gib=5.5)
    with gpu_budget(fraction=0.85, min_free_gib=4.0, cuda=cuda):
        assert cuda.fractions == [0.85]
    assert cuda.fractions == [0.85, 1.0]


def test_cache_released_by_the_unloaded_model_counts_as_free():
    cuda = FakeCuda(free_gib=3.0)
    cuda.reserved_gib = 2.5
    with gpu_budget(fraction=0.85, min_free_gib=4.0, cuda=cuda):
        pass
    assert cuda.events[:2] == ["empty_cache", "mem_get_info"]


def test_training_waits_when_another_program_holds_the_gpu():
    cuda = FakeCuda(free_gib=2.0)
    with pytest.raises(TrainingYielded) as stopped:
        with gpu_budget(fraction=0.85, min_free_gib=4.0, cuda=cuda):
            pytest.fail("training must not start")
    assert stopped.value.reason == "gpu_busy"
    assert cuda.fractions == []


def install_fake_training(monkeypatch, steps: int = 3):
    def train(**_kwargs):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        for _ in range(steps):
            parameter.grad = torch.ones(1)
            optimizer.step()
        return "manifest"

    profile = SimpleNamespace(backend="ministral")
    monkeypatch.setattr(trainer, "train_hub_adapter", train)
    monkeypatch.setattr(trainer, "Settings", lambda: SimpleNamespace(
        hub_model_key="hub-main", require_model_profile=lambda _key: profile))


def candidate(**extra):
    return trainer.train_continual_candidate(materialization_directory=Path("materialized"),
                                             recipe=AdaptiveRecipe(), seed_adapter=None, **extra)


def test_training_checks_the_limiter_after_every_optimizer_step(monkeypatch):
    install_fake_training(monkeypatch)
    calls = []

    def between_steps():
        calls.append(1)
        if len(calls) == 2:
            raise TrainingYielded("request_waiting")

    with pytest.raises(TrainingYielded):
        candidate(between_steps=between_steps)
    assert len(calls) == 2
    # The check belongs to that run only: later optimizers are unaffected.
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.ones(1)
    optimizer.step()
    assert len(calls) == 2


def test_training_without_a_limiter_runs_to_the_end(monkeypatch):
    install_fake_training(monkeypatch)
    assert candidate() == "manifest"
