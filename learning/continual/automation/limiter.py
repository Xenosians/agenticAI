"""Breathing room for idle training on a shared machine.

Continual training runs on the same GPU and RAM the owner and Janus use. It
therefore starts only with room to spare (mains power, free RAM, a free GPU),
takes a capped share of GPU memory, pauses between optimizer steps, and stops
at the next step when anything needs the machine back. A stopped cycle is
reported as interrupted and its inputs return to pending, so stopping costs
only the work of that cycle.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")
MEMINFO = Path("/proc/meminfo")


class TrainingYielded(Exception):
    """Training stopped early to give the machine back; nothing is promoted."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def on_mains_power(root: Path = POWER_SUPPLY_ROOT) -> bool:
    """True on mains power, or when the host reports no power supply at all (desktop, server)."""
    supplies = sorted(root.iterdir()) if root.is_dir() else []
    mains = [supply for supply in supplies if _read(supply / "type") == "Mains"]
    if mains:
        return any(_read(supply / "online") == "1" for supply in mains)
    batteries = [supply for supply in supplies if _read(supply / "type") == "Battery"]
    return not any(_read(battery / "status") == "Discharging" for battery in batteries)


def available_ram_gib(meminfo: Path = MEMINFO) -> float | None:
    text = _read(meminfo)
    for line in (text or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return None


def _cuda():
    import torch

    return torch.cuda


def trainable_gpu_gib(cuda=None) -> float | None:
    """GPU memory a cycle could use: free memory plus what this process holds.

    A cycle unloads this process's serving models first, so their memory
    counts; memory held by another program does not. None without a GPU.
    """
    cuda = cuda or _cuda()
    if not cuda.is_available():
        return None
    free_bytes, _total = cuda.mem_get_info()
    return (free_bytes + cuda.memory_reserved()) / 2**30


def room_to_train(*, require_mains_power: bool, min_free_ram_gib: float, min_free_gpu_gib: float,
                  power_root: Path = POWER_SUPPLY_ROOT, meminfo: Path = MEMINFO,
                  gpu_room_gib: Callable[[], float | None] = trainable_gpu_gib) -> str | None:
    """Why a cycle may not start now (None when it may). Unknown memory is not room."""
    if require_mains_power and not on_mains_power(power_root):
        return "on_battery"
    available = available_ram_gib(meminfo)
    if available is None:
        return "memory_unknown"
    if available < min_free_ram_gib:
        return "low_memory"
    gpu = gpu_room_gib()
    if gpu is None:
        return "gpu_unavailable"
    if gpu < min_free_gpu_gib:
        return "gpu_busy"
    return None


class StepLimiter:
    """Called after every optimizer step: stop when asked, otherwise pause before the next step.

    The pause is slept in short slices with a check before each, so a request
    that arrives during the pause stops training without waiting it out.
    """

    def __init__(self, *, should_yield: Callable[[], str | None], pause_seconds: float,
                 sleep: Callable[[float], None] = time.sleep, slice_seconds: float = 0.25):
        self.should_yield = should_yield
        self.pause_seconds = pause_seconds
        self.sleep = sleep
        self.slice_seconds = slice_seconds

    def _check(self) -> None:
        reason = self.should_yield()
        if reason is not None:
            raise TrainingYielded(reason)

    def __call__(self) -> None:
        self._check()
        remaining = self.pause_seconds
        while remaining > 0:
            step = min(self.slice_seconds, remaining)
            self.sleep(step)
            remaining -= step
            self._check()


@contextmanager
def gpu_budget(*, fraction: float, min_free_gib: float, cuda=None) -> Iterator[None]:
    """Cap this process's share of GPU memory while training; refuse when the GPU is in use.

    The cap applies to this process's allocator for the duration of the block
    and is lifted afterwards, so serving can use the whole device again.
    """
    cuda = cuda or _cuda()
    if not cuda.is_available():
        raise TrainingYielded("gpu_unavailable")
    # The serving models were just unloaded; return their cached blocks to
    # the device so the free-memory reading includes them.
    cuda.empty_cache()
    free_bytes, _total = cuda.mem_get_info()
    if free_bytes / 2**30 < min_free_gib:
        raise TrainingYielded("gpu_busy")
    cuda.set_per_process_memory_fraction(fraction)
    try:
        yield
    finally:
        cuda.set_per_process_memory_fraction(1.0)
