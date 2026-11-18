"""Choose the machine for each training job (owner decisions 18 November 2026, 23:30 WIB).

Proxmox first for everything that fits; the local GPU when it is idle; a free remote GPU (Kaggle) for
what fits nowhere else. Different jobs run side by side on different machines; a single job is never
split between a CPU worker and a GPU (every step would wait for the CPU).
"""
from __future__ import annotations

from dataclasses import dataclass

KIND_ORDER = {"proxmox": 0, "local": 1, "kaggle": 2}
CPU_STAGES = frozenset({"merge", "sft", "dpo"})   # PPO samples several answers per prompt: GPU only
FP32_FACTOR = 2.0          # bf16 weights on disk become fp32 on a CPU worker
QLORA_FACTOR = 0.28        # 4-bit NF4 weights incl. quantisation constants, against bf16 on disk
OVERHEAD_GIB = 1.2         # activations, LoRA optimizer state, tokenizer, Python
PPO_FACTOR = 1.5           # generation caches for several samples per prompt
MIN_KAGGLE_HOURS = 2.0


@dataclass(frozen=True)
class Worker:
    name: str
    kind: str              # proxmox | local | kaggle
    device: str            # cpu | cuda
    ram_gib: float
    gpu_gib: float = 0.0   # free accelerator memory it may use
    busy: bool = False
    available: bool = True
    weekly_hours_left: float | None = None


def needed_gib(worker: Worker, *, weights_gib: float, stages: list[str]) -> float:
    if worker.device == "cpu":
        need = weights_gib * FP32_FACTOR + OVERHEAD_GIB
    else:
        need = weights_gib * QLORA_FACTOR + OVERHEAD_GIB
    return need * (PPO_FACTOR if "ppo" in stages else 1.0)


def fits(worker: Worker, *, weights_gib: float, stages: list[str]) -> bool:
    if worker.device == "cpu" and not set(stages) <= CPU_STAGES:
        return False
    budget = worker.ram_gib if worker.device == "cpu" else worker.gpu_gib
    return needed_gib(worker, weights_gib=weights_gib, stages=stages) <= budget


def usable(worker: Worker) -> bool:
    if worker.busy or not worker.available:
        return False
    return worker.kind != "kaggle" or (worker.weekly_hours_left or 0) >= MIN_KAGGLE_HOURS


def assign(jobs: list[tuple[str, float, list[str]]], workers: list[Worker]) -> dict[str, str | None]:
    """``jobs`` are ``(name, weights_gib, stages)`` in priority order; each worker takes one job at a time."""
    free = sorted((worker for worker in workers if usable(worker)), key=lambda item: KIND_ORDER[item.kind])
    plan: dict[str, str | None] = {}
    for name, weights_gib, stages in jobs:
        chosen = next((worker for worker in free if fits(worker, weights_gib=weights_gib, stages=stages)), None)
        plan[name] = chosen.name if chosen else None
        if chosen is not None:
            free.remove(chosen)
    return plan
