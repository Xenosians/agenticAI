from __future__ import annotations

from learning.continual.cycle import ContinualLearningCycle, ContinualLearningPolicy


class ContinualLearningController(ContinualLearningCycle):
    """Named Phase-5 controller; behavior remains implemented by the cycle."""


def run_continual_cycle(**kwargs):
    return ContinualLearningController(**kwargs).run()


__all__ = ["ContinualLearningController", "ContinualLearningPolicy", "run_continual_cycle"]
