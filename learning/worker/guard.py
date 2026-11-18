"""Fit guard and held-out stage gate of the training worker.

Standard library only: the worker package is copied as-is to Proxmox containers, remote GPUs and
other machines, so nothing here may import Janus modules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FIT_VERDICTS = ("unknown", "good", "overfit", "underfit", "diverged")


def _finite(*values: float) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def diagnose_fit(history, *, patience: int = 2, underfit_loss: float | None = None,
                 min_improvement: float = 0.01, plateau_fraction: float = 0.05) -> str:
    """Classify evaluations ``[(step, train_loss, validation_loss), ...]`` (oldest first).

    * ``overfit``: the validation loss has not beaten its earlier best for ``patience``
      evaluations while the training loss kept falling.
    * ``underfit``: the training loss has stalled (less than ``plateau_fraction`` relative
      drop over the window) while still above ``underfit_loss`` (default: 90% of the first
      training loss, i.e. the model barely learned anything).
    * ``diverged``: a loss is NaN or infinite.
    """
    rows = [tuple(row) for row in history]
    if any(not _finite(row[1], row[2]) for row in rows):
        return "diverged"
    if len(rows) < patience + 1:
        return "unknown"
    earlier, window = rows[:-patience], rows[-patience:]
    best_step, best_train, best_validation = min(earlier, key=lambda row: row[2])
    stalled = all(row[2] > best_validation - min_improvement for row in window)
    if stalled and window[-1][1] < best_train - min_improvement:
        return "overfit"
    span = rows[-(patience + 1):]
    first_train, last_train = span[0][1], span[-1][1]
    ceiling = underfit_loss if underfit_loss is not None else rows[0][1] * 0.9
    relative_drop = (first_train - last_train) / first_train if first_train > 0 else 0.0
    if last_train > ceiling and relative_drop < plateau_fraction:
        return "underfit"
    return "good"


@dataclass
class FitGuard:
    """Early stopping that remembers the best evaluation and says why it stopped."""

    patience: int = 2
    min_improvement: float = 0.01
    underfit_loss: float | None = None
    history: list = field(default_factory=list)
    best_step: int | None = None
    best_validation: float = math.inf
    reason: str | None = None
    _misses: int = 0

    def observe(self, *, step: int, train_loss: float, validation_loss: float) -> str:
        if not _finite(train_loss, validation_loss):
            self.reason = "diverged"
            return "stop"
        self.history.append((step, float(train_loss), float(validation_loss)))
        if validation_loss < self.best_validation - self.min_improvement:
            self.best_validation, self.best_step, self._misses = float(validation_loss), step, 0
            return "continue"
        self._misses += 1
        if self._misses < self.patience:
            return "continue"
        best_train = next(row[1] for row in self.history if row[0] == self.best_step) \
            if self.best_step is not None else self.history[0][1]
        if train_loss < best_train - self.min_improvement:
            self.reason = "overfit"
        else:
            verdict = diagnose_fit(self.history, patience=self.patience, underfit_loss=self.underfit_loss,
                                   min_improvement=self.min_improvement)
            self.reason = "underfit" if verdict == "underfit" else "plateau"
        return "stop"

    def summary(self) -> dict:
        return {
            "best_step": self.best_step,
            "best_validation_loss": None if self.best_step is None else self.best_validation,
            "stopped_because": self.reason,
            "evaluations": len(self.history),
            "fit": diagnose_fit(self.history, patience=self.patience, underfit_loss=self.underfit_loss,
                                min_improvement=self.min_improvement) if self.reason != "diverged" else "diverged",
        }


@dataclass(frozen=True)
class StageGate:
    """A stage's result replaces the previous one only when it passes more held-out cases."""

    def judge(self, *, stage: str, passed: int, total: int, previous: dict | None = None,
              cases_sha256: str | None = None) -> dict:
        if total <= 0 or not 0 <= passed <= total:
            raise ValueError("passed must be between 0 and a positive total")
        verdict = {"stage": stage, "passed": passed, "total": total, "cases_sha256": cases_sha256,
                   "previous": None if previous is None else {"passed": previous["passed"],
                                                                "total": previous["total"]}}
        if previous is None:
            kept = passed > 0
            return verdict | {"kept": kept, "reason": "first" if kept else "nothing_passed"}
        earlier_digest = previous.get("cases_sha256")
        if previous["total"] != total or (earlier_digest and cases_sha256 and earlier_digest != cases_sha256):
            raise ValueError("stages must be compared on the same held-out cases")
        kept = passed > previous["passed"]
        return verdict | {"kept": kept, "reason": "better" if kept else "not_better"}
