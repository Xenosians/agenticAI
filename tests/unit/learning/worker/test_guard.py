"""The fit guard and held-out stage gate of the training worker (no model, no torch needed)."""
import math

import pytest

from learning.worker.guard import FitGuard, StageGate, diagnose_fit


def test_validation_loss_rising_while_training_falls_is_overfitting():
    history = [(10, 2.0, 2.1), (20, 1.4, 1.6), (30, 0.9, 1.7), (40, 0.5, 1.9)]
    assert diagnose_fit(history, patience=2) == "overfit"


def test_both_losses_staying_high_is_underfitting():
    history = [(10, 3.0, 3.1), (20, 2.98, 3.08), (30, 2.97, 3.07)]
    assert diagnose_fit(history, patience=2, underfit_loss=2.5) == "underfit"


def test_both_losses_falling_together_is_a_good_fit():
    history = [(10, 2.0, 2.1), (20, 1.5, 1.6), (30, 1.1, 1.2)]
    assert diagnose_fit(history, patience=2, underfit_loss=2.5) == "good"


def test_too_few_evaluations_say_nothing_yet():
    assert diagnose_fit([(10, 2.0, 2.1)], patience=2) == "unknown"


def test_the_guard_keeps_the_best_step_and_stops_after_patience_without_progress():
    guard = FitGuard(patience=2, min_improvement=0.01)
    assert guard.observe(step=10, train_loss=2.0, validation_loss=2.0) == "continue"
    assert guard.observe(step=20, train_loss=1.5, validation_loss=1.5) == "continue"
    assert guard.best_step == 20
    assert guard.observe(step=30, train_loss=1.2, validation_loss=1.505) == "continue"
    assert guard.observe(step=40, train_loss=1.0, validation_loss=1.6) == "stop"
    assert guard.best_step == 20
    assert guard.reason == "overfit"


def test_a_non_finite_loss_stops_at_once_and_is_never_the_best():
    guard = FitGuard(patience=3)
    guard.observe(step=10, train_loss=2.0, validation_loss=2.0)
    assert guard.observe(step=20, train_loss=float("nan"), validation_loss=math.inf) == "stop"
    assert guard.reason == "diverged"
    assert guard.best_step == 10


@pytest.mark.parametrize("new, previous, kept", [
    ((9, 10), (8, 10), True),     # more held-out cases pass: kept
    ((8, 10), (8, 10), False),    # equal is not better: the earlier stage stays
    ((7, 10), (8, 10), False),
    ((5, 10), None, True),        # nothing before it: kept when any case passes
    ((0, 10), None, False),
])
def test_a_stage_is_kept_only_when_it_beats_the_stage_before_it_on_held_out_cases(new, previous, kept):
    gate = StageGate()
    verdict = gate.judge(stage="dpo", passed=new[0], total=new[1],
                         previous=None if previous is None else {"passed": previous[0], "total": previous[1]})
    assert verdict["kept"] is kept
    assert verdict["stage"] == "dpo"
    assert verdict["reason"] in {"better", "not_better", "first", "nothing_passed"}


def test_the_gate_refuses_a_different_case_set_rather_than_comparing_it():
    with pytest.raises(ValueError, match="same held-out cases"):
        StageGate().judge(stage="sft", passed=5, total=12, previous={"passed": 4, "total": 10})
