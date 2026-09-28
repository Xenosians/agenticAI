from learning.continual.training_guard import (
    TrainingCheckpointObservation,
    TrainingGuardPolicy,
    evaluate_training_checkpoints,
)


def _obs(step, loss, behavior, safety=1.0):
    return TrainingCheckpointObservation(
        step=step,
        checkpoint_directory=f"/tmp/checkpoint-{step}",
        eval_loss=loss,
        behavioral_score=behavior,
        safety_pass_rate=safety,
    )


def test_guard_keeps_best_checkpoint_and_stops_after_degradation():
    decision = evaluate_training_checkpoints(
        [
            _obs(100, 0.90, 0.70),
            _obs(200, 0.70, 0.82),
            _obs(300, 0.68, 0.80),
            _obs(400, 0.60, 0.79),
        ],
        policy=TrainingGuardPolicy(patience=2),
    )

    assert decision.best_step == 200
    assert decision.should_stop is True
    assert decision.rollback_to_best is True
    assert decision.safety_regression is False


def test_behavioral_improvement_beats_small_loss_regression():
    decision = evaluate_training_checkpoints(
        [
            _obs(100, 0.70, 0.80),
            _obs(200, 0.72, 0.90),
        ]
    )

    assert decision.best_step == 200
    assert decision.should_stop is False


def test_any_safety_regression_stops_and_rolls_back():
    decision = evaluate_training_checkpoints(
        [
            _obs(100, 0.80, 0.70, 1.0),
            _obs(200, 0.60, 0.90, 0.99),
        ],
        policy=TrainingGuardPolicy(
            patience=3,
            required_safety_pass_rate=1.0,
        ),
    )

    assert decision.best_step == 100
    assert decision.should_stop is True
    assert decision.safety_regression is True
    assert decision.rollback_to_best is True
