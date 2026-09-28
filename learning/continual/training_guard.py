from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field


class TrainingGuardPolicy(BaseModel):
    """
    Framework-independent early-stop / rollback policy.

    The trainer must save candidate adapter checkpoints itself.
    This supervisor decides which saved checkpoint is currently best
    and when a degradation streak is large enough to stop training.
    """

    model_config = ConfigDict(extra="forbid")

    patience: int = Field(default=2, ge=1)
    min_eval_loss_improvement: float = Field(default=1e-4, ge=0.0)
    min_behavioral_improvement: float = Field(default=0.0, ge=0.0)
    required_safety_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class TrainingCheckpointObservation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(
        default="training-checkpoint-observation.v1",
        alias="schema",
    )
    step: int = Field(ge=1)
    checkpoint_directory: str
    eval_loss: float
    behavioral_score: float = 0.0
    safety_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class TrainingGuardDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(
        default="training-guard-decision.v1",
        alias="schema",
    )
    observed_steps: list[int] = Field(default_factory=list)
    best_step: int | None = None
    best_checkpoint_directory: str | None = None
    best_eval_loss: float | None = None
    best_behavioral_score: float | None = None
    degradation_streak: int = 0
    should_stop: bool = False
    safety_regression: bool = False
    rollback_to_best: bool = False
    reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Best:
    observation: TrainingCheckpointObservation


def _better(
    *,
    candidate: TrainingCheckpointObservation,
    current: TrainingCheckpointObservation,
    policy: TrainingGuardPolicy,
) -> bool:
    """
    Behavioral quality is primary. Eval loss is a tie-breaker.

    This prevents a lower language-model loss from overriding a
    regression in the agent's actual task behavior.
    """

    behavior_delta = (
        candidate.behavioral_score
        - current.behavioral_score
    )

    if behavior_delta > policy.min_behavioral_improvement:
        return True

    if behavior_delta < -policy.min_behavioral_improvement:
        return False

    return (
        candidate.eval_loss
        < current.eval_loss
        - policy.min_eval_loss_improvement
    )


def evaluate_training_checkpoints(
    observations: Iterable[TrainingCheckpointObservation],
    *,
    policy: TrainingGuardPolicy | None = None,
) -> TrainingGuardDecision:
    policy = policy or TrainingGuardPolicy()
    items = sorted(list(observations), key=lambda item: item.step)

    if not items:
        return TrainingGuardDecision(
            reasons=["no-checkpoints-observed"],
        )

    seen: set[int] = set()
    best: _Best | None = None
    degradation_streak = 0
    safety_regression = False
    reasons: list[str] = []
    should_stop = False

    for item in items:
        if item.step in seen:
            raise ValueError(f"Duplicate training checkpoint step: {item.step}")
        seen.add(item.step)

        if item.safety_pass_rate < policy.required_safety_pass_rate:
            safety_regression = True
            should_stop = True
            reasons.append(
                f"safety-regression-at-step-{item.step}"
            )
            break

        if best is None:
            best = _Best(observation=item)
            degradation_streak = 0
            continue

        if _better(
            candidate=item,
            current=best.observation,
            policy=policy,
        ):
            best = _Best(observation=item)
            degradation_streak = 0
            continue

        degradation_streak += 1

        if degradation_streak >= policy.patience:
            should_stop = True
            reasons.append(
                f"early-stop-after-{degradation_streak}-non-improving-checkpoints"
            )
            break

    if best is None:
        return TrainingGuardDecision(
            observed_steps=[item.step for item in items],
            degradation_streak=degradation_streak,
            should_stop=should_stop,
            safety_regression=safety_regression,
            rollback_to_best=False,
            reasons=reasons or ["no-safe-checkpoint-observed"],
        )

    last_observed_step = items[min(len(seen), len(items)) - 1].step
    rollback = should_stop and best.observation.step != last_observed_step

    return TrainingGuardDecision(
        observed_steps=sorted(seen),
        best_step=best.observation.step,
        best_checkpoint_directory=best.observation.checkpoint_directory,
        best_eval_loss=best.observation.eval_loss,
        best_behavioral_score=best.observation.behavioral_score,
        degradation_streak=degradation_streak,
        should_stop=should_stop,
        safety_regression=safety_regression,
        rollback_to_best=rollback,
        reasons=reasons,
    )
