from __future__ import annotations

from pathlib import Path
from typing import Any


def _loss(
    value: float | None,
) -> str:
    if value is None:
        return "n/a"

    return f"{value:.6f}"


def _loss_improvement(
    *,
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    """
    Positive means lower loss than baseline.
    Negative means regression.
    """

    if (
        baseline is None
        or candidate is None
        or baseline == 0
    ):
        return None

    return (
        (
            baseline
            - candidate
        )
        / abs(
            baseline
        )
        * 100.0
    )


def _percentage(
    value: float | None,
) -> str:
    if value is None:
        return "n/a"

    return f"{value:+.2f}%"


def _best_observation(
    result: Any,
):
    observations = list(
        result.checkpoint_observations
    )

    if not observations:
        return None

    for observation in observations:

        if (
            str(
                observation
                .checkpoint_directory
            )
            == str(
                result
                .best_checkpoint_directory
            )
        ):
            return observation

    return max(
        observations,
        key=lambda item: (
            item.score
        ),
    )


def _next_gate(
    diagnosis: str,
) -> str:

    if (
        diagnosis
        == "developer_candidate_requires_heldout"
    ):
        return (
            "developer heldout evaluation"
        )

    if (
        diagnosis
        == "overfit_guard_triggered"
    ):
        return (
            "review early-stop / fit evidence"
        )

    if (
        diagnosis
        == "underfit_suspected"
    ):
        return (
            "review training / evaluation coverage"
        )

    return (
        "review evaluation evidence"
    )


def render_developer_training_summary(
    result: Any,
    *,
    seed_checkpoint_id: str | None,
) -> str:
    """
    Compact developer-facing Phase-5 training summary.

    Full provenance remains in the immutable run manifest.
    This is intentionally optimized for humans reading terminals,
    CI logs, handoffs, and debugging sessions.
    """

    baseline = getattr(
        result,
        "pre_training_observation",
        None,
    )

    best = _best_observation(
        result
    )

    baseline_loss = (
        baseline.eval_sft_loss
        if baseline is not None
        else None
    )

    best_loss = (
        best.eval_sft_loss
        if best is not None
        else None
    )

    improvement = (
        _loss_improvement(
            baseline=(
                baseline_loss
            ),
            candidate=(
                best_loss
            ),
        )
    )

    initialization = (
        baseline.initialization
        if baseline is not None
        else (
            "seed_adapter"
            if seed_checkpoint_id
            is not None
            else "unknown"
        )
    )

    best_step = (
        best.step
        if best is not None
        else None
    )

    run_directory = (
        Path(
            result.adapter_directory
        )
        .parent
    )

    manifest_path = (
        run_directory
        / "manifest.json"
    )

    lines = [
        "",
        "Developer QLoRA Result",
        "======================",
        (
            "Status       "
            "COMPLETE (candidate only)"
        ),
        (
            "Target       "
            f"{result.target_component} "
            f"-> {result.target_model_key}"
        ),
        (
            "Run          "
            f"{result.run_id}"
        ),
        (
            "Init         "
            f"{initialization}"
        ),
        (
            "Steps        "
            f"{result.sft_optimizer_steps} SFT / "
            f"{result.dpo_optimizer_steps} DPO"
        ),
        (
            "Baseline     "
            f"{_loss(baseline_loss)}"
        ),
        (
            "Best eval    "
            f"{_loss(best_loss)}"
            + (
                f" @ step {best_step}"
                if best_step is not None
                else ""
            )
        ),
        (
            "Improvement  "
            f"{_percentage(improvement)}"
        ),
        (
            "Base frozen  "
            + (
                "YES"
                if result.base_model_unchanged
                else "NO"
            )
        ),
        (
            "Activation   "
            + (
                "PERFORMED"
                if (
                    result
                    .production_activation_performed
                )
                else "DISABLED"
            )
        ),
        (
            "Guard        "
            f"{result.fit_diagnosis}"
        ),
        (
            "Next gate    "
            f"{_next_gate(result.fit_diagnosis)}"
        ),
        (
            "Checkpoint   "
            f"{result.registered_checkpoint_id}"
        ),
        (
            "Manifest     "
            f"{manifest_path}"
        ),
    ]

    observations = list(
        result.checkpoint_observations
    )

    if baseline is not None or observations:

        lines.extend(
            [
                "",
                "Loss trajectory",
                "---------------",
                (
                    f"{'Step':>4}  "
                    f"{'Train':>10}  "
                    f"{'Eval':>10}  "
                    f"{'vs baseline':>12}"
                ),
            ]
        )

    if baseline is not None:

        lines.append(
            f"{0:>4}  "
            f"{'-':>10}  "
            f"{_loss(baseline_loss):>10}  "
            f"{'baseline':>12}"
        )

    for observation in observations:

        relative = (
            _loss_improvement(
                baseline=(
                    baseline_loss
                ),
                candidate=(
                    observation
                    .eval_sft_loss
                ),
            )
        )

        marker = (
            " *"
            if (
                best is not None
                and observation.step
                == best.step
            )
            else ""
        )

        lines.append(
            f"{observation.step:>4}  "
            f"{_loss(observation.train_loss):>10}  "
            f"{_loss(observation.eval_sft_loss):>10}  "
            f"{_percentage(relative):>12}"
            f"{marker}"
        )

    if best is not None:

        lines.extend(
            [
                "",
                "* selected candidate checkpoint",
            ]
        )

    return "\n".join(
        lines
    )
