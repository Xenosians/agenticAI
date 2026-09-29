from types import SimpleNamespace

from learning.cli.reporting import (
    render_developer_training_summary,
)


def test_developer_training_summary_is_compact_and_actionable(
):
    baseline = (
        SimpleNamespace(
            initialization="fresh_lora",
            eval_sft_loss=(
                1.5605189005533855
            ),
        )
    )

    observation = (
        SimpleNamespace(
            step=1,
            checkpoint_directory=(
                "/runtime/checkpoints/"
                "step-000001"
            ),
            train_loss=(
                1.6033371686935425
            ),
            eval_sft_loss=(
                1.5508736371994019
            ),
            score=(
                -1.5508736371994019
            ),
        )
    )

    result = (
        SimpleNamespace(
            run_id="phase5-train-test",
            target_component=(
                "developer-specialist"
            ),
            target_model_key="hub-main",
            sft_optimizer_steps=1,
            dpo_optimizer_steps=0,
            pre_training_observation=(
                baseline
            ),
            checkpoint_observations=[
                observation,
            ],
            best_checkpoint_directory=(
                observation
                .checkpoint_directory
            ),
            base_model_unchanged=True,
            production_activation_performed=False,
            fit_diagnosis=(
                "developer_candidate_requires_heldout"
            ),
            registered_checkpoint_id=(
                "checkpoint-test"
            ),
            adapter_directory=(
                "/runtime/run/"
                "best-adapter"
            ),
        )
    )

    rendered = (
        render_developer_training_summary(
            result,
            seed_checkpoint_id=None,
        )
    )

    assert (
        "COMPLETE (candidate only)"
        in rendered
    )

    assert (
        "Baseline     1.560519"
        in rendered
    )

    assert (
        "Best eval    1.550874 @ step 1"
        in rendered
    )

    assert (
        "Improvement  +0.62%"
        in rendered
    )

    assert (
        "developer heldout evaluation"
        in rendered
    )

    assert (
        "Activation   DISABLED"
        in rendered
    )

    assert (
        "checkpoint-test"
        in rendered
    )


def test_training_summary_handles_missing_baseline(
):
    observation = (
        SimpleNamespace(
            step=1,
            checkpoint_directory=(
                "/runtime/checkpoints/"
                "step-000001"
            ),
            train_loss=1.0,
            eval_sft_loss=0.9,
            score=-0.9,
        )
    )

    result = (
        SimpleNamespace(
            run_id="legacy-run",
            target_component="hub",
            target_model_key="hub-main",
            sft_optimizer_steps=1,
            dpo_optimizer_steps=0,
            pre_training_observation=None,
            checkpoint_observations=[
                observation,
            ],
            best_checkpoint_directory=(
                observation
                .checkpoint_directory
            ),
            base_model_unchanged=True,
            production_activation_performed=False,
            fit_diagnosis="no_guard_signal",
            registered_checkpoint_id=(
                "checkpoint-legacy"
            ),
            adapter_directory=(
                "/runtime/run/"
                "best-adapter"
            ),
        )
    )

    rendered = (
        render_developer_training_summary(
            result,
            seed_checkpoint_id=None,
        )
    )

    assert "Baseline     n/a" in rendered
    assert "Improvement  n/a" in rendered
