from types import SimpleNamespace

from learning.training.phase5_hybrid_qlora import (
    Phase5TrainingSettings,
    _target_checkpoint_metrics,
)


def test_developer_checkpoint_uses_validation_loss_not_hub_contract():

    materialization = (
        SimpleNamespace(
            evaluation_contract=(
                "developer_sft_loss"
            )
        )
    )

    (
        contract_rate,
        metric_name,
        metric_value,
        score,
    ) = _target_checkpoint_metrics(
        materialization=(
            materialization
        ),

        loaded=None,

        sft_validation=[
            object()
        ],

        environment=None,

        settings=(
            Phase5TrainingSettings()
        ),

        eval_sft_loss=0.625,

        eval_dpo_loss=None,
    )

    assert (
        contract_rate
        == 0.0
    )

    assert (
        metric_name
        == "developer_validation_loss"
    )

    assert (
        metric_value
        == 0.625
    )

    assert (
        score
        == -0.625
    )


def test_pre_training_developer_observation_is_step_zero(
    monkeypatch,
):
    from learning.training import (
        phase5_hybrid_qlora as module,
    )

    monkeypatch.setattr(
        module,
        "_evaluate_losses",
        lambda **_kwargs: (
            0.75,
            None,
        ),
    )

    monkeypatch.setattr(
        module,
        "_target_checkpoint_metrics",
        lambda **_kwargs: (
            0.0,
            "developer_validation_loss",
            0.75,
            -0.75,
        ),
    )

    observation = (
        module
        ._evaluate_pre_training_observation(
            materialization=object(),
            loaded=object(),
            sft_validation=[
                object(),
            ],
            dpo_validation=[],
            environment=None,
            settings=(
                module
                .Phase5TrainingSettings()
            ),
            initialization=(
                "fresh_lora"
            ),
        )
    )

    assert observation.optimizer_step == 0

    assert (
        observation.initialization
        == "fresh_lora"
    )

    assert (
        observation.eval_sft_loss
        == 0.75
    )

    assert (
        observation.evaluation_metric_name
        == "developer_validation_loss"
    )

    assert (
        observation.evaluation_metric_value
        == 0.75
    )

    assert observation.score == -0.75
