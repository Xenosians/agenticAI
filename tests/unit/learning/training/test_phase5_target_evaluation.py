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
