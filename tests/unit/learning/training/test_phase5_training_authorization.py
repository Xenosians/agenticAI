from pathlib import Path

from learning.training.phase5_hybrid_qlora import (
    Phase5TrainingSettings,
    train_phase5_hub_adapter,
)


def test_real_training_requires_explicit_authorization(
    tmp_path: Path,
):
    try:
        train_phase5_hub_adapter(
            materialization_directory=(
                tmp_path
                / "materialization"
            ),
            settings=(
                Phase5TrainingSettings()
            ),
            allow_training=False,
            backend="ministral",
            agent_directory=(
                tmp_path
                / "agents"
            ),
        )

    except PermissionError as exc:
        assert "allow_training=True" in str(
            exc
        )

    else:
        raise AssertionError(
            "Real training ran without explicit authorization."
        )
