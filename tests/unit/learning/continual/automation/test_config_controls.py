import pytest

from learning.continual.automation.config import (
    ContinualAutomationSettings,
)


def test_stage_defaults_fail_closed():
    settings = ContinualAutomationSettings()
    assert not settings.enabled
    assert not settings.auto_train
    assert not settings.auto_evaluate
    assert not settings.ppo_enabled
    assert not settings.auto_promote


def test_auto_promote_requires_evaluation_and_training():
    with pytest.raises(ValueError, match="AUTO_PROMOTE"):
        ContinualAutomationSettings(
            auto_promote=True,
            auto_evaluate=False,
            auto_train=True,
        )

    with pytest.raises(ValueError, match="AUTO_PROMOTE"):
        ContinualAutomationSettings(
            auto_promote=True,
            auto_evaluate=True,
            auto_train=False,
        )


def test_ppo_requires_training():
    with pytest.raises(ValueError, match="PPO_ENABLED"):
        ContinualAutomationSettings(
            ppo_enabled=True,
            auto_train=False,
        )


def test_autonomous_mode_expands_learning_pipeline():

    settings = (
        ContinualAutomationSettings(
            autonomous_mode=True,
        )
    )

    assert settings.enabled
    assert settings.auto_train
    assert settings.auto_evaluate
    assert settings.ppo_enabled
    assert settings.auto_promote
