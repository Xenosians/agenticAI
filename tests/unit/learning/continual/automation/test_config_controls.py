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


def test_breathing_room_defaults_protect_a_shared_laptop():
    settings = ContinualAutomationSettings()
    assert settings.require_mains_power
    assert settings.step_pause_seconds > 0
    assert 0 < settings.gpu_memory_fraction < 1
    assert settings.min_free_ram_gib > 0 and settings.min_free_gpu_gib > 0


def test_breathing_room_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("CONTINUAL_STEP_PAUSE_SECONDS", "3.5")
    monkeypatch.setenv("CONTINUAL_MIN_FREE_RAM_GIB", "2.5")
    monkeypatch.setenv("CONTINUAL_MIN_FREE_GPU_GIB", "4.5")
    monkeypatch.setenv("CONTINUAL_GPU_MEMORY_FRACTION", "0.7")
    monkeypatch.setenv("CONTINUAL_REQUIRE_MAINS_POWER", "false")
    settings = ContinualAutomationSettings.from_env()
    assert (settings.step_pause_seconds, settings.min_free_ram_gib, settings.min_free_gpu_gib,
            settings.gpu_memory_fraction, settings.require_mains_power) == (3.5, 2.5, 4.5, 0.7, False)


def test_gpu_memory_fraction_must_leave_the_gpu_a_share():
    with pytest.raises(ValueError):
        ContinualAutomationSettings(gpu_memory_fraction=0.0)
    with pytest.raises(ValueError):
        ContinualAutomationSettings(gpu_memory_fraction=1.5)
