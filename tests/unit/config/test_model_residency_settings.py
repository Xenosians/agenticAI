import pytest

from config import (
    Settings,
)


def test_model_residency_defaults_keep_hub_plus_one_specialist():
    settings = (
        Settings(
            _env_file=None,
        )
    )

    assert (
        settings.model_max_loaded_models
        == 2
    )

    assert (
        settings.model_pinned_keys
        == []
    )


def test_model_pinned_keys_are_trimmed_and_deduplicated():
    settings = (
        Settings(
            _env_file=None,
            model_pinned_keys=[
                " hub-main ",
                "jira-func",
                "hub-main",
            ],
        )
    )

    assert (
        settings.model_pinned_keys
        == [
            "hub-main",
            "jira-func",
        ]
    )


def test_model_max_loaded_models_must_be_positive():
    with pytest.raises(
        ValueError,
    ):
        Settings(
            _env_file=None,
            model_max_loaded_models=0,
        )
