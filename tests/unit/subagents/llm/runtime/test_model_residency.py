import pytest

import subagents.llm.runtime.model_manager as model_manager_module

from config import (
    ModelProfileSettings,
    Settings,
)

from subagents.llm.runtime.base import (
    LLMBackend,
)

from subagents.llm.runtime.model_manager import (
    ModelManager,
)

from subagents.llm.runtime.residency import (
    ModelResidencyController,
)


class FakeBackend(
    LLMBackend
):
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.closed = False

    def generate(
        self,
        messages,
        max_new_tokens=256,
    ) -> str:
        return self.name

    def close(
        self,
    ) -> None:
        self.closed = True


def _profile(
    tmp_path,
    backend: str,
):
    path = (
        tmp_path
        / backend
    )

    path.mkdir(
        exist_ok=True,
    )

    return (
        ModelProfileSettings(
            backend="hf-causal",
            model_path=path,
            quantization="none",
            model_dtype="float32",
            compute_dtype="float32",
            device_map="cpu",
        )
    )


def test_residency_controller_uses_lru_and_never_selects_pinned():
    residency = (
        ModelResidencyController(
            max_loaded_models=2,
            pinned_model_keys=[
                "hub-main",
            ],
        )
    )

    residency.touch(
        "hub-main"
    )
    residency.touch(
        "account-func"
    )

    victims = (
        residency.plan_load(
            target_model_key=(
                "jira-func"
            ),
            loaded_model_keys=[
                "hub-main",
                "account-func",
            ],
        )
    )

    assert victims == [
        "account-func",
    ]


def test_residency_fails_closed_when_only_pinned_models_can_be_evicted():
    residency = (
        ModelResidencyController(
            max_loaded_models=1,
            pinned_model_keys=[
                "hub-main",
            ],
        )
    )

    with pytest.raises(
        RuntimeError,
        match="pinned models",
    ):
        residency.plan_load(
            target_model_key=(
                "jira-func"
            ),
            loaded_model_keys=[
                "hub-main",
            ],
        )


def test_model_manager_keeps_hub_and_evicts_previous_specialist(
    monkeypatch,
    tmp_path,
):
    built: list[
        FakeBackend
    ] = []

    def fake_build(
        profile,
    ):
        backend = (
            FakeBackend(
                profile.model_path.name
            )
        )

        built.append(
            backend
        )

        return backend

    monkeypatch.setattr(
        model_manager_module,
        "build_model_backend",
        fake_build,
    )

    settings = (
        Settings(
            _env_file=None,
            model_max_loaded_models=2,
            model_pinned_keys=[
                "hub-main",
            ],
            model_profiles={
                "hub-main":
                    _profile(
                        tmp_path,
                        "hub-main",
                    ),

                "account-func":
                    _profile(
                        tmp_path,
                        "account-func",
                    ),

                "jira-func":
                    _profile(
                        tmp_path,
                        "jira-func",
                    ),
            },
        )
    )

    manager = (
        ModelManager(
            settings=settings,
        )
    )

    hub = manager.load(
        "hub-main"
    )

    account = manager.load(
        "account-func"
    )

    assert set(
        manager.list_loaded_models()
    ) == {
        "hub-main",
        "account-func",
    }

    jira = manager.load(
        "jira-func"
    )

    assert hub.closed is False
    assert account.closed is True
    assert jira.closed is False

    assert set(
        manager.list_loaded_models()
    ) == {
        "hub-main",
        "jira-func",
    }

    snapshot = (
        manager
        .residency_snapshot()
    )

    assert (
        snapshot.max_loaded_models
        == 2
    )

    assert (
        snapshot.pinned_models
        == (
            "hub-main",
        )
    )


def test_model_manager_lru_refreshes_when_loaded_model_is_reused(
    monkeypatch,
    tmp_path,
):
    def fake_build(
        profile,
    ):
        return (
            FakeBackend(
                profile.model_path.name
            )
        )

    monkeypatch.setattr(
        model_manager_module,
        "build_model_backend",
        fake_build,
    )

    settings = (
        Settings(
            _env_file=None,
            model_max_loaded_models=2,
            model_pinned_keys=[],
            model_profiles={
                "first":
                    _profile(
                        tmp_path,
                        "first",
                    ),

                "second":
                    _profile(
                        tmp_path,
                        "second",
                    ),

                "third":
                    _profile(
                        tmp_path,
                        "third",
                    ),
            },
        )
    )

    manager = (
        ModelManager(
            settings=settings,
        )
    )

    first = manager.load(
        "first"
    )
    second = manager.load(
        "second"
    )

    # first is now the most recently used resident.
    assert (
        manager.load(
            "first"
        )
        is first
    )

    manager.load(
        "third"
    )

    assert first.closed is False
    assert second.closed is True

    assert set(
        manager.list_loaded_models()
    ) == {
        "first",
        "third",
    }
