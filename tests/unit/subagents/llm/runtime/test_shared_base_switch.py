import subagents.llm.runtime.model_manager as model_manager_module

from config import ModelProfileSettings, Settings

from subagents.llm.runtime.base import LLMBackend
from subagents.llm.runtime.model_manager import ModelManager


class SwitchableFakeBackend(LLMBackend):
    def __init__(self) -> None:
        self.active = None
        self.closed = False
        self._hub_overlay_applied = False

    def supports_profile_switching(self) -> bool:
        return True

    def activate_profile_adapter(
        self,
        model_key,
        adapter_path,
    ) -> None:
        self.active = (
            model_key,
            None
            if adapter_path is None
            else str(adapter_path),
        )

    def generate(
        self,
        messages,
        max_new_tokens=256,
    ) -> str:
        return str(self.active)

    def close(self) -> None:
        self.closed = True


def _profile(
    model_path,
    adapter_path=None,
):
    return ModelProfileSettings(
        backend="ministral",
        model_path=model_path,
        adapter_path=adapter_path,
        quantization="bnb4",
        compute_dtype="bfloat16",
        device_map="auto",
    )


def test_same_physical_base_switches_without_rebuild(
    monkeypatch,
    tmp_path,
):
    base = tmp_path / "base"
    base.mkdir()

    adapter = tmp_path / "adapter"
    adapter.mkdir()

    built = []

    def fake_build(
        profile,
        **kwargs,
    ):
        backend = SwitchableFakeBackend()
        built.append(backend)
        return backend

    monkeypatch.setattr(
        model_manager_module,
        "build_model_backend",
        fake_build,
    )

    monkeypatch.setattr(
        model_manager_module,
        "apply_hub_adapter_overlay",
        lambda **kwargs: kwargs["backend"],
    )

    settings = Settings(
        _env_file=None,
        hub_model_key="hub-main",
        model_max_loaded_models=1,
        model_pin_hub=False,
        model_shared_base_enabled=True,
        model_artifact_cache_enabled=False,
        model_profiles={
            "hub-main":
                _profile(base),
            "developer":
                _profile(
                    base,
                    adapter,
                ),
        },
    )

    manager = ModelManager(
        settings=settings,
    )

    hub = manager.load(
        "hub-main"
    )

    developer = manager.load(
        "developer"
    )

    assert developer is hub
    assert len(built) == 1
    assert manager.list_loaded_models() == [
        "developer",
    ]
    assert hub.closed is False

    hub_again = manager.load(
        "hub-main"
    )

    assert hub_again is hub
    assert len(built) == 1
    assert manager.list_loaded_models() == [
        "hub-main",
    ]
    assert hub.closed is False
