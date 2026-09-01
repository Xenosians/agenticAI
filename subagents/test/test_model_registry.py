import pytest

from subagents.llm.base import LLMBackend
from subagents.llm.registry import ModelRegistry


class FakeBackend(LLMBackend):
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        return "fake"


def test_register_and_get_model():
    registry = ModelRegistry()
    backend = FakeBackend()

    registry.register(
        "test-model",
        backend,
    )

    assert registry.exists("test-model")
    assert registry.get("test-model") is backend
    assert registry.list_models() == [
        "test-model"
    ]


def test_duplicate_model_raises_error():
    registry = ModelRegistry()
    backend = FakeBackend()

    registry.register(
        "test-model",
        backend,
    )

    with pytest.raises(ValueError):
        registry.register(
            "test-model",
            backend,
        )


def test_unknown_model_raises_error():
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        registry.get(
            "does-not-exist"
        )