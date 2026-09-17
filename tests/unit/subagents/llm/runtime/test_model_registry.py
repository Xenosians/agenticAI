import pytest

from subagents.llm.runtime.base import (
    LLMBackend,
)

from subagents.llm.runtime.registry import (
    ModelRegistry,
)


class FakeBackend(
    LLMBackend
):
    def __init__(
        self,
    ) -> None:
        self.closed = (
            False
        )

    def generate(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 256,
    ) -> str:
        return (
            "fake"
        )

    def close(
        self,
    ) -> None:
        self.closed = (
            True
        )


def test_register_and_get_model(
):
    registry = (
        ModelRegistry()
    )

    backend = (
        FakeBackend()
    )

    registry.register(
        "test-model",
        backend,
    )

    assert (
        registry.exists(
            "test-model"
        )
    )

    assert (
        registry.get(
            "test-model"
        )
        is backend
    )

    assert (
        registry.list_models()
        == [
            "test-model"
        ]
    )


def test_duplicate_model_raises_error(
):
    registry = (
        ModelRegistry()
    )

    backend = (
        FakeBackend()
    )

    registry.register(
        "test-model",
        backend,
    )

    with pytest.raises(
        ValueError
    ):
        registry.register(
            "test-model",
            backend,
        )


def test_unknown_model_raises_error(
):
    registry = (
        ModelRegistry()
    )

    with pytest.raises(
        KeyError
    ):
        registry.get(
            "does-not-exist"
        )


def test_unload_releases_cached_backend_and_preserves_registration(
):
    registry = (
        ModelRegistry()
    )

    backend = (
        FakeBackend()
    )

    registry.register(
        "test-model",
        backend,
    )

    assert (
        registry.is_loaded(
            "test-model"
        )
        is True
    )

    unloaded = (
        registry.unload(
            "test-model"
        )
    )

    assert (
        unloaded
        is True
    )

    assert (
        backend.closed
        is True
    )

    assert (
        registry.exists(
            "test-model"
        )
        is True
    )

    assert (
        registry.is_loaded(
            "test-model"
        )
        is False
    )


def test_unload_unloaded_model_returns_false(
):
    registry = (
        ModelRegistry()
    )

    backend = (
        FakeBackend()
    )

    registry.register(
        "test-model",
        backend,
    )

    assert (
        registry.unload(
            "test-model"
        )
        is True
    )

    assert (
        registry.unload(
            "test-model"
        )
        is False
    )


def test_unload_all_releases_every_loaded_backend(
):
    registry = (
        ModelRegistry()
    )

    first = (
        FakeBackend()
    )

    second = (
        FakeBackend()
    )

    registry.register(
        "first",
        first,
    )

    registry.register(
        "second",
        second,
    )

    unloaded = (
        registry.unload_all()
    )

    assert (
        unloaded
        == [
            "first",
            "second",
        ]
    )

    assert (
        first.closed
        is True
    )

    assert (
        second.closed
        is True
    )

    assert (
        registry.list_loaded_models()
        == []
    )

    assert (
        registry.list_models()
        == [
            "first",
            "second",
        ]
    )