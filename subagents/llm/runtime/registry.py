from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable

from subagents.llm.runtime.base import (
    LLMBackend,
)


ModelLoader = Callable[
    [],
    LLMBackend,
]


@dataclass
class ModelEntry:
    """
    One logical model registration.

    A model may be registered either eagerly with an existing
    backend or lazily through a loader.
    """

    loader: ModelLoader | None = None

    backend: (
        LLMBackend | None
    ) = None


class ModelRegistry:
    """
    Thread-safe logical model registry.

    This object owns backend caching only.

    GPU admission, request priority, and orchestration do not
    belong here. Those responsibilities sit above the registry
    in ModelManager / InferenceCoordinator.
    """

    def __init__(
        self,
    ) -> None:
        self._models: dict[
            str,
            ModelEntry,
        ] = {}

        self._lock = (
            RLock()
        )

    def register(
        self,
        name: str,
        backend: LLMBackend,
    ) -> None:
        with self._lock:
            self._ensure_available_name(
                name
            )

            self._models[
                name
            ] = ModelEntry(
                backend=backend,
            )

    def register_lazy(
        self,
        name: str,
        loader: ModelLoader,
    ) -> None:
        if not callable(
            loader
        ):
            raise TypeError(
                "Model loader must be callable."
            )

        with self._lock:
            self._ensure_available_name(
                name
            )

            self._models[
                name
            ] = ModelEntry(
                loader=loader,
            )

    def get(
        self,
        name: str,
    ) -> LLMBackend:
        with self._lock:
            entry = self._models.get(
                name
            )

            if entry is None:
                raise KeyError(
                    f"Model '{name}' "
                    "is not registered."
                )

            if entry.backend is not None:
                return entry.backend

            if entry.loader is None:
                raise RuntimeError(
                    f"Model '{name}' has no "
                    "backend or loader."
                )

            print(
                "[MODEL] Lazy-loading "
                f"model '{name}'..."
            )

            backend = (
                entry.loader()
            )

            if not isinstance(
                backend,
                LLMBackend,
            ):
                raise TypeError(
                    f"Loader for model '{name}' "
                    "did not return an LLMBackend."
                )

            entry.backend = (
                backend
            )

            print(
                "[MODEL] Model ready "
                f"'{name}'."
            )

            return backend

    def exists(
        self,
        name: str,
    ) -> bool:
        with self._lock:
            return (
                name
                in self._models
            )

    def is_loaded(
        self,
        name: str,
    ) -> bool:
        with self._lock:
            entry = (
                self._models.get(
                    name
                )
            )

            if entry is None:
                return False

            return (
                entry.backend
                is not None
            )

    def unload(
        self,
        name: str,
    ) -> bool:
        """
        Remove a cached backend while preserving its loader.

        This is the lifecycle seam future HOT/WARM/COLD eviction
        logic can call. Actual VRAM cleanup remains backend-specific.
        """

        with self._lock:
            entry = (
                self._models.get(
                    name
                )
            )

            if entry is None:
                raise KeyError(
                    f"Model '{name}' "
                    "is not registered."
                )

            if entry.backend is None:
                return False

            backend = (
                entry.backend
            )

            entry.backend = None

        close = getattr(
            backend,
            "close",
            None,
        )

        if callable(
            close
        ):
            close()

        return True

    def list_models(
        self,
    ) -> list[str]:
        with self._lock:
            return list(
                self._models.keys()
            )

    def list_loaded_models(
        self,
    ) -> list[str]:
        with self._lock:
            return [
                name

                for name, entry
                in self._models.items()

                if (
                    entry.backend
                    is not None
                )
            ]

    def _ensure_available_name(
        self,
        name: str,
    ) -> None:
        if name in self._models:
            raise ValueError(
                f"Model '{name}' "
                "is already registered."
            )