from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable

from subagents.llm.base import LLMBackend


ModelLoader = Callable[
    [],
    LLMBackend,
]


@dataclass
class ModelEntry:
    """
    One logical model registration.

    A model may be registered either:

    - eagerly with an already-created backend
    - lazily with a loader function

    Lazy backends are created only on the first get().
    """

    loader: ModelLoader | None = None
    backend: LLMBackend | None = None


class ModelRegistry:
    """
    Stores model backends by logical model name.

    Specialist models may be registered lazily so merely
    registering an agent does not load its model weights.

    The first get(name) loads the backend and caches it.
    Later calls return the same backend instance.
    """

    def __init__(
        self,
    ) -> None:
        self._models: dict[
            str,
            ModelEntry,
        ] = {}

        #
        # Prevent two simultaneous first requests from loading
        # the same model twice.
        #
        self._lock = RLock()

    def register(
        self,
        name: str,
        backend: LLMBackend,
    ) -> None:
        """
        Register an already-created backend.

        Kept for compatibility with code that intentionally
        wants eager registration.
        """

        with self._lock:
            self._ensure_available_name(
                name
            )

            self._models[name] = (
                ModelEntry(
                    backend=backend,
                )
            )

    def register_lazy(
        self,
        name: str,
        loader: ModelLoader,
    ) -> None:
        """
        Register a model loader without loading its weights.

        The loader is executed only when get(name) is called
        for the first time.
        """

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

            self._models[name] = (
                ModelEntry(
                    loader=loader,
                )
            )

    def get(
        self,
        name: str,
    ) -> LLMBackend:
        """
        Return a backend.

        For a lazy model this is the point at which its model
        weights are actually loaded.
        """

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
                "[MODEL] Lazy-loading worker "
                f"model '{name}'..."
            )

            backend = entry.loader()

            if not isinstance(
                backend,
                LLMBackend,
            ):
                raise TypeError(
                    f"Loader for model '{name}' "
                    "did not return an LLMBackend."
                )

            entry.backend = backend

            print(
                "[MODEL] Worker model ready "
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
        """
        Return whether the registered model currently has a
        live backend instance.
        """

        with self._lock:
            entry = self._models.get(
                name
            )

            if entry is None:
                return False

            return (
                entry.backend
                is not None
            )

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
        """
        Useful for diagnostics and later model-management UI.
        """

        with self._lock:
            return [
                name
                for name, entry
                in self._models.items()
                if entry.backend
                is not None
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