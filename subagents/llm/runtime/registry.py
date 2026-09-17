from __future__ import annotations

from dataclasses import (
    dataclass,
)

from threading import (
    RLock,
)

from typing import (
    Callable,
)

from subagents.llm.runtime.base import (
    LLMBackend,
)

from subagents.llm.runtime.memory import (
    release_unused_accelerator_memory,
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

    This object owns backend caching and backend lifecycle.

    GPU admission, request priority, and orchestration do not
    belong here. Those responsibilities sit above the registry
    in ModelManager / InferenceCoordinator.

    Unloading follows a strict order:

        1. remove the registry's strong backend reference
        2. invoke an optional backend close hook
        3. drop the local backend reference
        4. collect unreachable Python objects
        5. release unused CUDA allocator cache

    Live model references elsewhere remain live and are never
    forcefully evicted.
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

    # ========================================================
    # REGISTRATION
    # ========================================================

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
            ] = (
                ModelEntry(
                    backend=(
                        backend
                    ),
                )
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
            ] = (
                ModelEntry(
                    loader=(
                        loader
                    ),
                )
            )

    # ========================================================
    # RESOLUTION
    # ========================================================

    def get(
        self,
        name: str,
    ) -> LLMBackend:
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

            if (
                entry.backend
                is not None
            ):
                return (
                    entry.backend
                )

            if (
                entry.loader
                is None
            ):
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

            return (
                backend
            )

    # ========================================================
    # DISCOVERY
    # ========================================================

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

            if (
                entry
                is None
            ):
                return False

            return (
                entry.backend
                is not None
            )

    # ========================================================
    # UNLOAD
    # ========================================================

    def unload(
        self,
        name: str,
    ) -> bool:
        """
        Remove one cached backend while preserving its lazy
        loader.

        The registry reference is removed before cleanup so the
        backend can actually become unreachable.

        Optional backend-specific cleanup is honored through a
        `close()` method when present.

        Unused CUDA allocator blocks are returned after the local
        backend reference has also been released.
        """

        with self._lock:
            entry = (
                self._models.get(
                    name
                )
            )

            if (
                entry
                is None
            ):
                raise KeyError(
                    f"Model '{name}' "
                    "is not registered."
                )

            if (
                entry.backend
                is None
            ):
                return False

            backend = (
                entry.backend
            )

            # Remove the registry-owned strong reference first.
            entry.backend = (
                None
            )

        try:
            close = (
                getattr(
                    backend,
                    "close",
                    None,
                )
            )

            if callable(
                close
            ):
                close()

        finally:
            # The registry no longer owns the backend and this
            # local reference must also disappear before GC /
            # allocator cleanup can reclaim it.
            del backend

            release_unused_accelerator_memory()

        print(
            "[MODEL] Model unloaded "
            f"'{name}'."
        )

        return True

    def unload_all(
        self,
    ) -> list[str]:
        """
        Unload every currently cached backend.

        Returns the names that were actually unloaded.

        Lazy registrations remain intact and may be loaded again
        later.
        """

        loaded = (
            self.list_loaded_models()
        )

        unloaded: list[
            str
        ] = []

        for name in loaded:
            if self.unload(
                name
            ):
                unloaded.append(
                    name
                )

        # Also clean any allocator blocks belonging to objects
        # that became unreachable outside the registry.
        release_unused_accelerator_memory()

        return (
            unloaded
        )

    # ========================================================
    # LISTING
    # ========================================================

    def list_models(
        self,
    ) -> list[str]:
        with self._lock:
            return (
                list(
                    self._models.keys()
                )
            )

    def list_loaded_models(
        self,
    ) -> list[str]:
        with self._lock:
            return [
                name

                for (
                    name,
                    entry,
                )
                in self._models.items()

                if (
                    entry.backend
                    is not None
                )
            ]

    # ========================================================
    # INTERNAL
    # ========================================================

    def _ensure_available_name(
        self,
        name: str,
    ) -> None:
        if (
            name
            in self._models
        ):
            raise ValueError(
                f"Model '{name}' "
                "is already registered."
            )