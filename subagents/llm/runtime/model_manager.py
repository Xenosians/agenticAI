from threading import (
    RLock,
)

from config import (
    ModelProfileSettings,
    Settings,
)

from subagents.llm.runtime.base import (
    GenerationOutput,
    LLMBackend,
)

from subagents.llm.runtime.factory import (
    build_model_backend,
)

from subagents.llm.runtime.registry import (
    ModelRegistry,
)

from subagents.llm.runtime.residency import (
    ModelResidencyController,
    ModelResidencySnapshot,
)


class ModelManager:
    """
    Process-local owner of configured model backends.

    Responsibilities:
    - logical model profile discovery
    - lazy backend construction
    - backend caching through ModelRegistry
    - bounded model residency / LRU eviction
    - explicit load/unload lifecycle
    - backend-native generation measurements
    - diagnostics

    It deliberately does NOT decide request priority or GPU
    scheduling. InferenceCoordinator owns that boundary.

    Residency is intentionally separate from scheduling:

        ModelManager / ModelResidencyController
            decide which backends may remain loaded.

        InferenceCoordinator / GpuScheduler
            decide when model work may enter the accelerator
            execution boundary.
    """

    def __init__(
        self,
        settings: Settings,
        registry: (
            ModelRegistry
            | None
        ) = None,
    ) -> None:
        self.settings = (
            settings
        )

        self.registry = (
            registry
            if registry is not None
            else ModelRegistry()
        )

        self.residency = (
            ModelResidencyController(
                max_loaded_models=(
                    settings
                    .model_max_loaded_models
                ),

                pinned_model_keys=(
                    [
                        settings
                        .hub_model_key,

                        *settings
                        .model_pinned_keys,
                    ]
                ),
            )
        )

        # ModelRegistry owns per-entry locking. This manager-level
        # lock protects the multi-step "plan eviction -> unload ->
        # load target -> update LRU" transaction.
        self._residency_lock = (
            RLock()
        )

        self._register_profiles()

    # ========================================================
    # REGISTRATION
    # ========================================================

    def _register_profiles(
        self,
    ) -> None:
        for (
            model_key,
            profile,
        ) in (
            self.settings
            .model_profiles
            .items()
        ):
            if not (
                profile.enabled
            ):
                continue

            def loader(
                model_key=model_key,
            ) -> LLMBackend:
                return (
                    self._build_backend(
                        model_key
                    )
                )

            self.registry.register_lazy(
                model_key,
                loader,
            )

    # ========================================================
    # BACKEND CONSTRUCTION
    # ========================================================

    def _build_backend(
        self,
        model_key: str,
    ) -> LLMBackend:
        profile = (
            self.settings
            .require_model_profile(
                model_key
            )
        )

        print(
            "[MODEL] Building backend "
            f"key='{model_key}' "
            f"backend='{profile.backend}'"
        )

        return (
            build_model_backend(
                profile
            )
        )

    # ========================================================
    # PROFILE RESOLUTION
    # ========================================================

    def model_profile(
        self,
        model_key: str,
    ) -> ModelProfileSettings:
        profile = (
            self.settings
            .model_profile(
                model_key
            )
        )

        if (
            profile
            is None
        ):
            raise KeyError(
                "Model profile "
                f"'{model_key}' "
                "is not configured."
            )

        return (
            profile
        )

    # ========================================================
    # DISCOVERY
    # ========================================================

    def exists(
        self,
        model_key: str,
    ) -> bool:
        return (
            self.registry
            .exists(
                model_key
            )
        )

    def is_loaded(
        self,
        model_key: str,
    ) -> bool:
        return (
            self.registry
            .is_loaded(
                model_key
            )
        )

    def list_models(
        self,
    ) -> list[str]:
        return (
            self.registry
            .list_models()
        )

    def list_loaded_models(
        self,
    ) -> list[str]:
        return (
            self.registry
            .list_loaded_models()
        )

    def residency_snapshot(
        self,
    ) -> ModelResidencySnapshot:
        return (
            self.residency
            .snapshot(
                self.list_loaded_models()
            )
        )

    # ========================================================
    # RESIDENCY
    # ========================================================

    def _prepare_residency_for_load(
        self,
        model_key: str,
    ) -> list[str]:
        loaded = (
            self.registry
            .list_loaded_models()
        )

        victims = (
            self.residency
            .plan_load(
                target_model_key=(
                    model_key
                ),

                loaded_model_keys=(
                    loaded
                ),
            )
        )

        evicted: list[str] = []

        for victim in victims:
            print(
                "[MODEL] Residency eviction "
                f"victim='{victim}' "
                f"target='{model_key}'"
            )

            if (
                self.registry
                .unload(
                    victim
                )
            ):
                self.residency.forget(
                    victim
                )

                evicted.append(
                    victim
                )

        return evicted

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def load(
        self,
        model_key: str,
    ) -> LLMBackend:
        with self._residency_lock:
            if not (
                self.registry
                .exists(
                    model_key
                )
            ):
                raise KeyError(
                    f"Model '{model_key}' "
                    "is not registered."
                )

            if not (
                self.registry
                .is_loaded(
                    model_key
                )
            ):
                self._prepare_residency_for_load(
                    model_key
                )

            backend = (
                self.registry
                .get(
                    model_key
                )
            )

            self.residency.touch(
                model_key
            )

            return backend

    def unload(
        self,
        model_key: str,
    ) -> bool:
        with self._residency_lock:
            unloaded = (
                self.registry
                .unload(
                    model_key
                )
            )

            if unloaded:
                self.residency.forget(
                    model_key
                )

            return unloaded

    def unload_all(
        self,
    ) -> list[str]:
        """
        Release every backend currently cached by this model
        manager while preserving all lazy model registrations.
        """

        with self._residency_lock:
            unloaded = (
                self.registry
                .unload_all()
            )

            self.residency.clear()

            return unloaded

    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        model_key: str,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int,
    ) -> str:
        backend = (
            self.load(
                model_key
            )
        )

        try:
            return (
                backend.generate(
                    messages,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                )
            )

        finally:
            # A completed generation is the strongest indication
            # that this model is currently hot.
            self.residency.touch(
                model_key
            )

    def generate_observed(
        self,
        model_key: str,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int,
    ) -> GenerationOutput:
        """
        Execute generation while preserving backend-native
        observability information.

        Backends that support exact token accounting return it
        through GenerationOutput.

        Backends that have not yet implemented specialized
        instrumentation inherit LLMBackend.generate_observed(),
        which remains backward compatible.
        """

        backend = (
            self.load(
                model_key
            )
        )

        try:
            return (
                backend
                .generate_observed(
                    messages,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                )
            )

        finally:
            self.residency.touch(
                model_key
            )
