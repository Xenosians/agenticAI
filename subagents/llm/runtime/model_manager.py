from threading import (
    RLock,
)

from config import (
    ModelProfileSettings,
    Settings,
)

from subagents.llm.runtime.adapter_overlay import (
    active_hub_overlay_target_model_key,
    apply_hub_adapter_overlay,
    apply_profile_adapter_overlay,
)

from subagents.llm.runtime.artifact_cache import (
    ModelArtifactCache,
    physical_model_signature,
)

from subagents.llm.runtime.base import (
    GenerationOutput,
    LLMBackend,
)

from subagents.llm.runtime.factory import (
    build_model_backend,
)

from subagents.llm.runtime.memory import (
    release_unused_accelerator_memory,
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
        checkpoint_id_override: (
            str | None
        ) = None,
        allow_unpromoted_checkpoint_override: bool = False,
    ) -> None:
        self.settings = (
            settings
        )

        self.registry = (
            registry
            if registry is not None
            else ModelRegistry()
        )

        self.checkpoint_id_override = (
            checkpoint_id_override
        )

        self.allow_unpromoted_checkpoint_override = (
            allow_unpromoted_checkpoint_override
        )

        self.artifact_cache = (
            ModelArtifactCache(
                settings
                .model_artifact_cache_root,
                materialize_quantized=(
                    settings
                    .model_artifact_cache_materialize_quantized
                ),
                prefetch=(
                    settings
                    .model_artifact_cache_prefetch
                ),
            )
            if (
                settings
                .model_artifact_cache_enabled
            )
            else None
        )

        self.residency = (
            ModelResidencyController(
                max_loaded_models=(
                    settings
                    .model_max_loaded_models
                ),

                pinned_model_keys=(
                    [
                        *(
                            [
                                settings
                                .hub_model_key,
                            ]
                            if (
                                settings
                                .model_pin_hub
                            )
                            else []
                        ),

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

        cache_resolution = (
            self.artifact_cache.resolve(
                profile
            )
            if (
                self.artifact_cache
                is not None
                and profile.model_path
                is not None
                and (
                    profile.backend
                    == "ministral"
                )
                and (
                    profile.quantization
                    == "bnb4"
                )
            )
            else None
        )

        if (
            cache_resolution
            is None
        ):
            backend = (
                build_model_backend(
                    profile
                )
            )

        else:
            backend = (
                build_model_backend(
                    profile,
                    model_path_override=(
                        cache_resolution.path
                    ),
                )
            )

        if (
            self.artifact_cache
            is not None
            and cache_resolution
            is not None
            and not cache_resolution.hit
        ):
            self.artifact_cache.materialize(
                profile=profile,
                backend=backend,
                cache_key=(
                    cache_resolution.key
                ),
            )

        backend = (
            apply_profile_adapter_overlay(
                backend=backend,
                model_key=model_key,
                adapter_path=(
                    profile.adapter_path
                ),
            )
        )

        model_path = (
            profile.model_path
        )

        if model_path is None:
            return backend

        return (
            apply_hub_adapter_overlay(
                backend=backend,
                model_key=model_key,
                base_model_path=model_path,
                checkpoint_id_override=(
                    self.checkpoint_id_override
                ),
                allow_unpromoted_override=(
                    self.allow_unpromoted_checkpoint_override
                ),
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

    def _find_shareable_loaded_model(
        self,
        target_model_key: str,
    ) -> str | None:
        if not (
            self.settings
            .model_shared_base_enabled
        ):
            return None

        target_profile = (
            self.settings
            .require_model_profile(
                target_model_key
            )
        )

        target_signature = (
            physical_model_signature(
                target_profile
            )
        )

        if not target_signature:
            return None

        dynamic_hub_target = (
            active_hub_overlay_target_model_key()
        )

        for source_model_key in (
            self.registry
            .list_loaded_models()
        ):
            if (
                source_model_key
                == target_model_key
            ):
                continue

            # A dynamic Hub continual-learning adapter is not yet part of the
            # shared-profile switcher. Fail closed to ordinary reload for only
            # the logical model it targets.
            if dynamic_hub_target in {
                source_model_key,
                target_model_key,
            }:
                continue

            source_profile = (
                self.settings
                .require_model_profile(
                    source_model_key
                )
            )

            if (
                physical_model_signature(
                    source_profile
                )
                != target_signature
            ):
                continue

            backend = (
                self.registry
                .get(
                    source_model_key
                )
            )

            supports = getattr(
                backend,
                "supports_profile_switching",
                None,
            )

            if (
                not callable(
                    supports
                )
                or not supports()
            ):
                continue

            if getattr(
                backend,
                "_hub_overlay_applied",
                False,
            ):
                continue

            return source_model_key

        return None

    def _switch_shared_backend(
        self,
        *,
        source_model_key: str,
        target_model_key: str,
    ) -> LLMBackend:
        source_profile = (
            self.settings
            .require_model_profile(
                source_model_key
            )
        )

        target_profile = (
            self.settings
            .require_model_profile(
                target_model_key
            )
        )

        backend = (
            self.registry
            .move_loaded_backend(
                source_model_key,
                target_model_key,
            )
        )

        try:
            activate = getattr(
                backend,
                "activate_profile_adapter",
            )

            activate(
                target_model_key,
                target_profile.adapter_path,
            )

        except Exception:
            self.registry.move_loaded_backend(
                target_model_key,
                source_model_key,
            )

            # Best-effort restore of the previously-good logical profile.
            try:
                activate(
                    source_model_key,
                    source_profile.adapter_path,
                )
            except Exception:
                pass

            raise

        self.residency.forget(
            source_model_key
        )

        self.residency.touch(
            target_model_key
        )

        print(
            "[MODEL] Shared-base switch "
            f"source='{source_model_key}' "
            f"target='{target_model_key}'"
        )

        return backend

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
                reusable_model_key = (
                    self
                    ._find_shareable_loaded_model(
                        model_key
                    )
                )

                if (
                    reusable_model_key
                    is not None
                ):
                    return (
                        self
                        ._switch_shared_backend(
                            source_model_key=(
                                reusable_model_key
                            ),
                            target_model_key=(
                                model_key
                            ),
                        )
                    )

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

    def _post_generation_cleanup(
        self,
        model_key: str,
    ) -> None:
        # Completed generation means this physical/logical model is hot.
        self.residency.touch(
            model_key
        )

        # Shared-base serving deliberately retains live model tensors.
        # Only unused allocator blocks / unreachable temporary tensors
        # are reclaimed here.
        if (
            self.settings
            .model_trim_accelerator_cache_after_generation
        ):
            release_unused_accelerator_memory()

            print(
                "[MODEL] Released unused accelerator cache "
                f"after generation model='{model_key}'"
            )

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
            self._post_generation_cleanup(
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
            self._post_generation_cleanup(
                model_key
            )
