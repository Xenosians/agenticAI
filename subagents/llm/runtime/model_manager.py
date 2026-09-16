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


class ModelManager:
    """
    Process-local owner of configured model backends.

    Responsibilities:
    - logical model profile discovery
    - lazy backend construction
    - backend caching through ModelRegistry
    - explicit load/unload lifecycle
    - backend-native generation measurements
    - diagnostics

    It deliberately does NOT decide request priority or GPU
    scheduling. InferenceCoordinator owns that boundary.
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

        self._register_profiles()

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
            if not profile.enabled:
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

        if profile is None:
            raise KeyError(
                "Model profile "
                f"'{model_key}' "
                "is not configured."
            )

        return profile

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

    def load(
        self,
        model_key: str,
    ) -> LLMBackend:
        return (
            self.registry
            .get(
                model_key
            )
        )

    def unload(
        self,
        model_key: str,
    ) -> bool:
        return (
            self.registry
            .unload(
                model_key
            )
        )

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

        return (
            backend.generate(
                messages,
                max_new_tokens=(
                    max_new_tokens
                ),
            )
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

        return (
            backend
            .generate_observed(
                messages,
                max_new_tokens=(
                    max_new_tokens
                ),
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