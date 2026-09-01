from subagents.llm.base import LLMBackend


class ModelRegistry:
    """
    Stores model backends by logical model name.
    """

    def __init__(self) -> None:
        self._models: dict[str, LLMBackend] = {}

    def register(
        self,
        name: str,
        backend: LLMBackend,
    ) -> None:
        if name in self._models:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        self._models[name] = backend

    def get(
        self,
        name: str,
    ) -> LLMBackend:
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' is not registered."
            )

        return self._models[name]

    def exists(
        self,
        name: str,
    ) -> bool:
        return name in self._models

    def list_models(self) -> list[str]:
        return list(self._models.keys())