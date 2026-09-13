from dataclasses import (
    dataclass,
)

from abc import (
    ABC,
    abstractmethod,
)


@dataclass(
    frozen=True
)
class GenerationOutput:
    """
    Model-generation result plus backend-native measurements.

    generated_tokens:
        Number of tokens produced by the model when the backend
        can report it exactly.

    first_token_seconds:
        True time-to-first-token when the backend supports
        measuring it.

        None means TTFT is not yet measurable for that backend.
    """

    text: str

    generated_tokens: (
        int | None
    ) = None

    first_token_seconds: (
        float | None
    ) = None


class LLMBackend(
    ABC
):
    """
    Interface implemented by language-model backends.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 256,
    ) -> str:
        raise NotImplementedError

    def generate_observed(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 256,
    ) -> GenerationOutput:
        """
        Compatibility observability path.

        Backends that can expose exact generation measurements
        should override this method.

        Backends that have not yet been instrumented still work
        normally and simply report no token/TTFT metrics.
        """

        return GenerationOutput(
            text=(
                self.generate(
                    messages,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                )
            )
        )