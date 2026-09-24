import json

from dataclasses import (
    asdict,
)

from config import (
    get_settings,
)

from subagents.core.orchestration.conversation_context import (
    conversation_messages,
)

from subagents.core.definitions.types import (
    AgentResult,
)

from subagents.llm.runtime.inference import (
    InferenceEngine,
)

from subagents.llm.runtime.scheduler import (
    InferencePriority,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


class PrimaryAssistant:
    """
    Main conversational model.

    It serves two purposes:

    1. direct responses when no specialist is needed;
    2. synthesis of successful trusted specialist results.

    It does not own model weights or runtime resources.
    """

    def __init__(
        self,
        inference: InferenceEngine,
        model_key: str,

        response_max_new_tokens: (
            int | None
        ) = None,

        synthesis_max_new_tokens: (
            int | None
        ) = None,
    ) -> None:

        self.inference = (
            inference
        )

        self.model_key = (
            model_key
        )

        settings = (
            get_settings()
        )

        self.response_max_new_tokens = (
            response_max_new_tokens
            if response_max_new_tokens
            is not None
            else (
                settings
                .primary_response_max_new_tokens
            )
        )

        self.synthesis_max_new_tokens = (
            synthesis_max_new_tokens
            if synthesis_max_new_tokens
            is not None
            else (
                settings
                .primary_synthesis_max_new_tokens
            )
        )

        if (
            not isinstance(
                self.response_max_new_tokens,
                int,
            )
            or self.response_max_new_tokens
            < 1
        ):

            raise ValueError(
                "Primary response max_new_tokens "
                "must be a positive integer."
            )

        if (
            not isinstance(
                self.synthesis_max_new_tokens,
                int,
            )
            or self.synthesis_max_new_tokens
            < 1
        ):

            raise ValueError(
                "Primary synthesis max_new_tokens "
                "must be a positive integer."
            )

        self.system_prompt = (
            load_prompt(
                "primary_assistant.txt"
            )
        )

    async def respond(
        self,
        user_request: str,
        *,
        context: list[dict[str, str]] | None = None,
    ) -> str:
        messages = [
            {
                "role":
                    "system",

                "content":
                    self.system_prompt,
            },

            *conversation_messages(
                context,
                max_turns=24,
            ),

            {
                "role":
                    "user",

                "content":
                    user_request,
            },
        ]

        response = (
            await self.inference.generate(
                model_key=(
                    self.model_key
                ),
                messages=messages,
                max_new_tokens=(
                    self.response_max_new_tokens
                ),
                priority=(
                    InferencePriority
                    .PRIMARY_RESPONSE
                ),
            )
        )

        return response.strip()

    async def synthesize(
        self,
        user_request: str,
        results: list[
            AgentResult
        ],
        *,
        context: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Produce the final Main-LLM answer from successful
        specialist results.

        Structured results remain authoritative. The Main model
        is responsible only for user-facing synthesis.
        """

        specialist_payload = [
            asdict(
                result
            )

            for result in results
        ]

        messages = [
            {
                "role":
                    "system",

                "content":
                    self.system_prompt,
            },

            *conversation_messages(
                context,
                max_turns=24,
            ),

            {
                "role":
                    "user",

                "content": (
                    "ORIGINAL USER REQUEST:\n"
                    f"{user_request}"
                    "\n\n"
                    "TRUSTED SPECIALIST RESULTS:\n"
                    f"{json.dumps(
                        specialist_payload,
                        indent=2,
                        default=str,
                    )}"
                    "\n\n"
                    "Synthesize the final answer "
                    "for the user using only these "
                    "specialist results for system "
                    "or external-state claims."
                ),
            },
        ]

        response = (
            await self.inference.generate(
                model_key=(
                    self.model_key
                ),
                messages=(
                    messages
                ),
                max_new_tokens=(
                    self.synthesis_max_new_tokens
                ),
                priority=(
                    InferencePriority
                    .PRIMARY_RESPONSE
                ),
            )
        )

        return response.strip()