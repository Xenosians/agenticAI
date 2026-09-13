import json

from dataclasses import (
    asdict,
)

from subagents.core.types import (
    AgentResult,
)

from subagents.llm.inference import (
    InferenceEngine,
)

from subagents.llm.scheduler import (
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
    ) -> None:
        self.inference = (
            inference
        )

        self.model_key = (
            model_key
        )

        self.system_prompt = (
            load_prompt(
                "primary_assistant.txt"
            )
        )

    async def respond(
        self,
        user_request: str,
    ) -> str:
        messages = [
            {
                "role":
                    "system",

                "content":
                    self.system_prompt,
            },

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
                max_new_tokens=384,
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
                max_new_tokens=512,
                priority=(
                    InferencePriority
                    .PRIMARY_RESPONSE
                ),
            )
        )

        return response.strip()