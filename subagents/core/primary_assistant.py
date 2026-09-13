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
    General conversational assistant.

    The assistant references a logical model profile.

    It does not own, cache, load, or directly call a model
    backend.
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