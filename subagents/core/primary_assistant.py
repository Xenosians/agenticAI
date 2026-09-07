from subagents.llm.base import LLMBackend
from subagents.prompts.prompt_loader import load_prompt


class PrimaryAssistant:
    """
    General conversational assistant.

    This assistant handles requests that do not require
    delegation to a specialist.

    It shares the Hub model backend. No additional model
    weights are loaded.
    """

    def __init__(
        self,
        backend: LLMBackend,
    ) -> None:
        self.backend = backend

        self.system_prompt = load_prompt(
            "primary_assistant.txt"
        )

    def respond(
        self,
        user_request: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_request,
            },
        ]

        response = self.backend.generate(
            messages,
            max_new_tokens=384,
        )

        return response.strip()