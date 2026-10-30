import json

from config import (
    get_settings,
)

from subagents.llm.runtime.stages import InferenceStage

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


# Fields a user-facing answer can draw on. Provenance, routing contracts and UI
# presentation are audit/rendering data that only consume the synthesis budget.
SYNTHESIS_FIELDS = (
    "agent_name",
    "status",
    "proposed_tool",
    "proposed_arguments",
    "tool_result",
    "outcome_code",
    "error",
)


def answer_word_limit(max_new_tokens: int) -> int:
    """Words that fit an output cap: formatted answers cost two to three tokens
    per word, so a quarter of the cap keeps an answer from being cut off."""
    return max(40, max_new_tokens // 4)


def synthesis_evidence(result: AgentResult) -> dict:
    evidence = {
        field: getattr(result, field)
        for field in SYNTHESIS_FIELDS
        if getattr(result, field) is not None
    }
    if result.tool_result is None and result.answer:
        # Without a structured result the specialist's answer is the evidence.
        evidence["answer"] = result.answer
    return evidence


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
        # A provider that reaches the output cap fails the job as truncated,
        # so the model is told the limit, as in synthesis.
        word_limit = answer_word_limit(self.response_max_new_tokens)

        messages = [
            {
                "role":
                    "system",

                "content":
                    self.system_prompt
                    + f"\n\nAnswer in at most {word_limit} words.",
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
                stage=InferenceStage.PRIMARY_RESPONSE,
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
            synthesis_evidence(
                result
            )

            for result in results
        ]

        word_limit = answer_word_limit(self.synthesis_max_new_tokens)

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
                        separators=(",", ":"),
                        default=str,
                    )}"
                    "\n\n"
                    "Synthesize the final answer "
                    "for the user using only these "
                    "specialist results for system "
                    "or external-state claims. "
                    f"Use at most {word_limit} words "
                    "and no tables. For long results "
                    "give counts and at most 10 of "
                    "the most relevant items."
                ),
            },
        ]

        response = (
            await self.inference.generate(
                stage=InferenceStage.SYNTHESIS,
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