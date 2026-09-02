import re
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from subagents.llm.base import LLMBackend


class Qwen3WorkerBackend(LLMBackend):
    """
    Qwen3 specialist-worker backend.

    Current role:
        Access Specialist

    Responsibilities:
    - local Qwen3 inference
    - disable Qwen3 thinking mode
    - deterministic generation
    - normalize harmless Markdown JSON fences

    This backend is NOT a security boundary.
    ToolGateway remains responsible for deterministic policy.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self.model_path = (
            Path(model_path)
            .expanduser()
            .resolve()
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        )

        self.model.eval()

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Normalize JSON wrapped in Markdown fences.

        Examples:

        ```json
        [...]
        ```

        becomes:

        [...]

        Also handles an opening fence without a closing one.
        """

        response = response.strip()

        fenced_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            flags=(
                re.DOTALL
                | re.IGNORECASE
            ),
        )

        if fenced_match:
            return (
                fenced_match
                .group(1)
                .strip()
            )

        if response.startswith("```"):
            newline_index = response.find("\n")

            if newline_index != -1:
                response = response[
                    newline_index + 1:
                ].strip()

        if response.endswith("```"):
            response = response[:-3].strip()

        return response

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate one deterministic worker response.
        """

        prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(self.model.device)
            for key, value
            in model_inputs.items()
        }

        input_length = (
            model_inputs["input_ids"]
            .shape[-1]
        )

        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated_tokens = generated[
            0,
            input_length:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return self._clean_response(
            response
        )