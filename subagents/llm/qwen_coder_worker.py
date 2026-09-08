import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from subagents.llm.base import (
    LLMBackend,
)


class QwenCoderWorkerBackend(
    LLMBackend
):
    """
    Qwen2.5 Coder specialist-worker backend.

    Intended for governed developer/workspace specialists.

    This backend performs reasoning only.

    It is NOT a security boundary.

    ToolGateway and the process runner remain responsible for
    deciding whether a proposed machine action is allowed.
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

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Developer model not found: "
                f"{self.model_path}"
            )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(self.model_path),
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
        )

        self.model.eval()

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Normalize harmless Markdown JSON fences.

        Worker protocol still expects structured JSON.
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

        if response.startswith(
            "```"
        ):
            newline_index = (
                response.find("\n")
            )

            if newline_index != -1:
                response = response[
                    newline_index + 1:
                ].strip()

        if response.endswith(
            "```"
        ):
            response = (
                response[:-3]
                .strip()
            )

        return response

    def generate(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 128,
    ) -> str:
        inputs = (
            self.tokenizer
            .apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        )

        inputs = inputs.to(
            self.model.device
        )

        with torch.no_grad():
            output = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                    do_sample=False,
                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    ),
                )
            )

        generated_tokens = output[
            0,
            inputs[
                "input_ids"
            ].shape[1]:,
        ]

        response = (
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        )

        return self._clean_response(
            response
        )