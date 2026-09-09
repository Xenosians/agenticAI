import json
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
        Normalize harmless model-specific formatting.

        Supported normalization:
        - Markdown JSON fences
        - one valid tool-call object -> one-element array

        The shared parser remains strict. Invalid JSON or unexpected
        structures are intentionally left untouched so validation
        still rejects them later.
        """

        response = response.strip()

        # --------------------------------------------------------
        # Markdown fenced JSON
        # --------------------------------------------------------

        fenced_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            flags=(
                re.DOTALL
                | re.IGNORECASE
            ),
        )

        if fenced_match:
            response = (
                fenced_match
                .group(1)
                .strip()
            )

        elif response.startswith(
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

        # --------------------------------------------------------
        # Qwen Coder singleton tool-call normalization
        #
        # Some Qwen Coder generations return:
        #
        # {
        #   "name": "...",
        #   "arguments": {...}
        # }
        #
        # Our worker protocol requires:
        #
        # [
        #   {
        #     "name": "...",
        #     "arguments": {...}
        #   }
        # ]
        #
        # Only normalize a structurally valid tool-call object.
        # Anything else remains unchanged and will be rejected by
        # the shared parser.
        # --------------------------------------------------------

        try:
            parsed = json.loads(
                response
            )
        except json.JSONDecodeError:
            return response

        if (
            isinstance(parsed, dict)
            and isinstance(
                parsed.get("name"),
                str,
            )
            and isinstance(
                parsed.get("arguments"),
                dict,
            )
        ):
            return json.dumps(
                [parsed],
                separators=(
                    ",",
                    ":",
                ),
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