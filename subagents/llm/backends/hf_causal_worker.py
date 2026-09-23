from __future__ import annotations

import re

from pathlib import Path

from typing import (
    Any,
)

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from subagents.llm.runtime.base import (
    LLMBackend,
)


class HFCausalWorkerBackend(
    LLMBackend
):
    """
    Generic deterministic local Hugging Face causal worker.

    Intended for small specialist models whose job is to propose
    exactly one structured capability call.

    This backend is NOT an authorization boundary.

    SemanticGuard and ToolGateway remain authoritative.

    The backend deliberately does not enable trust_remote_code.

    Models with a tokenizer chat template use that template.
    Models without one receive a deterministic role-labelled prompt.
    """

    def __init__(
        self,
        model_path: str | Path,
        model_load_kwargs: dict[
            str,
            Any,
        ],
    ) -> None:

        self.model_path = (
            Path(
                model_path
            )
            .expanduser()
            .resolve()
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model not found: "
                f"{self.model_path}"
            )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(
                    self.model_path
                ),
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(
                    self.model_path
                ),
                **model_load_kwargs,
            )
        )

        self.model.eval()

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Remove only harmless Markdown JSON fencing.

        Do not repair, infer, or rewrite semantic content.
        """

        response = (
            response.strip()
        )

        fenced = (
            re.fullmatch(
                (
                    r"```(?:json)?\s*"
                    r"(.*?)"
                    r"\s*```"
                ),
                response,
                flags=(
                    re.DOTALL
                    | re.IGNORECASE
                ),
            )
        )

        if fenced:
            return (
                fenced
                .group(1)
                .strip()
            )

        return response

    def _render_prompt(
        self,
        messages: list[
            dict[
                str,
                str,
            ]
        ],
    ) -> str:

        chat_template = (
            getattr(
                self.tokenizer,
                "chat_template",
                None,
            )
        )

        if (
            isinstance(
                chat_template,
                str,
            )
            and chat_template.strip()
        ):

            return (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        parts: list[str] = []

        role_labels = {
            "system":
                "SYSTEM",

            "user":
                "USER",

            "assistant":
                "ASSISTANT",
        }

        for message in messages:

            role = (
                message.get(
                    "role",
                    "user",
                )
            )

            content = (
                message.get(
                    "content",
                    ""
                )
            )

            label = (
                role_labels.get(
                    role,
                    role.upper(),
                )
            )

            parts.append(
                (
                    f"{label}:\n"
                    f"{content.strip()}"
                )
            )

        parts.append(
            "ASSISTANT:\n"
        )

        return (
            "\n\n".join(
                parts
            )
        )

    def generate(
        self,
        messages: list[
            dict[
                str,
                str,
            ]
        ],
        max_new_tokens: int = 256,
    ) -> str:

        prompt = (
            self._render_prompt(
                messages
            )
        )

        model_inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt",
            )
        )

        model_device = (
            next(
                self.model.parameters()
            )
            .device
        )

        model_inputs = {
            key:
                value.to(
                    model_device
                )

            for (
                key,
                value,
            )
            in model_inputs.items()
        }

        input_length = (
            model_inputs[
                "input_ids"
            ]
            .shape[-1]
        )

        pad_token_id = (
            self.tokenizer
            .pad_token_id
        )

        if pad_token_id is None:
            pad_token_id = (
                self.tokenizer
                .eos_token_id
            )

        with torch.no_grad():

            generated = (
                self.model.generate(
                    **model_inputs,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                    do_sample=False,
                    pad_token_id=(
                        pad_token_id
                    ),
                )
            )

        generated_tokens = (
            generated[
                0,
                input_length:
            ]
        )

        response = (
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        )

        return (
            self._clean_response(
                response
            )
        )
