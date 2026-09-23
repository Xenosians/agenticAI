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
    GenerationOutput,
    LLMBackend,
)

from subagents.llm.runtime.hf_prompt import (
    render_hf_causal_fallback_prompt,
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

    Lifecycle and observability intentionally match the common model
    runtime contract so BLOOMZ and future HF causal specialists do
    not need model-specific loader scripts.
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

        self._closed = False

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

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def _require_open(
        self,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "HFCausalWorkerBackend is closed."
            )

    def close(
        self,
    ) -> None:
        """
        Drop backend-owned references.

        ModelRegistry performs process-wide garbage collection /
        accelerator-cache release after this hook returns.
        """

        if self._closed:
            return

        self._closed = True

        if hasattr(
            self,
            "model",
        ):
            del self.model

        if hasattr(
            self,
            "tokenizer",
        ):
            del self.tokenizer

    # ========================================================
    # RESPONSE NORMALIZATION
    # ========================================================

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

    # ========================================================
    # PROMPT RENDERING
    # ========================================================

    def _render_prompt(
        self,
        messages: list[
            dict[
                str,
                str,
            ]
        ],
    ) -> str:

        self._require_open()

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

        return (
            render_hf_causal_fallback_prompt(
                messages
            )
        )

    # ========================================================
    # INPUT DEVICE
    # ========================================================

    def _input_device(
        self,
    ) -> torch.device:
        """
        Resolve the device that owns the model input embeddings.

        This is safer than assuming the first model parameter owns
        the input device when Transformers/Accelerate has applied an
        automatic device map or offload plan.
        """

        self._require_open()

        try:
            embeddings = (
                self.model
                .get_input_embeddings()
            )

            weight = getattr(
                embeddings,
                "weight",
                None,
            )

            device = getattr(
                weight,
                "device",
                None,
            )

            if (
                isinstance(
                    device,
                    torch.device,
                )
                and device.type
                != "meta"
            ):
                return device

        except Exception:
            pass

        device = (
            next(
                self.model.parameters()
            )
            .device
        )

        if device.type == "meta":
            raise RuntimeError(
                "Unable to resolve a concrete model input device."
            )

        return device

    # ========================================================
    # GENERATION
    # ========================================================

    def _generate_observed(
        self,
        messages: list[
            dict[
                str,
                str,
            ]
        ],
        max_new_tokens: int,
    ) -> GenerationOutput:

        self._require_open()

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

        input_device = (
            self._input_device()
        )

        model_inputs = {
            key:
                value.to(
                    input_device
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

        with torch.inference_mode():

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

        generated_token_ids = (
            generated[
                0,
                input_length:
            ]
        )

        generated_token_count = int(
            generated_token_ids.numel()
        )

        response = (
            self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
        )

        return (
            GenerationOutput(
                text=(
                    self._clean_response(
                        response
                    )
                ),

                generated_tokens=(
                    generated_token_count
                ),

                # Non-streaming Transformers generation cannot
                # provide true TTFT. Keep the metric explicitly
                # unavailable instead of inventing a value.
                first_token_seconds=None,
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
        return (
            self._generate_observed(
                messages,
                max_new_tokens,
            )
            .text
        )

    def generate_observed(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 256,
    ) -> GenerationOutput:
        return (
            self._generate_observed(
                messages,
                max_new_tokens,
            )
        )
