import re
from pathlib import Path

import torch
from transformers import (
    FineGrainedFP8Config,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

from subagents.llm.base import LLMBackend


class MinistralHubBackend(LLMBackend):
    """
    Ministral-3-3B Hub backend.

    Responsibilities:
    - load the local Ministral checkpoint
    - dequantize FP8 weights for GPUs without native FP8 support
    - generate deterministic Hub output
    - normalize Markdown-fenced JSON into plain JSON

    This backend does NOT perform routing validation.
    That remains the responsibility of LLMRouter / planner logic.
    """

    def __init__(
        self,
        model_path: str | Path,
        dequantize_fp8: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.dequantize_fp8 = dequantize_fp8

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Ministral Hub model not found: "
                f"{self.model_path}"
            )

        self.tokenizer = (
            MistralCommonBackend.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        self.model = (
            Mistral3ForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                local_files_only=True,
                quantization_config=FineGrainedFP8Config(
                    dequantize=self.dequantize_fp8,
                ),
            )
        )

        self.model.eval()

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Normalize model output into plain structured text.

        Example:

            ```json
            {"agents": ["account-specialist"]}
            ```

        becomes:

            {"agents": ["account-specialist"]}
        """

        response = response.strip()

        fenced_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_match:
            response = fenced_match.group(1).strip()

        return response

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {
            key: (
                value.to(self.model.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[
            0,
            input_length:,
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return self._clean_response(
            response
        )