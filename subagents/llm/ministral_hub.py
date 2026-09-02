import re
from pathlib import Path
from typing import Any

from transformers import (
    FineGrainedFP8Config,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

from subagents.llm.base import LLMBackend


class MinistralHubBackend(LLMBackend):
    """
    Local Ministral 3B backend for the Main Hub.
    """

    def __init__(
        self,
        model_path: str | Path,
        dequantize_fp8: bool = True,
        offload_folder: str | Path | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path)
            .expanduser()
            .resolve()
        )

        self.dequantize_fp8 = dequantize_fp8

        # --------------------------------------------------------
        # Optional disk offload directory
        # --------------------------------------------------------

        self.offload_folder: Path | None = None

        if offload_folder is not None:
            path = (
                Path(offload_folder)
                .expanduser()
            )

            if not path.is_absolute():
                path = Path.cwd() / path

            path = path.resolve()

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.offload_folder = path

        # --------------------------------------------------------
        # Tokenizer
        # --------------------------------------------------------

        self.tokenizer = (
            MistralCommonBackend.from_pretrained(
                str(self.model_path)
            )
        )

        # --------------------------------------------------------
        # FP8 fallback configuration
        # --------------------------------------------------------

        quantization_config = (
            FineGrainedFP8Config(
                dequantize=self.dequantize_fp8
            )
        )

        # --------------------------------------------------------
        # Model loading
        # --------------------------------------------------------

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "quantization_config": (
                quantization_config
            ),
        }

        if self.offload_folder is not None:
            load_kwargs["offload_folder"] = str(
                self.offload_folder
            )

        self.model = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                str(self.model_path),
                **load_kwargs,
            )
        )

        self.model.eval()

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
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

        # Handle opening fence without closing fence.
        if response.startswith("```"):
            newline_index = response.find("\n")

            if newline_index != -1:
                response = response[
                    newline_index + 1 :
                ].strip()

        if response.endswith("```"):
            response = response[:-3].strip()

        return response

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        tokenized = (
            self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            )
        )

        input_device = self.model.device

        for key, value in tokenized.items():
            if hasattr(value, "to"):
                tokenized[key] = value.to(
                    input_device
                )

        input_length = (
            tokenized["input_ids"]
            .shape[-1]
        )

        output = self.model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]

        generated_tokens = output[
            input_length:
        ]

        response = self.tokenizer.decode(
            generated_tokens
        )

        return self._clean_response(
            response
        )
        
    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Normalize Ministral output before application code sees it.

        Handles:
        - Markdown JSON fences
        - trailing model EOS tokens such as </s>
        """

        response = response.strip()

        # Remove known model special tokens from the edges.
        response = re.sub(
            r"^(?:<s>\s*)+",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(
            r"(?:\s*</s>)+$",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = response.strip()

        # Complete Markdown-fenced response.
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

        # Opening fence without closing fence.
        elif response.startswith("```"):
            newline_index = response.find("\n")

            if newline_index != -1:
                response = response[
                    newline_index + 1:
                ].strip()

        if response.endswith("```"):
            response = response[:-3].strip()

        # One final EOS cleanup in case the EOS token was
        # inside/after a Markdown wrapper.
        response = re.sub(
            r"(?:\s*</s>)+$",
            "",
            response,
            flags=re.IGNORECASE,
        )

        return response.strip()