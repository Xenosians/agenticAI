import json
import re

from pathlib import (
    Path,
)

from typing import (
    Any,
)

import torch

from transformers import (
    BitsAndBytesConfig,
    FineGrainedFP8Config,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

from subagents.llm.base import (
    GenerationOutput,
    LLMBackend,
)


class MinistralHubBackend(
    LLMBackend
):
    """
    Local Ministral backend.

    Runtime representation is deployment-configurable.

    Supported quantization modes:

        auto
            Detect checkpoint-native FP8 where present.
            Otherwise load the checkpoint normally.

        none
            No additional quantization.

        fp8
            Fine-grained FP8 through Transformers.

        bnb4
            bitsandbytes 4-bit quantization, intended for
            constrained consumer GPUs when appropriate.

    Quantization selection is not an orchestration concern.
    """

    def __init__(
        self,
        model_path: (
            str | Path
        ),

        quantization: str = (
            "auto"
        ),

        compute_dtype: str = (
            "bfloat16"
        ),

        device_map: (
            str | None
        ) = (
            "auto"
        ),

        dequantize_fp8: bool = (
            True
        ),

        bnb_4bit_quant_type: str = (
            "nf4"
        ),

        bnb_4bit_use_double_quant: bool = (
            True
        ),

        offload_folder: (
            str
            | Path
            | None
        ) = None,
    ) -> None:
        self.model_path = (
            Path(
                model_path
            )
            .expanduser()
            .resolve()
        )

        self.quantization = (
            quantization
            .strip()
            .lower()
        )

        self.compute_dtype = (
            compute_dtype
            .strip()
            .lower()
        )

        self.device_map = (
            device_map
        )

        self.dequantize_fp8 = (
            dequantize_fp8
        )

        self.bnb_4bit_quant_type = (
            bnb_4bit_quant_type
            .strip()
            .lower()
        )

        self.bnb_4bit_use_double_quant = (
            bnb_4bit_use_double_quant
        )

        self.offload_folder: (
            Path | None
        ) = None

        if (
            offload_folder
            is not None
        ):
            path = (
                Path(
                    offload_folder
                )
                .expanduser()
            )

            if not path.is_absolute():
                path = (
                    Path.cwd()
                    / path
                )

            path = (
                path.resolve()
            )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.offload_folder = (
                path
            )

        # --------------------------------------------------------
        # Tokenizer
        # --------------------------------------------------------

        self.tokenizer = (
            MistralCommonBackend
            .from_pretrained(
                str(
                    self.model_path
                )
            )
        )

        # --------------------------------------------------------
        # Runtime quantization
        # --------------------------------------------------------

        resolved_quantization = (
            self._resolve_quantization_mode(
                self.model_path,
                self.quantization,
            )
        )

        print(
            "[MODEL] Ministral runtime "
            f"quantization='{resolved_quantization}' "
            f"compute_dtype='{self.compute_dtype}' "
            f"device_map={self.device_map!r}"
        )

        quantization_config = (
            self._build_quantization_config(
                mode=(
                    resolved_quantization
                ),

                compute_dtype=(
                    self.compute_dtype
                ),

                dequantize_fp8=(
                    self.dequantize_fp8
                ),

                bnb_4bit_quant_type=(
                    self.bnb_4bit_quant_type
                ),

                bnb_4bit_use_double_quant=(
                    self.bnb_4bit_use_double_quant
                ),
            )
        )

        load_kwargs: dict[
            str,
            Any,
        ] = {
            "local_files_only":
                True,
        }

        if (
            self.device_map
            is not None
        ):
            load_kwargs[
                "device_map"
            ] = (
                self.device_map
            )

        if (
            quantization_config
            is not None
        ):
            load_kwargs[
                "quantization_config"
            ] = (
                quantization_config
            )

        if (
            self.offload_folder
            is not None
        ):
            load_kwargs[
                "offload_folder"
            ] = str(
                self.offload_folder
            )

        self.model = (
            Mistral3ForConditionalGeneration
            .from_pretrained(
                str(
                    self.model_path
                ),
                **load_kwargs,
            )
        )

        self.model.eval()

    @staticmethod
    def _resolve_torch_dtype(
        value: str,
    ):
        normalized = (
            value
            .strip()
            .lower()
        )

        mapping = {
            "bfloat16":
                torch.bfloat16,

            "float16":
                torch.float16,

            "float32":
                torch.float32,
        }

        try:
            return (
                mapping[
                    normalized
                ]
            )

        except KeyError as exc:
            raise ValueError(
                "Unsupported compute dtype: "
                f"{value}"
            ) from exc

    @staticmethod
    def _checkpoint_quantization(
        model_path: Path,
    ) -> (
        str | None
    ):
        config_path = (
            model_path
            / "config.json"
        )

        if not (
            config_path.exists()
        ):
            return None

        try:
            with config_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                config = (
                    json.load(
                        handle
                    )
                )

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        quantization_config = (
            config.get(
                "quantization_config"
            )
        )

        if not isinstance(
            quantization_config,
            dict,
        ):
            return None

        quant_method = (
            quantization_config
            .get(
                "quant_method"
            )
        )

        if not isinstance(
            quant_method,
            str,
        ):
            return None

        return (
            quant_method
            .strip()
            .lower()
        )

    @classmethod
    def _resolve_quantization_mode(
        cls,
        model_path: Path,
        requested: str,
    ) -> str:
        normalized = (
            requested
            .strip()
            .lower()
        )

        if normalized != "auto":
            return normalized

        checkpoint_mode = (
            cls._checkpoint_quantization(
                model_path
            )
        )

        if (
            checkpoint_mode
            == "fp8"
        ):
            return "fp8"

        return "none"

    @classmethod
    def _build_quantization_config(
        cls,
        *,
        mode: str,
        compute_dtype: str,
        dequantize_fp8: bool,
        bnb_4bit_quant_type: str,
        bnb_4bit_use_double_quant: bool,
    ):
        normalized = (
            mode
            .strip()
            .lower()
        )

        if normalized == "none":
            return None

        if normalized == "fp8":
            return (
                FineGrainedFP8Config(
                    dequantize=(
                        dequantize_fp8
                    )
                )
            )

        if normalized == "bnb4":
            return (
                BitsAndBytesConfig(
                    load_in_4bit=True,

                    bnb_4bit_quant_type=(
                        bnb_4bit_quant_type
                    ),

                    bnb_4bit_compute_dtype=(
                        cls._resolve_torch_dtype(
                            compute_dtype
                        )
                    ),

                    bnb_4bit_use_double_quant=(
                        bnb_4bit_use_double_quant
                    ),
                )
            )

        raise ValueError(
            "Unsupported Ministral "
            "quantization mode: "
            f"{mode}"
        )

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        response = (
            response.strip()
        )

        response = re.sub(
            r"^(?:<s>\s*)+",
            "",
            response,
            flags=(
                re.IGNORECASE
            ),
        )

        response = re.sub(
            r"(?:\s*</s>)+$",
            "",
            response,
            flags=(
                re.IGNORECASE
            ),
        )

        response = (
            response.strip()
        )

        fenced_match = (
            re.fullmatch(
                r"```(?:json)?\s*(.*?)\s*```",
                response,
                flags=(
                    re.DOTALL
                    | re.IGNORECASE
                ),
            )
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
                response.find(
                    "\n"
                )
            )

            if (
                newline_index
                != -1
            ):
                response = (
                    response[
                        newline_index + 1:
                    ]
                    .strip()
                )

        if response.endswith(
            "```"
        ):
            response = (
                response[
                    :-3
                ]
                .strip()
            )

        response = re.sub(
            r"(?:\s*</s>)+$",
            "",
            response,
            flags=(
                re.IGNORECASE
            ),
        )

        return (
            response.strip()
        )

    def _generate_output(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int,
    ) -> GenerationOutput:
        tokenized = (
            self.tokenizer
            .apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            )
        )

        input_device = (
            self.model.device
        )

        for (
            key,
            value,
        ) in (
            tokenized.items()
        ):
            if hasattr(
                value,
                "to",
            ):
                tokenized[
                    key
                ] = value.to(
                    input_device
                )

        input_length = (
            tokenized[
                "input_ids"
            ]
            .shape[
                -1
            ]
        )

        output = (
            self.model
            .generate(
                **tokenized,

                max_new_tokens=(
                    max_new_tokens
                ),

                do_sample=False,
            )[
                0
            ]
        )

        generated_tokens = (
            output[
                input_length:
            ]
        )

        generated_token_count = int(
            generated_tokens
            .shape[
                -1
            ]
        )

        response = (
            self.tokenizer
            .decode(
                generated_tokens
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

                first_token_seconds=None,
            )
        )

    def generate_observed(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 256,
    ) -> GenerationOutput:
        return (
            self._generate_output(
                messages,
                max_new_tokens,
            )
        )

    def generate(
        self,
        messages: list[
            dict[str, str]
        ],
        max_new_tokens: int = 256,
    ) -> str:
        return (
            self._generate_output(
                messages,
                max_new_tokens,
            )
            .text
        )