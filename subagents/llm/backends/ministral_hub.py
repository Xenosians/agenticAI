import hashlib
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

from subagents.llm.runtime.base import (
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

        if (
            self.quantization
            == "bnb4"
            and self._checkpoint_is_prequantized_bnb4(
                self.model_path
            )
        ):
            resolved_quantization = (
                "bnb4-prequantized"
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

        # Logical-profile adapter state layered on one physical base.
        self._active_profile_adapter: (
            str | None
        ) = None

        self._hub_overlay_applied: bool = (
            False
        )

    def supports_profile_switching(
        self,
    ) -> bool:
        return True

    @staticmethod
    def _profile_adapter_name(
        adapter_path: Path,
    ) -> str:
        digest = hashlib.sha256(
            str(
                adapter_path
                .expanduser()
                .resolve()
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            "profile-"
            + digest[:16]
        )

    def activate_profile_adapter(
        self,
        model_key: str,
        adapter_path: (
            str
            | Path
            | None
        ),
    ) -> None:
        """
        Select a logical model profile on the already-loaded physical base.

        adapter_path=None means base-model inference. If a PEFT wrapper is
        already installed, generation temporarily disables adapters for that
        logical profile instead of reconstructing the multi-gigabyte base.
        """
        if adapter_path is None:
            self._active_profile_adapter = (
                None
            )

            print(
                "[MODEL] Activated shared-base profile "
                f"model='{model_key}' adapter=None"
            )

            return

        adapter_directory = (
            Path(
                adapter_path
            )
            .expanduser()
            .resolve()
        )

        if not adapter_directory.is_dir():
            raise ValueError(
                "Configured model adapter directory does not exist: "
                f"{adapter_directory}"
            )

        from peft import PeftModel

        adapter_name = (
            self._profile_adapter_name(
                adapter_directory
            )
        )

        if not isinstance(
            self.model,
            PeftModel,
        ):
            self.model = (
                PeftModel
                .from_pretrained(
                    self.model,
                    str(
                        adapter_directory
                    ),
                    adapter_name=(
                        adapter_name
                    ),
                    is_trainable=False,
                )
            )

        elif (
            adapter_name
            not in self.model.peft_config
        ):
            try:
                self.model.load_adapter(
                    str(
                        adapter_directory
                    ),
                    adapter_name=(
                        adapter_name
                    ),
                    is_trainable=False,
                    low_cpu_mem_usage=True,
                )

            except TypeError:
                self.model.load_adapter(
                    str(
                        adapter_directory
                    ),
                    adapter_name=(
                        adapter_name
                    ),
                    is_trainable=False,
                )

        try:
            self.model.set_adapter(
                adapter_name,
                inference_mode=True,
            )

        except TypeError:
            self.model.set_adapter(
                adapter_name
            )

        self.model.eval()

        self._active_profile_adapter = (
            adapter_name
        )

        print(
            "[MODEL] Activated shared-base profile "
            f"model='{model_key}' "
            f"adapter='{adapter_name}'"
        )

    def _generate_with_active_profile(
        self,
        **kwargs,
    ):
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = ()

        if (
            isinstance(
                self.model,
                PeftModel,
            )
            and self._active_profile_adapter
            is None
        ):
            with self.model.disable_adapter():
                return (
                    self.model
                    .generate(
                        **kwargs
                    )
                )

        return (
            self.model
            .generate(
                **kwargs
            )
        )

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
    def _checkpoint_is_prequantized_bnb4(
        model_path: Path,
    ) -> bool:
        config_path = (
            model_path
            / "config.json"
        )

        if not config_path.is_file():
            return False

        try:
            config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return False

        quant = config.get(
            "quantization_config"
        )

        if not isinstance(
            quant,
            dict,
        ):
            return False

        if quant.get(
            "load_in_4bit"
        ) is True:
            return True

        if quant.get(
            "_load_in_4bit"
        ) is True:
            return True

        method = quant.get(
            "quant_method"
        )

        if isinstance(
            method,
            str,
        ):
            normalized = (
                method
                .strip()
                .lower()
            )

            return (
                "bitsandbytes"
                in normalized
                and (
                    "4"
                    in normalized
                    or quant.get(
                        "_load_in_4bit"
                    ) is True
                )
            )

        return False

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

        if normalized in {
            "none",
            "bnb4-prequantized",
        }:
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

        print(f"[INFERENCE_INPUT] model={self.model_path.name!r} prompt_tokens={input_length} max_new_tokens={max_new_tokens}", flush=True)

        output = (
            self
            ._generate_with_active_profile(
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
