from __future__ import annotations

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
)

from config import (
    ModelProfileSettings,
)


def resolve_torch_dtype(
    value: str,
    *,
    allow_auto: bool,
):
    """
    Resolve validated configuration into a Transformers/PyTorch
    dtype value.

    This function is model-brand independent.
    """

    normalized = (
        value
        .strip()
        .lower()
    )

    if (
        allow_auto
        and normalized
        == "auto"
    ):

        return "auto"

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
            "Unsupported torch dtype: "
            f"{value}"
        ) from exc


def build_transformers_model_load_kwargs(
    profile: ModelProfileSettings,
) -> dict[
    str,
    Any,
]:
    """
    Translate one validated logical model profile into common
    Transformers model-loading arguments.

    Model storage dtype, quantization, device placement and
    offload location are supplied by configuration.

    local_files_only remains a trusted local-first runtime
    invariant rather than a model-selected setting.
    """

    kwargs: dict[
        str,
        Any,
    ] = {
        "local_files_only":
            True,

        "torch_dtype":
            resolve_torch_dtype(
                profile.model_dtype,
                allow_auto=True,
            ),
    }

    if (
        profile.device_map
        is not None
    ):

        kwargs[
            "device_map"
        ] = (
            profile.device_map
        )

    if (
        profile.offload_folder
        is not None
    ):

        offload_folder = (
            Path(
                profile.offload_folder
            )
            .expanduser()
            .resolve()
        )

        offload_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        kwargs[
            "offload_folder"
        ] = (
            str(
                offload_folder
            )
        )

    quantization = (
        profile.quantization
        .strip()
        .lower()
    )

    if (
        quantization
        in {
            "auto",
            "none",
        }
    ):

        return kwargs

    if (
        quantization
        == "fp8"
    ):

        kwargs[
            "quantization_config"
        ] = (
            FineGrainedFP8Config(
                dequantize=(
                    profile
                    .dequantize_fp8
                )
            )
        )

        return kwargs

    if (
        quantization
        == "bnb4"
    ):

        kwargs[
            "quantization_config"
        ] = (
            BitsAndBytesConfig(
                load_in_4bit=True,

                bnb_4bit_quant_type=(
                    profile
                    .bnb_4bit_quant_type
                ),

                bnb_4bit_compute_dtype=(
                    resolve_torch_dtype(
                        profile.compute_dtype,
                        allow_auto=False,
                    )
                ),

                bnb_4bit_use_double_quant=(
                    profile
                    .bnb_4bit_use_double_quant
                ),
            )
        )

        return kwargs

    raise ValueError(
        "Unsupported model quantization mode: "
        f"{profile.quantization}"
    )
