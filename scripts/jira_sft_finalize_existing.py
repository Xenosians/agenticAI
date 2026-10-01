from __future__ import annotations

import argparse
import ctypes
import gc
import json
import shutil
import sys

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config import Settings  # noqa: E402

from learning.training.jira_sft import (  # noqa: E402
    JiraSftTrainingManifest,
    assert_training_versions,
    fingerprint_directory,
    verify_jira_sft_corpus,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/c/project/agenticaiPersonal/Models/"
    "BLOOMZ-560M-Jira-SFT"
)

DEFAULT_CORPUS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "jira-sft"
    / "v1"
)


def utc_now() -> str:
    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def trim_memory() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass

    try:
        libc = ctypes.CDLL(
            "libc.so.6"
        )

        malloc_trim = getattr(
            libc,
            "malloc_trim",
            None,
        )

        if malloc_trim is not None:
            malloc_trim(0)

    except Exception:
        pass

    gc.collect()


def latest_run(
    output_root: Path,
) -> Path:
    runs = (
        output_root
        .expanduser()
        .resolve()
        / "runs"
    )

    if not runs.is_dir():
        raise ValueError(
            f"Runs directory does not exist: {runs}"
        )

    candidates = [
        path
        for path in runs.iterdir()
        if path.is_dir()
    ]

    if not candidates:
        raise ValueError(
            f"No Jira SFT runs exist beneath {runs}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a Jira SFT run whose LoRA adapter was saved "
            "but standalone checkpoint merge/serialization failed."
        )
    )

    parser.add_argument(
        "--run-directory",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--max-shard-size",
        default="256MB",
    )

    args = parser.parse_args()

    run_directory = (
        args.run_directory
        .expanduser()
        .resolve()

        if args.run_directory is not None

        else latest_run(
            args.output_root
        )
    )

    adapter_directory = (
        run_directory
        / "adapter"
    )

    merged_directory = (
        run_directory
        / "merged"
    )

    adapter_config_path = (
        adapter_directory
        / "adapter_config.json"
    )

    adapter_weights_path = (
        adapter_directory
        / "adapter_model.safetensors"
    )

    for required in [
        adapter_config_path,
        adapter_weights_path,
    ]:
        if not required.is_file():
            raise ValueError(
                f"Required trained adapter artifact missing: {required}"
            )

    # Validate the safetensors container before doing any large load.
    from safetensors import safe_open

    with safe_open(
        str(adapter_weights_path),
        framework="pt",
        device="cpu",
    ) as handle:
        adapter_tensor_keys = list(
            handle.keys()
        )

    if not adapter_tensor_keys:
        raise ValueError(
            "Adapter safetensors contains no tensors."
        )

    adapter_config = json.loads(
        adapter_config_path.read_text(
            encoding="utf-8"
        )
    )

    lora_r = int(
        adapter_config[
            "r"
        ]
    )

    lora_alpha = int(
        adapter_config[
            "lora_alpha"
        ]
    )

    versions = (
        assert_training_versions()
    )

    (
        corpus_manifest,
        _train,
        _validation,
        _system_prompt,
    ) = verify_jira_sft_corpus(
        project_root=PROJECT_ROOT,
        corpus_directory=args.corpus,
    )

    settings = Settings()

    profile = settings.require_model_profile(
        corpus_manifest.base_model_key
    )

    if profile.model_path is None:
        raise ValueError(
            "Base Jira profile has no model_path."
        )

    base_model_path = (
        profile.model_path
        .expanduser()
        .resolve()
    )

    if not base_model_path.is_dir():
        raise ValueError(
            f"Base model directory does not exist: {base_model_path}"
        )

    adapter_sha256 = (
        fingerprint_directory(
            adapter_directory
        )
    )

    print(
        "JIRA SFT EXISTING-RUN FINALIZER"
    )
    print(
        "==============================="
    )
    print(
        "run:",
        run_directory,
    )
    print(
        "base:",
        base_model_path,
    )
    print(
        "adapter:",
        adapter_directory,
    )
    print(
        "adapter tensors:",
        len(
            adapter_tensor_keys
        ),
    )
    print(
        "adapter sha256:",
        adapter_sha256,
    )
    print(
        "LoRA r:",
        lora_r,
    )
    print(
        "LoRA alpha:",
        lora_alpha,
    )
    print(
        "output shard target:",
        args.max_shard_size,
    )

    trim_memory()

    import torch

    from peft import (
        PeftModel,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    # The adapter tokenizer is the tokenizer actually persisted by
    # the completed training run.
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                adapter_directory
            ),
            local_files_only=True,
        )
    )

    print()
    print(
        "Loading base checkpoint on CPU..."
    )

    merge_base = (
        AutoModelForCausalLM
        .from_pretrained(
            str(
                base_model_path
            ),
            local_files_only=True,
            dtype=torch.float16,
            device_map={
                "": "cpu",
            },
            low_cpu_mem_usage=True,
        )
    )

    print(
        "Attaching trained adapter..."
    )

    merge_model = (
        PeftModel
        .from_pretrained(
            merge_base,
            str(
                adapter_directory
            ),
            is_trainable=True,
        )
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in merge_model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in merge_model.parameters()
    )

    trainable_ratio = (
        trainable_parameters
        / total_parameters

        if total_parameters

        else 0.0
    )

    print(
        "trainable parameters:",
        trainable_parameters,
    )
    print(
        "total parameters:",
        total_parameters,
    )
    print(
        "trainable ratio:",
        trainable_ratio,
    )

    print(
        "Merging adapter into base..."
    )

    merged = (
        merge_model
        .merge_and_unload(
            safe_merge=True
        )
    )

    if hasattr(
        merged,
        "config",
    ):
        merged.config.use_cache = True

    # Drop aliases around the merged model before serialization.
    del merge_model
    del merge_base

    trim_memory()

    if merged_directory.exists():
        shutil.rmtree(
            merged_directory
        )

    print()
    print(
        "Saving merged checkpoint using safetensors..."
    )

    serialization_format = (
        "safetensors"
    )

    try:
        merged.save_pretrained(
            merged_directory,
            safe_serialization=True,
            max_shard_size=(
                args.max_shard_size
            ),
        )

    except Exception as exc:
        # safetensors can need a sizeable contiguous userspace
        # allocation while serializing. The model itself remains
        # valid, so retry with PyTorch sharded serialization rather
        # than repeating training.
        print(
            "Safetensors serialization failed:",
            repr(
                exc
            ),
        )
        print(
            "Retrying as sharded PyTorch checkpoint..."
        )

        if merged_directory.exists():
            shutil.rmtree(
                merged_directory
            )

        trim_memory()

        serialization_format = (
            "pytorch"
        )

        merged.save_pretrained(
            merged_directory,
            safe_serialization=False,
            max_shard_size=(
                args.max_shard_size
            ),
        )

    tokenizer.save_pretrained(
        merged_directory
    )

    del merged
    del tokenizer

    trim_memory()

    merged_sha256 = (
        fingerprint_directory(
            merged_directory
        )
    )

    manifest = (
        JiraSftTrainingManifest(
            created_at=utc_now(),
            base_model_key=(
                corpus_manifest
                .base_model_key
            ),
            base_model_path=str(
                base_model_path
            ),
            output_directory=str(
                run_directory
            ),
            adapter_directory=str(
                adapter_directory
            ),
            merged_model_directory=str(
                merged_directory
            ),
            corpus_directory=str(
                args.corpus
                .expanduser()
                .resolve()
            ),
            corpus_train_sha256=(
                corpus_manifest
                .train_sha256
            ),
            corpus_validation_sha256=(
                corpus_manifest
                .validation_sha256
            ),
            max_steps=(
                args.max_steps
            ),
            learning_rate=(
                args.learning_rate
            ),
            max_length=(
                args.max_length
            ),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            trainable_parameters=(
                trainable_parameters
            ),
            total_parameters=(
                total_parameters
            ),
            trainable_ratio=(
                trainable_ratio
            ),
            adapter_sha256=(
                adapter_sha256
            ),
            merged_model_sha256=(
                merged_sha256
            ),
            training_versions=(
                versions
            ),
        )
    )

    manifest_path = (
        run_directory
        / "training-manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_pointer = (
        output_root
        / "latest-training.json"
    )

    latest_pointer.write_text(
        json.dumps(
            {
                "schema":
                    "jira-specialist-sft-latest.v1",

                "training_manifest":
                    str(
                        manifest_path
                    ),

                "merged_model_directory":
                    str(
                        merged_directory
                    ),

                "merged_model_sha256":
                    merged_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    recovery_path = (
        run_directory
        / "recovery.json"
    )

    recovery_path.write_text(
        json.dumps(
            {
                "schema":
                    "jira-specialist-sft-recovery.v1",

                "created_at":
                    utc_now(),

                "reason":
                    (
                        "optimizer completed but original merged "
                        "checkpoint serialization exhausted host memory"
                    ),

                "optimizer_rerun":
                    False,

                "serialization":
                    serialization_format,

                "max_shard_size":
                    args.max_shard_size,

                "adapter_sha256":
                    adapter_sha256,

                "merged_model_sha256":
                    merged_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "JIRA SFT FINALIZATION: PASS"
    )
    print(
        "optimizer rerun: False"
    )
    print(
        "serialization:",
        serialization_format,
    )
    print(
        "merged:",
        merged_directory,
    )
    print(
        "merged sha256:",
        merged_sha256,
    )
    print(
        "training manifest:",
        manifest_path,
    )
    print(
        "latest pointer:",
        latest_pointer,
    )
    print(
        "recovery record:",
        recovery_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
