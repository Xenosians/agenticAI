"""Fetch and load what a job names, only after the host admitted it.

The host runs every input through its Lua admission before dispatch (docs/lua-design-contract.md) and
lists the receipts in the job; this module refuses anything the job does not list. Weights load only from
safetensors or GGUF (never pickle) and never run repository code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from collections.abc import Callable
from typing import TypeVar

from .job import TrainingJob

T = TypeVar("T")

SAFE_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"]
DATASET_PATTERNS = ["*.jsonl", "*.json", "*.parquet"]
OLLAMA_REGISTRY = "https://registry.ollama.ai"
OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")


class NotAdmitted(PermissionError):
    """An input the host did not admit."""


def require_admissions(job: TrainingJob) -> None:
    admitted = {admission.artifact for admission in job.admissions}
    missing = [key for key in job.input_artifacts() if key not in admitted]
    if missing:
        raise NotAdmitted("the host did not admit: " + ", ".join(missing))


def admitted_load(job: TrainingJob, artifact: str, load: Callable[[], T]) -> T:
    """Load an outside input only when the host admitted exactly that artifact."""
    if artifact not in {admission.artifact for admission in job.admissions}:
        raise NotAdmitted(f"the host did not admit: {artifact}")
    return load()


def own_output(load: Callable[[], T]) -> T:
    """Reload weights this run wrote itself (a stage's adapter); never an outside input."""
    return load()


def _huggingface_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def fetch_model(job: TrainingJob, root: Path, cache: Path) -> tuple[Path, str | None]:
    """The directory holding the base weights, and the GGUF file name when the weights are GGUF."""
    reference = job.model
    if reference.source == "local":
        directory = job.resolve(root, str(reference.path))
        if not directory.is_dir():
            raise FileNotFoundError(f"the job's model directory {reference.path} is missing")
        return directory, reference.gguf_file
    if reference.source == "huggingface":
        from huggingface_hub import snapshot_download

        patterns = [reference.gguf_file, "*.json", "*.txt", "*.model", "*.jinja"] if reference.gguf_file \
            else SAFE_PATTERNS
        directory = Path(snapshot_download(repo_id=reference.name, revision=reference.revision or "main",
                                           allow_patterns=patterns, cache_dir=str(cache / "huggingface"),
                                           token=_huggingface_token()))
        if not reference.gguf_file and not any(directory.glob("*.safetensors")):
            raise ValueError("the model has no safetensors weights; pickle weights are refused")
        return directory, reference.gguf_file
    return fetch_ollama(str(reference.name), cache / "ollama"), "model.gguf"


def _ollama_parts(name: str) -> tuple[str, str, str]:
    repository, _, tag = name.partition(":")
    namespace, _, model = repository.rpartition("/")
    return namespace or "library", model, tag or "latest"


def fetch_ollama(name: str, cache: Path, *, opener=urllib.request.urlopen,
                 registry: str = OLLAMA_REGISTRY) -> Path:
    """Download an Ollama model's GGUF weights from the registry and check them against their digest."""
    namespace, model, tag = _ollama_parts(name)
    directory = Path(cache) / namespace / model / tag
    weights = directory / "model.gguf"
    request = urllib.request.Request(f"{registry}/v2/{namespace}/{model}/manifests/{tag}",
                                     headers={"Accept": MANIFEST_ACCEPT, "User-Agent": "janus-training-worker"})
    with opener(request, timeout=60) as response:
        manifest = json.loads(response.read().decode("utf-8"))
    layers = [layer for layer in manifest.get("layers", []) if layer.get("mediaType") == OLLAMA_MODEL_LAYER]
    if len(layers) != 1:
        raise ValueError(f"the Ollama manifest for {name} has {len(layers)} model layers, expected 1")
    match = DIGEST.match(str(layers[0].get("digest", "")))
    if match is None:
        raise ValueError("the Ollama model layer has no sha256 digest")
    expected = match.group(1)
    if weights.is_file() and (directory / "digest").is_file() \
            and (directory / "digest").read_text().strip() == expected:
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / "model.gguf.part"
    digest = hashlib.sha256()
    blob = urllib.request.Request(f"{registry}/v2/{namespace}/{model}/blobs/sha256:{expected}",
                                  headers={"User-Agent": "janus-training-worker"})
    with opener(blob, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(f"the downloaded Ollama weights for {name} do not match their digest")
    os.replace(partial, weights)
    (directory / "digest").write_text(expected + "\n")
    return directory


def fetch_dataset(job: TrainingJob, root: Path, cache: Path) -> Path:
    reference = job.dataset
    if reference.source == "local":
        directory = job.resolve(root, str(reference.path))
        if not directory.is_dir():
            raise FileNotFoundError(f"the job's dataset directory {reference.path} is missing")
        return directory
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=reference.name, revision=reference.revision or "main",
                                  repo_type="dataset", allow_patterns=DATASET_PATTERNS,
                                  cache_dir=str(cache / "huggingface"), token=_huggingface_token()))


def load_tokenizer(job: TrainingJob, directory: Path, gguf_file: str | None):
    from transformers import AutoTokenizer

    options: dict = {"trust_remote_code": False}
    if gguf_file:
        options["gguf_file"] = gguf_file
    tokenizer = admitted_load(job, job.model.key(),
                              lambda: AutoTokenizer.from_pretrained(str(directory), **options))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("the tokenizer has neither a pad nor an end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(job: TrainingJob, directory: Path, gguf_file: str | None, *, device: str, dtype_name: str,
              quantize_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    options: dict = {"trust_remote_code": False, "dtype": dtype}
    if gguf_file:
        options["gguf_file"] = gguf_file
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
        options["device_map"] = {"": 0}
    model = admitted_load(job, job.model.key(),
                          lambda: AutoModelForCausalLM.from_pretrained(str(directory), **options))
    if not quantize_4bit:
        model.to(device)
    model.config.use_cache = False
    return model


def weights_bytes(directory: Path, gguf_file: str | None) -> int:
    if gguf_file:
        return (Path(directory) / gguf_file).stat().st_size
    return sum(path.stat().st_size for path in Path(directory).glob("*.safetensors"))
