"""The job file every training worker runs: one schema for every specialist, the hub and every stage.

A job directory holds ``job.json`` plus whatever it references (dataset files, adapters to merge,
held-out cases). Every path in a job is relative to that directory and may not leave it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STAGE_ORDER = ("merge", "sft", "dpo", "ppo")
HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
OLLAMA = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}(/[a-z0-9][a-z0-9_.-]{0,63})?(:[A-Za-z0-9][A-Za-z0-9_.-]{0,63})?$")
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}-\d{8}T\d{6}Z$")


def _inside(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if not value or path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", value):
        raise ValueError("paths must stay inside the job directory")
    return str(path)


def _relative(value: str | None) -> str | None:
    return None if value is None else _inside(value)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRef(Strict):
    source: Literal["huggingface", "ollama", "local"]
    name: str | None = None
    revision: str | None = None
    path: str | None = None
    gguf_file: str | None = None

    @field_validator("path")
    @classmethod
    def _inside(cls, value):
        return _relative(value)

    def key(self) -> str:
        """The artifact name the host's Lua admission approves before the job is dispatched."""
        if self.source == "local":
            return f"local:{self.path}"
        if self.source == "huggingface":
            return f"huggingface:{self.name}@{self.revision or 'main'}"
        return f"ollama:{self.name}"

    @model_validator(mode="after")
    def _reference(self):
        if self.source == "local":
            if not self.path or self.name:
                raise ValueError("a local model names a path inside the job")
        elif self.source == "huggingface":
            if not self.name or not HF_REPO.match(self.name) or ".." in self.name:
                raise ValueError("a Hugging Face model is owner/name")
        elif not self.name or not OLLAMA.match(self.name):
            raise ValueError("an Ollama model is name or name:tag")
        if self.revision is not None and (not REVISION.match(self.revision) or ".." in self.revision):
            raise ValueError("a revision is a branch, tag or commit")
        if self.gguf_file is not None and not FILE_NAME.match(self.gguf_file):
            raise ValueError("gguf_file is a file name")
        return self


class DatasetRef(Strict):
    source: Literal["huggingface", "local"]
    name: str | None = None
    revision: str | None = None
    path: str | None = None
    train: str = "train.jsonl"
    validation: str | None = "validation.jsonl"
    preferences: str | None = "preferences.jsonl"

    @field_validator("path", "train", "validation", "preferences")
    @classmethod
    def _inside(cls, value):
        return _relative(value)

    def key(self) -> str:
        if self.source == "local":
            return f"local:{self.path}"
        return f"huggingface-dataset:{self.name}@{self.revision or 'main'}"

    @model_validator(mode="after")
    def _reference(self):
        if self.source == "local" and (not self.path or self.name):
            raise ValueError("a local dataset names a directory inside the job")
        if self.source == "huggingface" and (not self.name or not HF_REPO.match(self.name)):
            raise ValueError("a Hugging Face dataset is owner/name")
        if self.revision is not None and not REVISION.match(self.revision):
            raise ValueError("a revision is a branch, tag or commit")
        return self


class MergeSettings(Strict):
    """Weight-space start: combine existing LoRA adapters of the same base before training.

    The ``*_svd`` methods and ``cat`` combine the full weight deltas (B @ A); the others combine the A and
    B factors separately, which only approximates a delta merge. ``ties_svd`` is the default."""
    adapters: list[str] | None = None          # None: every adapter under the job's adapters/ directory
    method: Literal["ties_svd", "dare_ties_svd", "dare_linear_svd", "magnitude_prune_svd", "svd", "cat",
                    "ties", "dare_ties", "dare_linear", "linear", "magnitude_prune"] = "ties_svd"
    density: float = Field(default=0.5, gt=0, le=1)
    weights: list[float] | None = None

    @field_validator("adapters")
    @classmethod
    def _adapters(cls, value):
        if value is None:
            return None
        if not value:
            raise ValueError("merge needs at least one adapters entry (omit it to use the adapters/ directory)")
        return [_relative(item) for item in value]

    @model_validator(mode="after")
    def _weights(self):
        if self.weights is not None and self.adapters is not None and len(self.weights) != len(self.adapters):
            raise ValueError("one weight per adapter")
        return self


class SFTSettings(Strict):
    tuning: Literal["lora", "qlora"] = "lora"      # qlora needs CUDA; a CPU worker trains lora and says so
    epochs: float = Field(default=3, gt=0, le=50)
    learning_rate: float = Field(default=2e-4, gt=0, lt=1)
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    max_length: int = Field(default=1024, ge=64, le=32768)
    batch_size: int = Field(default=1, ge=1, le=64)
    gradient_accumulation: int = Field(default=8, ge=1, le=256)
    evaluations_per_epoch: int = Field(default=4, ge=1, le=100)


class DPOSettings(Strict):
    beta: float = Field(default=0.1, gt=0, le=10)
    epochs: float = Field(default=1, gt=0, le=20)
    learning_rate: float = Field(default=5e-6, gt=0, lt=1)
    loss_type: Literal["sigmoid", "ipo", "hinge"] = "sigmoid"
    max_length: int = Field(default=1024, ge=64, le=32768)
    gradient_accumulation: int = Field(default=8, ge=1, le=256)
    self_generated_pairs: bool = True   # the model's own wrong answers become the rejected side
    pair_limit: int = Field(default=64, ge=1, le=10000)


class PPOSettings(Strict):
    """PPO-family clipped policy optimisation (group-relative baseline, no value model) against the
    job's verifiable tool-call reward."""
    epochs: float = Field(default=1, gt=0, le=20)
    learning_rate: float = Field(default=1e-6, gt=0, lt=1)
    samples_per_prompt: int = Field(default=4, ge=2, le=32)
    clip_range: float = Field(default=0.2, gt=0, lt=1)
    kl_beta: float = Field(default=0.04, ge=0, le=1)
    max_new_tokens: int = Field(default=256, ge=8, le=4096)


class GuardSettings(Strict):
    patience: int = Field(default=2, ge=1, le=20)
    min_improvement: float = Field(default=0.01, ge=0, le=1)
    underfit_loss: float | None = Field(default=None, gt=0)
    validation_fraction: float = Field(default=0.1, gt=0, lt=0.5)


class Limits(Strict):
    """Breathing room when the worker shares a machine with serving (the host's idle-training limits)."""
    step_pause_seconds: float = Field(default=0, ge=0, le=60)
    gpu_memory_fraction: float | None = Field(default=None, gt=0, le=1)


class EvaluationSettings(Strict):
    cases: str
    max_new_tokens: int = Field(default=256, ge=8, le=4096)
    # Chat-template arguments of the serving backend; training, mining and scoring all render with them.
    template_options: dict[str, bool] = Field(default_factory=dict)

    @field_validator("template_options")
    @classmethod
    def _known_options(cls, value):
        unknown = set(value) - {"enable_thinking"}
        if unknown:
            raise ValueError(f"unknown chat-template options: {sorted(unknown)}")
        return value

    @field_validator("cases")
    @classmethod
    def _inside(cls, value):
        return _relative(value)


class Admission(Strict):
    """One input the host's Lua stage admitted before dispatch (see docs/lua-design-contract.md)."""
    artifact: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingJob(Strict):
    schema_: Literal["janus-training-job.v1"] = Field(alias="schema")
    job_id: str
    role: Literal["specialist", "hub"] = "specialist"
    model: ModelRef
    dataset: DatasetRef
    stages: list[Literal["merge", "sft", "dpo", "ppo"]] = Field(min_length=1)
    evaluation: EvaluationSettings
    baseline_adapter: str | None = None   # the adapter serving now; stages must beat base + this adapter
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp32", "bf16"] = "auto"   # auto: bf16/fp16 on CUDA, fp32 on CPU
    threads: int | None = Field(default=None, ge=1, le=256)
    seed: int = 42
    admissions: list[Admission] = Field(default_factory=list)
    merge: MergeSettings = Field(default_factory=MergeSettings)
    sft: SFTSettings = Field(default_factory=SFTSettings)
    dpo: DPOSettings = Field(default_factory=DPOSettings)
    ppo: PPOSettings = Field(default_factory=PPOSettings)
    guard: GuardSettings = Field(default_factory=GuardSettings)
    limits: Limits = Field(default_factory=Limits)

    @field_validator("baseline_adapter")
    @classmethod
    def _baseline_inside(cls, value):
        return _relative(value)

    @field_validator("job_id")
    @classmethod
    def _job_id(cls, value):
        if not JOB_ID.match(value):
            raise ValueError("job_id is <name>-<UTC stamp YYYYmmddTHHMMSSZ>")
        return value

    @model_validator(mode="after")
    def _stages(self):
        positions = [STAGE_ORDER.index(stage) for stage in self.stages]
        if positions != sorted(set(positions)):
            raise ValueError("stages run once each in the order " + " -> ".join(STAGE_ORDER))
        if "ppo" in self.stages and self.role != "hub":
            raise ValueError("ppo trains the routing hub only")
        return self

    def input_artifacts(self) -> list[str]:
        """Every outside input this job loads; each must carry a host admission."""
        keys = [self.model.key(), self.dataset.key()]
        if self.baseline_adapter:
            keys.append(f"local:{self.baseline_adapter}")
        if "merge" in self.stages:
            keys += [f"local:{item}" for item in self.merge.adapters] if self.merge.adapters else ["local:adapters"]
        keys.append(f"local:{self.evaluation.cases}")
        return keys

    def resolve(self, root: Path, relative: str) -> Path:
        root = root.resolve()
        target = (root / _inside(relative)).resolve()
        if target != root and root not in target.parents:
            raise ValueError("paths must stay inside the job directory")
        return target


def load_job(path: Path) -> tuple[TrainingJob, Path]:
    path = Path(path)
    return TrainingJob.model_validate(json.loads(path.read_text(encoding="utf-8"))), path.parent.resolve()
