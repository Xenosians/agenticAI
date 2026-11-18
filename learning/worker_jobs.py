"""Host side of the training worker: turn a reviewed specialist into a self-contained job folder.

The folder holds everything a worker needs (``job.json``, corpus, held-out cases, the base model and the
serving adapter to beat), so the same job runs inside this service, in a Proxmox container or on a
remote GPU. Every input crosses the operator's Lua admission here, before any file leaves the host
(docs/lua-design-contract.md); the receipts' digests travel in the job and the worker refuses inputs
without one.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learning.paths import REPOSITORY_ROOT

STAGES = ("merge", "sft", "dpo", "ppo")

# Chat-template arguments each serving backend passes (subagents/llm/backends). A backend missing here
# is refused: a job must never train or score with a prompt its model will not be served with.
SERVING_TEMPLATE_OPTIONS = {
    "qwen3": {"enable_thinking": False},     # qwen3_worker.py
    "hf-causal": {},                         # hf_causal_worker.py (template, else the fallback renderer)
    "qwen-funccall": {},                     # qwen_funcall.py
    "qwen-coder": {},                        # qwen_coder_worker.py
    "ministral": {},                         # ministral_hub.py
}


def serving_template_options(profile) -> dict[str, bool]:
    backend = str(getattr(profile, "backend", "") or "").strip().lower()
    if backend not in SERVING_TEMPLATE_OPTIONS:
        raise ValueError(f"no known prompt rendering for serving backend {backend!r}; add it before training")
    return dict(SERVING_TEMPLATE_OPTIONS[backend])


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link model files where the filesystem allows it; a job folder must not double the disk."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(source).rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue  # download caches and VCS files are not part of the model
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def _system_prompt(target, profile) -> str:
    from learning.training.specialist_sft import load_training_config, resolve_worker_prompt_profile
    from subagents.core.definitions.loader import load_agent_definition
    from subagents.core.tooling.capabilities import build_agent_capability_catalog
    from subagents.core.tooling.prompt import build_worker_system_prompt

    config = load_training_config(target.config_path)
    agent = load_agent_definition(REPOSITORY_ROOT / target.agent_definition)
    catalog = build_agent_capability_catalog(agent, include_arguments=True)
    return build_worker_system_prompt(agent, capability_catalog=catalog,
                                      prompt_profile=resolve_worker_prompt_profile(config=config,
                                                                                   base_profile=profile))


def _build_corpus(target, directory: Path) -> None:
    from learning.training.specialist_sft import build_specialist_sft_corpus

    build_specialist_sft_corpus(project_root=REPOSITORY_ROOT, config_path=target.config_path,
                                output_directory=directory)


def _admit(artifact: str, details: dict[str, Any]) -> str:
    """Run one job input through the operator's Lua admission; returns the receipt's digest."""
    from services.lua_adapters.host_boundaries import inspect_data

    receipt = inspect_data("model", "warm", {"artifacts": [artifact], "options": details},
                           {"loader": "training-worker-job", "module": __name__})
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _runtime_messages(system_prompt: str, user_request: str, semantic_context: dict[str, Any]):
    from learning.training.jira_sft import build_runtime_messages

    return build_runtime_messages(system_prompt=system_prompt, user_request=user_request,
                                  semantic_context=semantic_context)


def specialist_heldout_rows(target, *, system_prompt: str) -> list[dict[str, Any]]:
    """The specialist's reviewed held-out cases, each with the exact prompt the runtime would send."""
    from learning.evaluation.config_specialist_gate import _semantic_context
    from learning.training.specialist_sft import load_training_config

    config = load_training_config(target.config_path)
    rows = []
    for case in config.eval_cases:
        rows.append({
            "name": case.name,
            "prompt_messages": _runtime_messages(system_prompt, case.user, _semantic_context(case)),
            "expected_tool": case.expected_tool,
            "expected_arguments": case.expected_arguments,
            "allowed_arguments": case.allowed_arguments,
            "allowed_extra_arguments": case.allowed_extra_arguments,
            "max_new_tokens": config.max_new_tokens,
        })
    return rows


def _training_rows(corpus: Path, system_prompt: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for line in (corpus / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({"prompt_messages": _runtime_messages(system_prompt, record["user_request"],
                                                          record["semantic_context"]),
                     "chosen": record["target_response"]})
    return rows


def _external_rows(target) -> list[dict[str, Any]]:
    from learning.training.specialist_sft import load_external_sft_jsonl

    if not target.external_sft_jsonl:
        return []
    records, _sources = load_external_sft_jsonl(list(target.external_sft_jsonl))
    return [{"prompt_messages": record["prompt_messages"], "chosen": record["chosen"]} for record in records]


def _same_base(serving, base_profile) -> bool:
    """The serving adapter can be the worker's baseline only on the same base weights and backend."""
    try:
        same_weights = Path(serving.model_path).resolve() == Path(base_profile.model_path).resolve()
    except TypeError:
        return False
    same_backend = str(getattr(serving, "backend", "")).lower() == str(getattr(base_profile, "backend", "")).lower()
    return same_weights and same_backend


def build_specialist_job(target, *, profiles: dict, directory: Path, stages=("sft", "dpo"),
                         stamp: str | None = None, device: str = "auto", settings: dict | None = None) -> Path:
    """Write a worker job folder for one specialist; nothing is written when an admission is refused."""
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages: {unknown}")
    base_profile = profiles.get(target.base_model_key)
    if base_profile is None or base_profile.model_path is None:
        raise ValueError(f"{target.name}: the base model {target.base_model_key} has no local weights")
    # register_candidate_profile serves a new candidate with a copy of the BASE profile, so the base
    # profile's backend decides how its prompts are rendered.
    template_options = serving_template_options(base_profile)
    serving = profiles.get(target.candidate_model_key)
    serving_adapter = None
    if serving is not None and serving.adapter_path and _same_base(serving, base_profile):
        serving_adapter = Path(serving.adapter_path)
    # Otherwise the worker measures against the base alone; the host's own gate still compares the
    # candidate with the serving specialist under that specialist's real profile.
    directory = Path(directory)
    staging = directory.with_name(directory.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        system_prompt = _system_prompt(target, base_profile)
        _build_corpus(target, staging / "corpus")
        train = _training_rows(staging / "corpus", system_prompt, "train") + _external_rows(target)
        validation = _training_rows(staging / "corpus", system_prompt, "validation")
        _write_jsonl(staging / "data" / "train.jsonl", train)
        _write_jsonl(staging / "data" / "validation.jsonl", validation)
        _write_jsonl(staging / "data" / "heldout.jsonl", specialist_heldout_rows(target, system_prompt=system_prompt))
        shutil.rmtree(staging / "corpus")
        _link_or_copy(Path(base_profile.model_path), staging / "base")
        if serving_adapter is not None:
            _link_or_copy(serving_adapter, staging / "serving-adapter")
        job: dict[str, Any] = {
            "schema": "janus-training-job.v1", "job_id": f"{target.name}-{stamp or _stamp()}",
            "role": "specialist", "model": {"source": "local", "path": "base"},
            "dataset": {"source": "local", "path": "data"}, "stages": list(stages),
            "evaluation": {"cases": "data/heldout.jsonl", "template_options": template_options},
            "device": device,
            "sft": {"tuning": "qlora" if device == "cuda" else "lora"},
        }
        if serving_adapter is not None:
            job["baseline_adapter"] = "serving-adapter"
        for key, value in (settings or {}).items():
            job[key] = job.get(key, {}) | value if isinstance(value, dict) else value
        from learning.worker.job import TrainingJob

        parsed = TrainingJob.model_validate(job)
        sources = {"local:base": Path(base_profile.model_path), "local:data": staging / "data",
                   "local:data/heldout.jsonl": staging / "data" / "heldout.jsonl",
                   "local:serving-adapter": serving_adapter}
        job["admissions"] = [{"artifact": artifact,
                              "receipt_sha256": _admit(artifact, {"specialist": target.specialist,
                                                                  "source": str(sources.get(artifact) or "")})}
                             for artifact in parsed.input_artifacts()]
        TrainingJob.model_validate(job)
        (staging / "job.json").write_text(json.dumps(job, indent=2, sort_keys=True), encoding="utf-8")
        shutil.rmtree(directory, ignore_errors=True)
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory
