"""Candidate-only causal adaptation and SFT with full, LoRA, or QLoRA tuning.

RAG is retrieval, and continual learning is orchestration; neither is an optimizer.
Preference and reward objectives use their existing provenance-aware trainers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from learning.evidence.sanitizer import sanitize_text


class AdaptationRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: Literal["sft", "causal_adaptation"]
    role: Literal["hub", "specialist"] = "specialist"
    tuning: Literal["full", "lora", "qlora"] = "qlora"
    model_path: Path
    dataset: Path
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_sources: list[str] = Field(min_length=1)
    allowed_licenses: list[str] = Field(min_length=1)
    output_root: Path
    max_steps: int = Field(default=80, ge=1, le=4000)
    max_length: int = Field(default=768, ge=64, le=8192)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    lora_r: int = Field(default=16, ge=1, le=128)
    lora_alpha: int = Field(default=32, ge=1, le=256)
    min_free_vram_gib: float = Field(default=2.0, gt=0)
    seed: int = 42


def admit_records(recipe: AdaptationRecipe) -> tuple[list[dict], list[dict]]:
    from learning.training.methods import validate_method_role
    validate_method_role(recipe.objective, recipe.role, recipe.tuning)
    raw = recipe.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != recipe.dataset_sha256:
        raise ValueError("Dataset hash differs from the reviewed recipe.")
    partitions = {"train": [], "validation": []}
    identities: set[str] = set()
    content_hashes: set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Dataset records must be JSON objects.")
        if row.get("reviewed") is not True or row.get("source") not in recipe.allowed_sources or row.get("license") not in recipe.allowed_licenses:
            raise ValueError("Dataset record lacks approved provenance/licensing/review.")
        split = row.get("split")
        if split not in partitions or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each record requires an identity and train/validation split.")
        payload = row.get("text") if recipe.objective == "causal_adaptation" else row.get("messages")
        if recipe.objective == "causal_adaptation":
            valid = isinstance(payload, str) and bool(payload.strip())
        else:
            valid = isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[-1], dict) and payload[-1].get("role") == "assistant"
            valid = valid and all(isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"} and isinstance(m.get("content"), str) and m["content"].strip() for m in payload)
        if not valid:
            raise ValueError("Invalid objective-specific dataset payload.")
        text = payload if isinstance(payload, str) else "\n".join(m["content"] for m in payload)
        if sanitize_text(text) != text:
            raise ValueError("Dataset contains material flagged by the privacy/secret scanner.")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if row["id"] in identities or digest in content_hashes:
            raise ValueError("Duplicate identity/content or cross-split leakage.")
        identities.add(row["id"])
        content_hashes.add(digest)
        partitions[split].append(row)
    if not all(partitions.values()):
        raise ValueError("Both training and validation evidence are required.")
    return partitions["train"], partitions["validation"]


def encode_record(tokenizer, row: dict, recipe: AdaptationRecipe) -> dict:
    if recipe.objective == "causal_adaptation":
        ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
        labels = list(ids)
    else:
        messages = row["messages"]
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError("Chat template does not preserve an assistant-only loss boundary.")
        labels = [-100] * len(prefix) + ids[len(prefix):]
    if len(ids) > recipe.max_length:
        raise ValueError("Example exceeds sequence budget; do not silently truncate supervision.")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def train_recipe(recipe: AdaptationRecipe, *, allow_training: bool = False) -> Path:
    if not allow_training:
        raise PermissionError("Training requires explicit --allow-training.")
    train_rows, validation_rows = admit_records(recipe)
    if not recipe.model_path.is_dir():
        raise ValueError("A local acquired model is required.")
    if recipe.output_root.resolve().is_relative_to(recipe.model_path.resolve()):
        raise ValueError("Candidate output cannot overwrite or live inside base weights.")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from learning.training.specialist_sft import SpecialistCollator, SpecialistTokenDataset
    from learning.continual.storage import fingerprint_directory
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires CUDA.")
    minimum = max(recipe.min_free_vram_gib, 16.0 if recipe.tuning == "full" else 2.0)
    if torch.cuda.mem_get_info()[0] / 1024**3 < minimum:
        raise RuntimeError(f"Resource preflight requires {minimum:g} GiB free VRAM; no model loaded.")
    set_seed(recipe.seed)
    tokenizer = AutoTokenizer.from_pretrained(recipe.model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("A causal EOS token is required.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = [encode_record(tokenizer, row, recipe) for row in train_rows]
    validation_data = [encode_record(tokenizer, row, recipe) for row in validation_rows]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"local_files_only": True, "dtype": dtype, "device_map": {"": 0}}
    if recipe.tuning == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(recipe.model_path, **kwargs)
    try:
        model.config.use_cache = False
        if recipe.tuning == "qlora":
            model = prepare_model_for_kbit_training(model)
        if recipe.tuning != "full":
            model = get_peft_model(model, LoraConfig(r=recipe.lora_r, lora_alpha=recipe.lora_alpha, target_modules="all-linear", task_type="CAUSAL_LM"))
            model.enable_input_require_grads()
        run = recipe.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run.mkdir(parents=True, exist_ok=False)
        args = TrainingArguments(output_dir=str(run / "trainer"), max_steps=recipe.max_steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=recipe.gradient_accumulation_steps,
            learning_rate=recipe.learning_rate, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            report_to="none", save_strategy="no", logging_steps=10, seed=recipe.seed,
            dataloader_num_workers=0, optim="adamw_torch")
        trainer = Trainer(model=model, args=args, train_dataset=SpecialistTokenDataset(train_data),
            eval_dataset=SpecialistTokenDataset(validation_data), data_collator=SpecialistCollator(pad_token_id=tokenizer.pad_token_id))
        trained = trainer.train()
        evaluated = trainer.evaluate()
        artifact = run / ("model" if recipe.tuning == "full" else "adapter")
        model.save_pretrained(artifact, safe_serialization=True)
        tokenizer.save_pretrained(artifact)
        report = {"schema": "learning-recipe-run.v1", "recipe": recipe.model_dump(mode="json"),
            "base_artifact_sha256": fingerprint_directory(recipe.model_path),
            "train_metrics": trained.metrics, "validation_metrics": evaluated,
            "artifact_sha256": fingerprint_directory(artifact), "artifact": str(artifact),
            "promotion_status": "candidate_only", "training_records": len(train_rows), "validation_records": len(validation_rows)}
        (run / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return run
    finally:
        if "trainer" in locals():
            del trainer
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
