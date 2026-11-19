"""Run one training job: weight-space merge -> SFT (LoRA/QLoRA) -> DPO -> PPO, each kept only when it beats
the stage before it on the job's held-out cases.

Usage on any worker:  python -m learning.worker.stages <job.json> <output directory> [--stop-file PATH]

Nothing here imports Janus; the host turns ``result.json`` into a candidate and re-scores it with its own
gate before anything can serve.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import Collator, cases_digest, encode_answer, encode_pair, heldout_cases, preference_rows, sft_rows, split
from .guard import FitGuard, StageGate
from .job import TrainingJob, load_job
from .loading import (NotAdmitted, admitted_load, fetch_dataset, fetch_model, load_base, load_tokenizer,
                      lora_targets, own_output, require_admissions)
from .loss import answer_only_trainer
from .prompts import encode_prompt, render_prompt, tokenizes_messages_itself
from .scoring import score_case, tool_call_reward

FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class Interrupted(RuntimeError):
    """The host asked the worker to yield (the stop file exists)."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint_directory(directory: Path) -> str:
    """Same digest as Janus's ``learning.continual.storage.fingerprint_directory``."""
    directory = Path(directory).resolve()
    digest = hashlib.sha256()
    files = sorted((path for path in directory.rglob("*") if path.is_file()),
                   key=lambda item: str(item.relative_to(directory)))
    if not files:
        raise ValueError(f"Directory is empty: {directory}")
    for path in files:
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                file_digest.update(chunk)
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def _clean(text: str) -> str:
    """Janus serving's response normalisation: strip, then drop one Markdown JSON fence."""
    text = text.strip()
    fenced = FENCE.fullmatch(text)
    return fenced.group(1).strip() if fenced else text


def pretokenized(trainer_class):
    """A DPO trainer for rows that already hold ``prompt_ids``, ``chosen_ids`` and ``rejected_ids``.

    trl tokenizes text rows itself; Mistral's tokenizer must encode the messages instead (see prompts.py).
    trl's own filter for rows whose prompt alone fills ``max_length`` is kept.
    """
    class PreTokenized(trainer_class):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            if "prompt_ids" not in dataset.column_names:
                return super()._prepare_dataset(dataset, processing_class, args, dataset_name)
            if args.max_length is not None and getattr(args, "truncation_mode", "keep_start") == "keep_start":
                dataset = dataset.filter(lambda row: len(row["prompt_ids"]) < args.max_length)
            return dataset

    return PreTokenized


def pin_one_gpu() -> None:
    """One job runs on one GPU (owner decision 2: never split across devices).

    With two GPUs visible (Kaggle's two T4s) the trainer wraps the model in DataParallel. The first visible
    GPU is kept; an empty value (a CPU job) stays empty. Must run before torch initialises CUDA.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif "," in visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",", 1)[0].strip()


def choose_dtype(*, device: str, precision: str, capability: tuple[int, int] | None) -> str:
    """fp32 on a CPU; bf16 only where the GPU computes it natively (compute capability 8.0 and newer).

    torch reports bf16 as supported on older GPUs (Kaggle's T4 is 7.5) through slow emulation; fp16 is native.
    """
    if precision == "fp32" or device == "cpu":
        return "fp32"
    if precision == "bf16":
        return "bf16"
    return "bf16" if capability is not None and capability[0] >= 8 else "fp16"


def release_freed_memory() -> None:
    """Give freed heap pages back to the system (glibc keeps them after a model is freed).

    Measured on a 0.5B model on a CPU: after generate(), freeing it left 2.1 GB resident, and the next stage's
    model doubled memory past a 3.5 GiB container; malloc_trim returned it (0.8 GB). Elsewhere it does nothing.
    """
    import ctypes

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Run:
    def __init__(self, job: TrainingJob, root: Path, output: Path, stop_file: Path | None):
        import torch

        self.torch = torch
        self.job, self.root, self.output, self.stop_file = job, root, output, stop_file
        self.cache = Path(os.environ.get("JANUS_WORKER_CACHE", output / "cache"))
        if job.threads:
            torch.set_num_threads(job.threads)
        cuda = torch.cuda.is_available()
        if job.device == "cuda" and not cuda:
            raise RuntimeError("the job asks for CUDA and this worker has no CUDA device")
        self.device = "cuda" if job.device == "cuda" or (job.device == "auto" and cuda) else "cpu"
        self.dtype = choose_dtype(device=self.device, precision=job.precision,
                                  capability=torch.cuda.get_device_capability() if self.device == "cuda" else None)
        if self.device == "cuda" and job.limits.gpu_memory_fraction:
            torch.cuda.set_per_process_memory_fraction(job.limits.gpu_memory_fraction)
        self.quantize = job.sft.tuning == "qlora" and self.device == "cuda"
        self.notes: list[str] = []
        self.progress: dict[str, Any] = {"stages": []}   # what finished, kept when a later step fails
        if job.sft.tuning == "qlora" and not self.quantize:
            self.notes.append("QLoRA needs CUDA; this CPU worker trained plain LoRA in fp32 instead")
        torch.manual_seed(job.seed)

    # -- plumbing -------------------------------------------------------------------------------------------

    def check_stop(self) -> None:
        if self.stop_file is not None and self.stop_file.exists():
            raise Interrupted("the host asked the worker to yield")

    def stop_callback(self):
        from transformers import TrainerCallback

        run = self

        class Yield(TrainerCallback):
            def on_substep_end(self, args, state, control, **kwargs):
                # Between micro-batches too: with gradient accumulation a full step on a CPU takes minutes.
                run.check_stop()

            def on_step_end(self, args, state, control, **kwargs):
                run.check_stop()
                pause = run.job.limits.step_pause_seconds
                while pause > 0:  # slept in slices so a stop request is seen during the pause
                    time.sleep(min(0.25, pause))
                    pause -= 0.25
                    run.check_stop()

        return Yield()

    def args(self, directory: Path, **overrides):
        from transformers import TrainingArguments

        common = dict(output_dir=str(directory), report_to=[], seed=self.job.seed, save_only_model=True,
                      use_cpu=self.device == "cpu", bf16=self.dtype == "bf16", fp16=self.dtype == "fp16",
                      dataloader_pin_memory=self.device == "cuda", disable_tqdm=True, logging_steps=1,
                      # Activations are recomputed on every device: a CPU worker in a small container would
                      # otherwise hold gigabytes of them (a 0.5B model with 1.5k-token rows needs ~3.5 GB).
                      remove_unused_columns=False, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={"use_reentrant": False})
        common.update(overrides)
        return common, TrainingArguments

    def collect(self) -> None:
        """Return a finished step's memory; call it after the step's own references are gone."""
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        release_freed_memory()

    def base(self):
        return load_base(self.job, self.model_directory, self.gguf_file, device=self.device,
                         dtype_name=self.dtype, quantize_4bit=self.quantize)

    def with_adapter(self, adapter: Path | None, *, trainable: bool, outside: bool = False):
        """The base model plus ``adapter`` (a stage output, or the admitted serving adapter)."""
        from peft import PeftModel

        model = self.base()
        if trainable and self.quantize:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                    gradient_checkpointing_kwargs={"use_reentrant": False})
        elif trainable:
            model.enable_input_require_grads()   # checkpointing through frozen embeddings needs input grads
        if adapter is None:
            return model
        if outside:
            relative = adapter.relative_to(self.root).as_posix()
            return admitted_load(self.job, f"local:{relative}",
                                 lambda: PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable))
        return own_output(lambda: PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable))

    # -- held-out evaluation --------------------------------------------------------------------------------

    def render(self, messages) -> str:
        return render_prompt(self.tokenizer, messages, self.job.evaluation.template_options)

    def encode(self, messages) -> list[int]:
        """The prompt's token ids exactly as serving encodes them; every stage and the evaluation use these."""
        return encode_prompt(self.tokenizer, messages, self.job.evaluation.template_options)

    def generate(self, model, messages, max_new_tokens: int) -> str:
        torch = self.torch
        prompt_ids = self.encode(messages)
        input_ids = torch.tensor([prompt_ids], device=self.device if not self.quantize else "cuda")
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id)
        return _clean(self.tokenizer.decode(output[0, len(prompt_ids):], skip_special_tokens=True))

    def evaluate(self, model, label: str) -> dict:
        model.eval()
        results = []
        for case in self.cases:
            self.check_stop()
            budget = case.get("max_new_tokens") or self.job.evaluation.max_new_tokens
            results.append(score_case(case, self.generate(model, case["prompt_messages"], budget)))
        report = {"label": label, "passed": sum(item["passed"] for item in results), "total": len(results),
                  "cases_sha256": self.cases_sha256, "needs_host_policy": sum(item["needs_host_policy"]
                                                                            for item in results),
                  "cases": results}
        (self.output / "evaluations").mkdir(parents=True, exist_ok=True)
        (self.output / "evaluations" / f"{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {key: report[key] for key in ("passed", "total", "cases_sha256", "needs_host_policy")}

    # -- stages ---------------------------------------------------------------------------------------------

    def merge(self, current: Path | None) -> dict:
        from peft import PeftModel

        settings = self.job.merge
        if settings.adapters:
            directories = [self.job.resolve(self.root, item) for item in settings.adapters]
        else:
            folder = self.root / "adapters"
            directories = sorted(path for path in folder.iterdir() if (path / "adapter_config.json").is_file()) \
                if folder.is_dir() else []
        if len(directories) < 2:
            return {"ran": False, "reason": f"merge needs two or more adapters of the base; found {len(directories)}"}
        names = [f"a{index}" for index in range(len(directories))]
        model = self.base()

        def admitted(index: int):
            relative = directories[index].relative_to(self.root).as_posix()
            key = f"local:{relative}" if settings.adapters else "local:adapters"
            return key

        model = admitted_load(self.job, admitted(0), lambda: PeftModel.from_pretrained(
            model, str(directories[0]), adapter_name=names[0]))
        for index in range(1, len(directories)):
            admitted_load(self.job, admitted(index), lambda index=index: model.load_adapter(
                str(directories[index]), adapter_name=names[index]))
        ranks = [model.peft_config[name].r for name in names]
        weights = settings.weights or [1.0 / len(names)] * len(names)
        options: dict[str, Any] = {"combination_type": settings.method}
        if settings.method in {"ties", "ties_svd", "dare_ties", "dare_ties_svd", "dare_linear", "dare_linear_svd",
                               "magnitude_prune", "magnitude_prune_svd"}:
            options["density"] = settings.density
        if settings.method.endswith("svd") or settings.method == "svd":
            options["svd_rank"] = max(ranks)
        model.add_weighted_adapter(names, weights, "merged", **options)
        model.set_adapter("merged")
        for name in names:
            model.delete_adapter(name)
        directory = self.output / "stages" / "merge"
        model.save_pretrained(str(directory), selected_adapters=["merged"])
        adapter = directory / "merged"
        evaluation = self.evaluate(model, "merge")
        return {"ran": True, "adapter": adapter, "adapters": len(directories), "method": settings.method,
                "rank": max(ranks), **evaluation}

    def sft(self, current: Path | None) -> dict:
        from peft import LoraConfig, get_peft_model
        from transformers import EarlyStoppingCallback, Trainer, TrainerCallback

        settings, job = self.job.sft, self.job
        train_path = self.dataset_directory / job.dataset.train
        if not train_path.is_file():
            return {"ran": False, "reason": f"the dataset has no {job.dataset.train}"}
        train = sft_rows(train_path)
        validation_path = self.dataset_directory / job.dataset.validation if job.dataset.validation else None
        if validation_path is not None and validation_path.is_file():
            validation = sft_rows(validation_path)
        else:
            train, validation = split(train, job.guard.validation_fraction, job.seed)
        if not train or not validation:
            return {"ran": False, "reason": "SFT needs training rows and validation rows for the fit guard"}

        def encode(rows):
            return [encode_pair(self.tokenizer, self.encode(row["prompt_messages"]), row["chosen"],
                                settings.max_length) for row in rows]

        train_features, validation_features = encode(train), encode(validation)
        model = self.with_adapter(current, trainable=True)
        if current is None:
            model = get_peft_model(model, LoraConfig(r=settings.lora_r, lora_alpha=settings.lora_alpha,
                                                     lora_dropout=settings.lora_dropout,
                                                     target_modules=lora_targets(self.model_directory),
                                                     task_type="CAUSAL_LM"))
        steps_per_epoch = max(1, math.ceil(len(train_features) / (settings.batch_size *
                                                                  settings.gradient_accumulation)))
        eval_every = max(1, steps_per_epoch // settings.evaluations_per_epoch)
        directory = self.output / "stages" / "sft"
        guard = FitGuard(patience=job.guard.patience, min_improvement=job.guard.min_improvement,
                         underfit_loss=job.guard.underfit_loss)
        losses: dict[str, float] = {}

        class Observe(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    losses["train"] = float(logs["loss"])

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                validation_loss = float((metrics or {}).get("eval_loss", float("nan")))
                if guard.observe(step=state.global_step, train_loss=losses.get("train", validation_loss),
                                 validation_loss=validation_loss) == "stop":
                    control.should_training_stop = True

        common, Arguments = self.args(
            directory, num_train_epochs=settings.epochs, learning_rate=settings.learning_rate,
            per_device_train_batch_size=settings.batch_size, per_device_eval_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation, eval_strategy="steps",
            eval_steps=eval_every, save_strategy="steps", save_steps=eval_every, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            lr_scheduler_type="cosine", warmup_steps=0)
        trainer = answer_only_trainer(Trainer)(model=model, args=Arguments(**common), train_dataset=train_features,
                          eval_dataset=validation_features, data_collator=Collator(self.tokenizer.pad_token_id),
                          callbacks=[Observe(), self.stop_callback(),
                                     EarlyStoppingCallback(early_stopping_patience=job.guard.patience,
                                                           early_stopping_threshold=job.guard.min_improvement)])
        trainer.train()
        adapter = directory / "adapter"
        trainer.model.save_pretrained(str(adapter))
        for checkpoint in directory.glob("checkpoint-*"):
            shutil.rmtree(checkpoint, ignore_errors=True)
        history = guard.history
        summary = guard.summary() | {
            "first_validation_loss": history[0][2] if history else None,
            "last_train_loss": losses.get("train"),
        }
        evaluation = self.evaluate(trainer.model, "sft")
        return {"ran": True, "adapter": adapter, "tuning": "qlora" if self.quantize else "lora", "rows": len(train),
                "validation_rows": len(validation), "guard": summary, **evaluation}

    def dpo(self, current: Path | None) -> dict:
        from trl import DPOConfig, DPOTrainer

        settings, job = self.job.dpo, self.job
        pairs = []
        preference_file = self.dataset_directory / job.dataset.preferences if job.dataset.preferences else None
        if preference_file is not None and preference_file.is_file():
            pairs = preference_rows(preference_file)
        mined = 0
        if not pairs and settings.self_generated_pairs:
            pairs, mined = self.mine_pairs(current, settings.pair_limit), 1
            self.collect()
        if not pairs:
            return {"ran": False, "reason": "no preference pairs: no preferences file, and the model already "
                                            "answers every training prompt as chosen"}
        # Token ids, not text: DPO sees the prompts serving and the SFT stage see, and no boundary merge of
        # prompt and answer can shift where the answer starts.
        dataset = self.as_dataset([{"prompt_ids": self.encode(row["prompt_messages"]),
                                    "chosen_ids": encode_answer(self.tokenizer, row["chosen"]),
                                    "rejected_ids": encode_answer(self.tokenizer, row["rejected"])}
                                   for row in pairs])
        model = self.with_adapter(current, trainable=True)
        peft_config = None
        if current is None:
            from peft import LoraConfig

            peft_config = LoraConfig(r=job.sft.lora_r, lora_alpha=job.sft.lora_alpha,
                                     target_modules=lora_targets(self.model_directory),
                                     lora_dropout=job.sft.lora_dropout, task_type="CAUSAL_LM")
        directory = self.output / "stages" / "dpo"
        common, _ = self.args(directory, num_train_epochs=settings.epochs, learning_rate=settings.learning_rate,
                              per_device_train_batch_size=1, gradient_accumulation_steps=settings.gradient_accumulation,
                              save_strategy="no", lr_scheduler_type="cosine")
        common.pop("save_only_model")
        trainer = pretokenized(DPOTrainer)(model=model, ref_model=None, args=DPOConfig(
            **common, beta=settings.beta, loss_type=[settings.loss_type], max_length=settings.max_length),
            train_dataset=dataset, processing_class=self.tokenizer, peft_config=peft_config,
            callbacks=[self.stop_callback()])
        trainer.train()
        adapter = directory / "adapter"
        trained = trainer.model
        if "ref" in getattr(trained, "peft_config", {}):
            trained.delete_adapter("ref")
        trained.save_pretrained(str(adapter), selected_adapters=["default"])
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        evaluation = self.evaluate(trained, "dpo")
        return {"ran": True, "adapter": adapter, "pairs": len(pairs), "self_generated": bool(mined),
                "beta": settings.beta, "first_loss": losses[0] if losses else None,
                "last_loss": losses[-1] if losses else None, **evaluation}

    def mine_pairs(self, current: Path | None, limit: int) -> list[dict]:
        """The model's own wrong answers to training prompts, paired with the reviewed answer."""
        train_path = self.dataset_directory / self.job.dataset.train
        if not train_path.is_file():
            return []
        model = self.with_adapter(current, trainable=False)
        model.eval()
        pairs = []
        for row in sft_rows(train_path):
            self.check_stop()
            produced = self.generate(model, row["prompt_messages"], self.job.evaluation.max_new_tokens)
            if produced.strip() and tool_call_reward(row["chosen"], produced) < 1.0:
                pairs.append(row | {"rejected": produced})
            if len(pairs) >= limit:
                break
        return pairs

    def ppo(self, current: Path | None) -> dict:
        """PPO-family clipped policy optimisation with a group-relative baseline (GRPO: no value model, which
        keeps it trainable on one small GPU or a CPU) against the verifiable tool-call reward."""
        from trl import GRPOConfig, GRPOTrainer

        settings, job = self.job.ppo, self.job
        train_path = self.dataset_directory / job.dataset.train
        if not train_path.is_file():
            return {"ran": False, "reason": f"the dataset has no {job.dataset.train}"}
        if tokenizes_messages_itself(self.tokenizer):
            # trl's GRPO samples from text prompts; this tokenizer's prompt as text re-tokenizes to other ids.
            return {"ran": False, "reason": "PPO needs a text chat template; this tokenizer encodes messages itself"}
        rows = sft_rows(train_path)
        dataset = self.as_dataset([{"prompt": self.render(row["prompt_messages"]),
                                    "expected": row["chosen"]} for row in rows])
        rewards: list[float] = []

        def reward(completions, expected, **kwargs):
            values = [tool_call_reward(answer, _clean(completion)) for completion, answer in zip(completions, expected)]
            rewards.extend(values)
            return values

        model = self.with_adapter(current, trainable=True)
        peft_config = None
        if current is None:
            from peft import LoraConfig

            peft_config = LoraConfig(r=job.sft.lora_r, lora_alpha=job.sft.lora_alpha,
                                     target_modules=lora_targets(self.model_directory),
                                     lora_dropout=job.sft.lora_dropout, task_type="CAUSAL_LM")
        directory = self.output / "stages" / "ppo"
        common, _ = self.args(directory, num_train_epochs=settings.epochs, learning_rate=settings.learning_rate,
                              per_device_train_batch_size=settings.samples_per_prompt,
                              gradient_accumulation_steps=1, save_strategy="no")
        common.pop("save_only_model")
        trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=GRPOConfig(
            **common, num_generations=settings.samples_per_prompt, max_completion_length=settings.max_new_tokens,
            beta=settings.kl_beta, epsilon=settings.clip_range, loss_type="grpo"),
            train_dataset=dataset, processing_class=self.tokenizer, peft_config=peft_config,
            callbacks=[self.stop_callback()])
        trainer.train()
        adapter = directory / "adapter"
        trained = trainer.model
        if "ref" in getattr(trained, "peft_config", {}):
            trained.delete_adapter("ref")
        trained.save_pretrained(str(adapter), selected_adapters=["default"])
        evaluation = self.evaluate(trained, "ppo")
        steps = trainer.state.global_step
        return {"ran": True, "adapter": adapter, "method": "ppo-clip, group-relative baseline (GRPO)",
                "steps": steps, "samples_per_prompt": settings.samples_per_prompt,
                "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None, **evaluation}

    @staticmethod
    def as_dataset(rows: list[dict]):
        from datasets import Dataset

        return Dataset.from_list(rows)

    # -- the run --------------------------------------------------------------------------------------------

    def execute(self) -> dict:
        job = self.job
        self.model_directory, self.gguf_file = fetch_model(job, self.root, self.cache)
        self.dataset_directory = fetch_dataset(job, self.root, self.cache)
        self.tokenizer = load_tokenizer(job, self.model_directory, self.gguf_file)
        self.cases = heldout_cases(job.resolve(self.root, job.evaluation.cases))
        self.cases_sha256 = cases_digest(self.cases)
        current: Path | None = None
        if job.baseline_adapter:
            current = job.resolve(self.root, job.baseline_adapter)
            baseline_model = self.with_adapter(current, trainable=False, outside=True)
        else:
            baseline_model = self.base()
        baseline = self.evaluate(baseline_model, "baseline")
        del baseline_model
        self.collect()
        self.progress["baseline"] = baseline
        previous: dict | None = baseline if baseline["passed"] > 0 else None
        gate, stages = StageGate(), self.progress["stages"]
        for name in job.stages:
            if previous is not None and previous["passed"] >= previous["total"]:
                # A stage is kept only when strictly better: nothing can beat a perfect start, so no stage
                # trains for hours (a CPU worker's SFT) towards a verdict known in advance.
                stages.append({"stage": name, "ran": False, "kept": False, "seconds": 0.0,
                               "reason": "the starting model already passes every held-out case"})
                continue
            self.check_stop()
            started = time.monotonic()
            outcome = getattr(self, name)(current)
            self.collect()
            outcome = {"stage": name, **outcome, "seconds": round(time.monotonic() - started, 1)}
            if outcome["ran"]:
                verdict = gate.judge(stage=name, passed=outcome["passed"], total=outcome["total"],
                                     previous=previous, cases_sha256=self.cases_sha256)
                outcome |= {"kept": verdict["kept"], "reason": verdict["reason"]}
                if verdict["kept"]:
                    current, previous = outcome["adapter"], {"passed": outcome["passed"], "total": outcome["total"],
                                                             "cases_sha256": self.cases_sha256}
                outcome["adapter"] = Path(outcome["adapter"]).relative_to(self.output).as_posix()
            else:
                outcome["kept"] = False
            stages.append(outcome)
        trained = current is not None and current != (job.resolve(self.root, job.baseline_adapter)
                                                       if job.baseline_adapter else None)
        result = {"outcome": "trained" if trained else "no_improvement", "baseline": baseline, "stages": stages}
        if trained:
            final = self.output / "final"
            shutil.copytree(current, final)
            result |= {"final_adapter": "final", "final_adapter_sha256": fingerprint_directory(final),
                       "final_stage": next(item["stage"] for item in reversed(stages) if item.get("kept"))}
        return result

    def describe(self) -> dict:
        import importlib.metadata as metadata

        torch = self.torch
        versions = {}
        for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        gpu = None
        if self.device == "cuda":
            properties = torch.cuda.get_device_properties(0)
            gpu = {"name": properties.name, "memory_gib": round(properties.total_memory / 2**30, 1)}
        return {"host": platform.node(), "device": self.device, "precision": self.dtype, "gpu": gpu,
                "cpu_count": os.cpu_count(), "threads": torch.get_num_threads(), "python": platform.python_version(),
                "versions": versions, "notes": self.notes}


def _partial(worker) -> dict:
    """The baseline and the stages that finished before a run stopped (their outputs stay unkept)."""
    if worker is None:
        return {}
    progress = dict(worker.progress)
    progress["stages"] = [item | {"kept": False} for item in progress.get("stages", [])]
    return progress


def _peak_memory(worker) -> dict:
    """Peak memory of this process, so the host can size the next job for each machine."""
    import resource

    peak = {"peak_ram_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    if worker is not None and worker.device == "cuda":
        peak["peak_gpu_gib"] = round(worker.torch.cuda.max_memory_allocated() / 2**30, 2)
    return peak


def run(job_path: Path, output: Path, *, stop_file: Path | None = None,
        propagate: tuple[type[BaseException], ...] = ()) -> dict:
    """Run a job; ``propagate`` lists exceptions an in-process host raises to stop it (its own yield)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record: dict[str, Any] = {"started_at": _utc()}
    try:
        job, root = load_job(Path(job_path))
        record["job_id"] = job.job_id
        require_admissions(job)
    except NotAdmitted as refusal:
        record |= {"outcome": "refused", "error": str(refusal)}
    except Exception as error:  # noqa: BLE001 - an invalid job is reported, not raised
        record |= {"outcome": "invalid", "error": f"{type(error).__name__}: {str(error)[:500]}"}
    else:
        worker = None
        try:
            worker = Run(job, root, output, stop_file)
            record["worker"] = worker.describe()
            record |= worker.execute()
        except propagate:
            shutil.rmtree(output / "final", ignore_errors=True)
            raise
        except Interrupted as stop:
            record |= _partial(worker) | {"outcome": "interrupted", "error": str(stop)}
            shutil.rmtree(output / "final", ignore_errors=True)
        except NotAdmitted as refusal:
            record |= _partial(worker) | {"outcome": "refused", "error": str(refusal)}
        except Exception as error:  # noqa: BLE001 - the host needs the failure, not a crashed worker
            record |= _partial(worker) | {"outcome": "failed", "error": f"{type(error).__name__}: {str(error)[:500]}",
                                          "trace": traceback.format_exc()[-4000:]}
        if worker is not None and "worker" not in record:
            record["worker"] = worker.describe()
        if worker is not None:
            record["worker"] |= _peak_memory(worker)
    record |= {"finished_at": _utc(), "seconds": round(time.monotonic() - started, 1)}
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, output / "result.json")
    return record


def main(argv: list[str] | None = None) -> int:
    pin_one_gpu()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job")
    parser.add_argument("output")
    parser.add_argument("--stop-file")
    arguments = parser.parse_args(argv)
    result = run(Path(arguments.job), Path(arguments.output),
                 stop_file=Path(arguments.stop_file) if arguments.stop_file else None)
    print(json.dumps({key: result.get(key) for key in ("job_id", "outcome", "final_stage", "error")}))
    return 0 if result["outcome"] in {"trained", "no_improvement"} else 1


if __name__ == "__main__":
    sys.exit(main())
