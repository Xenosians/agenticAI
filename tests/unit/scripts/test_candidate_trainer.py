"""The candidate trainer reports one known code per failure and finishes with its candidate digest."""
import json
from pathlib import Path

import pytest

from scripts import run_candidate_training as trainer

RECIPE = {"objective": "sft", "role": "hub", "tuning": "qlora", "model_path": "/weights/base",
          "dataset": "/runs/x/dataset.jsonl", "dataset_sha256": "a" * 64, "allowed_sources": ["Owned"],
          "allowed_licenses": ["Owned"], "output_root": "/runs/x/candidate", "max_steps": 6,
          "max_length": 768, "gradient_accumulation_steps": 4}


@pytest.mark.parametrize("message, code", [
    ("Only 1.2 GiB VRAM free", "insufficient_gpu_memory"),
    ("CUDA out of memory. Tried to allocate", "insufficient_gpu_memory"),
    ("QLoRA requires CUDA", "cuda_unavailable"),
    ("Chat template has no assistant loss boundary", "chat_template_unsupported"),
    ("Example exceeds the sequence budget", "example_too_long"),
    ("Dataset digest changed", "dataset_admission_failed"),
    ("/home/owner/private/path missing: token=abc", "training_failed"),
])
def test_each_failure_is_reported_as_one_known_code(message, code):
    assert trainer.error_code(RuntimeError(message)) == code


def run_trainer(tmp_path, monkeypatch, train):
    (tmp_path / "recipe.json").write_text(json.dumps(RECIPE))
    monkeypatch.setattr(trainer, "train_recipe", train)
    monkeypatch.setattr(trainer.sys, "argv", ["run_candidate_training.py", str(tmp_path)])
    exit_code = trainer.main()
    return exit_code, json.loads((tmp_path / "status.json").read_text())


def test_a_finished_run_reports_its_candidate_digest_and_final_losses(tmp_path, monkeypatch):
    def train(recipe, *, allow_training, progress):
        assert allow_training is True
        progress(3, train_loss=0.9)
        progress(4, train_loss=float("nan"), validation_loss=0.6)
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "manifest.json").write_text(json.dumps({
            "artifact_sha256": "c" * 64, "train_metrics": {"train_loss": 0.47},
            "validation_metrics": {"eval_loss": 0.42}}))
        return candidate

    exit_code, status = run_trainer(tmp_path, monkeypatch, train)

    assert exit_code == 0
    assert status == {"state": "completed", "candidate_sha256": "c" * 64, "step": 6, "max_steps": 6,
                      "train_loss": 0.47, "validation_loss": 0.42}


def test_a_failed_run_reports_only_its_code_and_last_progress(tmp_path, monkeypatch, capsys):
    def train(recipe, *, allow_training, progress):
        progress(2, train_loss=1.1)
        raise RuntimeError("CUDA out of memory at /home/owner/private")

    exit_code, status = run_trainer(tmp_path, monkeypatch, train)

    assert exit_code == 1
    assert status == {"state": "failed", "error": "insufficient_gpu_memory", "step": 2, "max_steps": 6,
                      "train_loss": 1.1, "validation_loss": None}
    assert "/home/owner/private" not in json.dumps(status)
    assert not Path(tmp_path / "status.json.tmp").exists()
