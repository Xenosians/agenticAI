"""A CPU worker in a small container recomputes activations instead of storing them.

Without it, LoRA on a 0.5B model with 1.5k-token rows keeps about 3.5 GB of activations, more than the
janus-trainer container's 3.5 GiB of RAM; with it, only each layer's input is kept.
"""
from learning.worker.job import TrainingJob
from learning.worker.stages import Run


def test_a_cpu_worker_recomputes_activations_like_a_gpu_worker(tmp_path):
    job = TrainingJob.model_validate({
        "schema": "janus-training-job.v1", "job_id": "account-20261119T000000Z", "device": "cpu",
        "model": {"source": "local", "path": "base"}, "dataset": {"source": "local", "path": "data"},
        "stages": ["sft"], "evaluation": {"cases": "data/heldout.jsonl"}})
    arguments, _ = Run(job, tmp_path, tmp_path / "out", None).args(tmp_path / "stage")
    assert arguments["use_cpu"] is True
    assert arguments["gradient_checkpointing"] is True
    assert arguments["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_a_stop_request_is_seen_after_every_micro_batch_not_only_after_a_full_step(tmp_path):
    """With 8 micro-batches per optimizer step, a CPU worker checked the stop file once every ~10 minutes;
    the host waits at most WORKER_GRACE_SECONDS before it kills the worker."""
    job = TrainingJob.model_validate({
        "schema": "janus-training-job.v1", "job_id": "account-20261119T000000Z", "device": "cpu",
        "model": {"source": "local", "path": "base"}, "dataset": {"source": "local", "path": "data"},
        "stages": ["sft"], "evaluation": {"cases": "data/heldout.jsonl"}})
    stop = tmp_path / "stop"
    callback = Run(job, tmp_path, tmp_path / "out", stop).stop_callback()
    callback.on_substep_end(None, None, None)              # nothing asked yet
    stop.write_text("yield")
    from learning.worker.stages import Interrupted

    import pytest

    with pytest.raises(Interrupted):
        callback.on_substep_end(None, None, None)


def test_a_finished_steps_memory_goes_back_to_the_system_on_a_cpu_worker(tmp_path, monkeypatch):
    """Measured on a 0.5B model: after generate(), freeing it left 2.1 GB resident (glibc keeps the pages), so
    the next stage's model doubled memory past the 3.5 GiB container; malloc_trim gave it back (0.8 GB)."""
    from learning.worker import stages as stages_module

    trimmed = []
    monkeypatch.setattr(stages_module, "release_freed_memory", lambda: trimmed.append(1))
    job = TrainingJob.model_validate({
        "schema": "janus-training-job.v1", "job_id": "account-20261119T000000Z", "device": "cpu",
        "model": {"source": "local", "path": "base"}, "dataset": {"source": "local", "path": "data"},
        "stages": ["sft"], "evaluation": {"cases": "data/heldout.jsonl"}})
    Run(job, tmp_path, tmp_path / "out", None).collect()
    assert trimmed == [1]


def test_releasing_freed_memory_never_fails_where_glibc_is_absent(monkeypatch):
    import ctypes

    from learning.worker.stages import release_freed_memory

    def missing(name):
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", missing)
    release_freed_memory()           # musl, macOS, Windows: nothing to trim, nothing raised


def test_bf16_only_where_the_gpu_computes_it_natively():
    from learning.worker.stages import choose_dtype

    assert choose_dtype(device="cuda", precision="auto", capability=(7, 5)) == "fp16"   # Kaggle's T4
    assert choose_dtype(device="cuda", precision="auto", capability=(8, 9)) == "bf16"   # an RTX 4050
    assert choose_dtype(device="cpu", precision="auto", capability=None) == "fp32"
    assert choose_dtype(device="cuda", precision="bf16", capability=(7, 5)) == "bf16"   # asked for by name


def test_one_job_uses_one_gpu_even_where_the_machine_has_several(monkeypatch):
    """Owner decision 2: one job is never split across GPUs. Kaggle offers two T4s; with both visible the
    trainer wrapped the model in DataParallel. The worker pins the first one before torch starts CUDA."""
    from learning.worker.stages import pin_one_gpu

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pin_one_gpu()
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")          # a CPU job stays without CUDA
    pin_one_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")       # the host chose: its first choice is kept
    pin_one_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
