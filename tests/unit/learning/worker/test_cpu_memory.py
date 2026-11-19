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
