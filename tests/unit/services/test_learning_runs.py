"""Candidate SFT runs started from the Learning page.

A run starts only from the exact ready check the owner saw, runs one at a time on
the host in a separate trainer process without credentials, reports progress
through its status file, can be cancelled by its owner, and never changes what
serves requests.
"""
import json
import math

import pytest

from learning.training.recipes import AdaptationRecipe, admit_records
from services.extensions.store import ExtensionStore
from services.learning_control.application import (
    LearningSelectionError,
    SelectionRequest,
    selection_preflight,
)
from services.learning_control.runs import (
    TRAINER_ENVIRONMENT,
    RunPolicy,
    RunRegistry,
    RunRequest,
)

OWNER, OTHER = "owner-a", "owner-b"


def examples(count=12, prefix="question"):
    return "\n".join(json.dumps({"messages": [{"role": "user", "content": f"{prefix} {index}"},
                                              {"role": "assistant", "content": f"answer {index}"}]})
                     for index in range(count))


class FakeProcesses:
    """Stands in for trainer processes: records launches, liveness and stop signals."""

    def __init__(self):
        self.launched, self.stopped, self.alive = [], [], {}
        self.next_pid = 4000
        self.fail_launch = False

    def launch(self, run_dir, environment):
        if self.fail_launch:
            raise OSError("cannot start")
        self.next_pid += 1
        self.launched.append((run_dir, environment))
        self.alive[self.next_pid] = f"start-{self.next_pid}"
        return self.next_pid

    def identity(self, pid):
        return self.alive.get(pid)

    def stop(self, pid):
        self.stopped.append(pid)
        self.alive.pop(pid, None)

    def exit(self, pid):
        self.alive.pop(pid, None)


@pytest.fixture
def store(tmp_path):
    return ExtensionStore(tmp_path / "registry")


@pytest.fixture
def base(tmp_path):
    path = tmp_path / "base-model"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return path


@pytest.fixture
def processes():
    return FakeProcesses()


@pytest.fixture
def registry(tmp_path, processes):
    return RunRegistry(tmp_path / "runs", policy=RunPolicy(epochs=3, max_steps=200, gradient_accumulation_steps=4),
                       launch=processes.launch, identity=processes.identity, stop=processes.stop)


def owned(store, owner=OWNER, backend="uploaded", count=12, prefix="question"):
    model = store.put(owner, "model", {"name": "My weights", "backend": backend, "model": "a" * 32,
                                       "state": "ready"})
    data = store.dataset(owner, name="Ticket triage", text=examples(count, prefix), source="Owned tickets",
                         license="Owned", purpose="training", reviewed=True)
    return model, data


def selection(model, data, method="sft", role="hub", tuning="qlora"):
    return SelectionRequest(method=method, role=role, tuning=tuning, model_id=model["id"],
                            dataset_id=data["id"], dataset_sha256=data["sha256"])


def ready_request(store, base, model, data, owner=OWNER, **choice):
    chosen = selection(model, data, **choice)
    receipt = selection_preflight(owner, chosen, store, weights=lambda row: base)
    return RunRequest(**chosen.model_dump(), preflight_sha256=receipt["preflight_sha256"])


def start(registry, store, base, owner=OWNER, **kwargs):
    model, data = owned(store, owner, **kwargs)
    return registry.start(owner, ready_request(store, base, model, data, owner), store, weights=lambda row: base)


def run_dir(processes, index=-1):
    return processes.launched[index][0]


def report(processes, index=-1, **status):
    (run_dir(processes, index) / "status.json").write_text(json.dumps(status))


def test_a_ready_check_starts_one_candidate_run_with_server_chosen_inputs(registry, store, base, processes, tmp_path):
    model, data = owned(store)
    request = ready_request(store, base, model, data)

    run = registry.start(OWNER, request, store, weights=lambda row: base)

    assert run["schema"] == "learning-run.v1" and run["state"] == "running"
    assert (run["method"], run["role"], run["tuning"]) == ("sft", "hub", "qlora")
    assert run["model"] == {"id": model["id"], "backend": "uploaded", "source_kind": "local"}
    assert run["dataset"] == {"id": data["id"], "sha256": data["sha256"], "version": 1,
                              "train_count": 10, "validation_count": 2}
    assert run["preflight_sha256"] == request.preflight_sha256
    expected_steps = math.ceil(3 * 10 / 4)
    assert run["progress"] == {"step": 0, "max_steps": expected_steps}
    assert run["metrics"] == {"train_loss": None, "validation_loss": None}
    assert (run["candidate_sha256"], run["error"], run["finished_at"]) == (None, None, None)
    assert run["serving_changed"] is False and run["promotion_enabled"] is False
    assert len(processes.launched) == 1
    # The trainer receives only server-chosen inputs; the receipt never shows a path.
    recipe = AdaptationRecipe.model_validate_json((run_dir(processes) / "recipe.json").read_text())
    assert recipe.model_path == base and recipe.output_root == run_dir(processes) / "candidate"
    assert recipe.dataset == run_dir(processes) / "dataset.jsonl"
    assert (recipe.objective, recipe.tuning, recipe.role, recipe.max_steps) == ("sft", "qlora", "hub", expected_steps)
    assert (recipe.allowed_sources, recipe.allowed_licenses) == (["Owned tickets"], ["Owned"])
    train, validation = admit_records(recipe)
    assert (len(train), len(validation)) == (10, 2)
    assert str(tmp_path) not in json.dumps(run)


def test_the_trainer_environment_carries_no_credentials(registry, store, base, processes, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "provider-credential")
    monkeypatch.setenv("INTERNAL_JOB_TOKEN", "service-credential")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    start(registry, store, base)

    environment = processes.launched[0][1]
    assert set(environment) <= set(TRAINER_ENVIRONMENT) | {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                           "PYTHONDONTWRITEBYTECODE"}
    assert "provider-credential" not in environment.values()
    assert "service-credential" not in environment.values()
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PATH"] == "/usr/bin:/bin"


def test_a_run_needs_the_exact_ready_check(registry, store, base, processes):
    model, data = owned(store)
    exact = ready_request(store, base, model, data)
    remote = store.put(OWNER, "model", {"name": "Remote", "backend": "http", "model": "x", "state": "ready"})
    foreign_model, _ = owned(store, OTHER)

    def checked(chosen):
        # A selection that is not runnable, presented with its own (blocked) check.
        digest = selection_preflight(OWNER, chosen, store, weights=lambda row: base)["preflight_sha256"]
        return RunRequest(**chosen.model_dump(), preflight_sha256=digest)

    cases = [
        (exact.model_copy(update={"preflight_sha256": "0" * 64}), "learning_check_changed"),
        (RunRequest(**selection(remote, data).model_dump(), preflight_sha256=exact.preflight_sha256),
         "learning_check_changed"),
        (checked(selection(remote, data)), "learning_run_not_ready"),
        (checked(selection(model, data, method="dpo", role="specialist")), "learning_run_not_ready"),
        (checked(selection(model, data, method="ppo", role="hub")), "learning_run_not_ready"),
        (checked(selection(model, data, tuning="full")), "learning_run_not_ready"),
        (exact.model_copy(update={"dataset_sha256": "1" * 64}), "learning_dataset_changed"),
        (exact.model_copy(update={"model_id": foreign_model["id"]}), "learning_resource_unavailable"),
    ]
    for request, code in cases:
        with pytest.raises(LearningSelectionError, match=f"^{code}$"):
            registry.start(OWNER, request, store, weights=lambda row: base)
    # Weights that disappeared after the check: the check no longer matches, and a fresh one is blocked.
    with pytest.raises(LearningSelectionError, match="^learning_check_changed$"):
        registry.start(OWNER, exact, store, weights=lambda row: None)
    gone = selection_preflight(OWNER, selection(model, data), store, weights=lambda row: None)
    assert gone["status"] == "blocked" and gone["blockers"] == ["local_weights_unavailable"]
    with pytest.raises(LearningSelectionError, match="^learning_run_not_ready$"):
        registry.start(OWNER, RunRequest(**selection(model, data).model_dump(),
                                         preflight_sha256=gone["preflight_sha256"]),
                       store, weights=lambda row: None)
    assert processes.launched == [] and registry.list(OWNER) == []


def test_only_one_run_at_a_time_on_the_host(registry, store, base, processes):
    first = start(registry, store, base)

    with pytest.raises(LearningSelectionError, match="^learning_host_busy$"):
        start(registry, store, base, owner=OTHER)
    report(processes, state="completed", step=8, max_steps=8, train_loss=0.5, validation_loss=0.7,
           candidate_sha256="a" * 64)
    processes.exit(processes.next_pid)

    second = start(registry, store, base, owner=OTHER)
    assert second["state"] == "running" and second["run_id"] != first["run_id"]


def test_progress_and_completion_come_from_the_trainer_status(registry, store, base, processes):
    run = start(registry, store, base)
    report(processes, state="running", step=3, max_steps=8, train_loss=1.25, validation_loss=None)

    [observed] = registry.list(OWNER)
    assert observed["state"] == "running" and observed["progress"] == {"step": 3, "max_steps": 8}
    assert observed["metrics"] == {"train_loss": 1.25, "validation_loss": None}

    report(processes, state="completed", step=8, max_steps=8, train_loss=0.5, validation_loss=0.75,
           candidate_sha256="c" * 64)
    processes.exit(processes.next_pid)
    [finished] = registry.list(OWNER)
    assert finished["run_id"] == run["run_id"] and finished["state"] == "completed"
    assert finished["candidate_sha256"] == "c" * 64 and finished["finished_at"] is not None
    assert finished["metrics"] == {"train_loss": 0.5, "validation_loss": 0.75}
    # A finished result is kept even if the status file disappears later.
    (run_dir(processes) / "status.json").unlink()
    assert registry.list(OWNER)[0] == finished


def test_a_completed_status_without_a_valid_candidate_digest_is_not_completion(registry, store, base, processes):
    start(registry, store, base)
    report(processes, state="completed", step=8, max_steps=8, candidate_sha256="not-a-digest")
    processes.exit(processes.next_pid)

    [observed] = registry.list(OWNER)
    assert observed["state"] == "failed" and observed["error"] == "training_failed"
    assert observed["candidate_sha256"] is None


def test_a_trainer_that_dies_without_a_result_is_interrupted(registry, store, base, processes):
    start(registry, store, base)
    report(processes, state="running", step=2, max_steps=8)
    processes.exit(processes.next_pid)

    [observed] = registry.list(OWNER)
    assert observed["state"] == "interrupted" and observed["error"] is None
    assert observed["finished_at"] is not None


@pytest.mark.parametrize("reported, shown", [
    ("insufficient_gpu_memory", "insufficient_gpu_memory"),
    ("chat_template_unsupported", "chat_template_unsupported"),
    ("Traceback: /private/path/secret", "training_failed"),
    (None, "training_failed"),
])
def test_a_failed_trainer_reports_only_a_known_error_code(registry, store, base, processes, reported, shown):
    start(registry, store, base)
    report(processes, state="failed", step=1, max_steps=8, error=reported)

    [observed] = registry.list(OWNER)
    assert observed["state"] == "failed" and observed["error"] == shown


def test_an_unreadable_status_file_is_not_trusted(registry, store, base, processes):
    start(registry, store, base)
    (run_dir(processes) / "status.json").write_text("not json")
    assert registry.list(OWNER)[0]["progress"]["step"] == 0
    (run_dir(processes) / "status.json").write_text(json.dumps({"state": "running", "step": 99, "max_steps": 8}))
    assert registry.list(OWNER)[0]["progress"] == {"step": 0, "max_steps": 8}
    processes.exit(processes.next_pid)
    assert registry.list(OWNER)[0]["state"] == "interrupted"


def test_cancel_stops_only_the_owners_active_run(registry, store, base, processes):
    run = start(registry, store, base)
    pid = processes.next_pid

    for owner, identity in ((OTHER, run["run_id"]), (OWNER, "0" * 32), (OWNER, "../escape")):
        with pytest.raises(LearningSelectionError, match="^learning_run_unavailable$"):
            registry.cancel(owner, identity)
    assert processes.stopped == []

    cancelled = registry.cancel(OWNER, run["run_id"])
    assert cancelled["state"] == "cancelled" and cancelled["finished_at"] is not None
    assert processes.stopped == [pid]
    assert registry.cancel(OWNER, run["run_id"]) == cancelled
    assert processes.stopped == [pid]


def test_a_finished_run_is_never_stopped(registry, store, base, processes):
    run = start(registry, store, base)
    report(processes, state="completed", step=8, max_steps=8, candidate_sha256="d" * 64)

    finished = registry.cancel(OWNER, run["run_id"])

    assert finished["state"] == "completed" and processes.stopped == []


def test_a_finished_record_never_changes_after_the_fact(registry, store, base, processes):
    run = start(registry, store, base)
    report(processes, state="completed", step=8, max_steps=8, train_loss=0.5, candidate_sha256="d" * 64)
    finished = registry.list(OWNER)[0]
    # A trainer writing late, or a status file replaced afterwards, cannot rewrite the outcome.
    report(processes, state="failed", step=3, max_steps=8, error="cuda_unavailable")
    processes.exit(processes.next_pid)

    assert registry.list(OWNER)[0] == finished
    assert registry.cancel(OWNER, run["run_id"]) == finished


def test_a_reused_process_id_is_never_stopped(registry, store, base, processes):
    run = start(registry, store, base)
    processes.alive[processes.next_pid] = "another-process"

    observed = registry.cancel(OWNER, run["run_id"])

    assert processes.stopped == []
    assert observed["state"] == "interrupted"


def test_listing_shows_only_the_owners_runs_newest_first_and_bounded(registry, store, base, processes):
    started = []
    for index in range(22):
        started.append(start(registry, store, base, prefix=f"batch {index} question")["run_id"])
        report(processes, state="completed", step=8, max_steps=8, candidate_sha256="e" * 64)
        processes.exit(processes.next_pid)
    other = start(registry, store, base, owner=OTHER)

    mine = registry.list(OWNER)
    assert [row["run_id"] for row in mine] == list(reversed(started))[:20]
    assert [row["run_id"] for row in registry.list(OTHER)] == [other["run_id"]]


def test_a_trainer_that_cannot_start_is_recorded_and_frees_the_host(registry, store, base, processes):
    processes.fail_launch = True

    run = start(registry, store, base)

    assert run["state"] == "failed" and run["error"] == "trainer_unavailable"
    processes.fail_launch = False
    assert start(registry, store, base, owner=OTHER)["state"] == "running"
