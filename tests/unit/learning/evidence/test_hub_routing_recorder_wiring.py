from pathlib import Path

import learning.evidence.recorder as recorder_module

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from subagents.core.definitions.types import (
    HubResult,
)


def test_recorder_invokes_hub_provenance_persistence(
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    def fake_record_current_hub_routing_ledger(
        *,
        trajectory_id,
        job_id,
        job_attempt,
    ):
        calls.append(
            {
                "trajectory_id": trajectory_id,
                "job_id": job_id,
                "job_attempt": job_attempt,
            }
        )
        return 2

    monkeypatch.setattr(
        recorder_module,
        "record_current_hub_routing_ledger",
        fake_record_current_hub_routing_ledger,
    )

    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectories.jsonl",
        enabled=True,
        hub_model="hub-main",
    )

    payload = recorder.record(
        job_id="job-1",
        attempt=3,
        result=HubResult(
            status="success",
            user_request="Hello",
            routes=[],
            results=[],
            answer="Hello.",
        ),
    )

    assert payload is not None
    assert len(calls) == 1
    assert calls[0]["job_id"] == "job-1"
    assert calls[0]["job_attempt"] == 3
    assert calls[0]["trajectory_id"] == payload["trajectory_id"]
