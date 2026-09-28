import asyncio
import json

from pathlib import Path
from types import SimpleNamespace

from learning.continual.automation.config import (
    ContinualAutomationSettings,
)
from learning.continual.automation.service import (
    ContinualAutomationService,
)
from learning.paths import REPOSITORY_ROOT


class _GpuScheduler:
    active = False
    queue_depth = 0


def test_training_disabled_is_collection_only(tmp_path: Path):
    corpus_sources = tmp_path / "corpus-sources.json"
    corpus_sources.write_text(json.dumps([]), encoding="utf-8")

    runtime = SimpleNamespace(
        settings=SimpleNamespace(
            hub_model_key="hub-main",
            agents_dir=REPOSITORY_ROOT / "subagents" / "agents",
        ),
        active_jobs=set(),
        gpu_scheduler=_GpuScheduler(),
    )

    settings = ContinualAutomationSettings(
        enabled=True,
        auto_train=False,
        auto_evaluate=False,
        ppo_enabled=False,
        auto_promote=False,
        min_events_per_cycle=1,
        single_gpu_idle_only=False,
        state_db_path=tmp_path / "continuous.sqlite3",
        corpus_sources_path=corpus_sources,
    )

    service = ContinualAutomationService(
        runtime=runtime,
        settings=settings,
    )

    assert service.enqueue_trajectory(
        {
            "trajectory_id": "trajectory-collection-only",
            "job_id": "job-collection-only",
            "attempt": 1,
            "hub_model": "hub-main",
            "signals": {
                "overall_success": True,
                "had_error": False,
                "waiting_approval": False,
            },
            "steps": [],
        }
    )

    result = asyncio.run(service.run_cycle())
    assert result is None
    assert service.store.pending_event_counts()["total"] == 1
    assert service.store.latest_cycle() is None
