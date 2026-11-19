"""One trainer for every specialist (owner, 18 Nov 2026): the specialist's name picks its dataset and model.

``python -m learning.cli.train_specialist <name> --device cpu|cuda [--stages sft,dpo]`` builds the job from
``learning/config/specialists/<name>.json`` and runs the portable worker on this machine, the same worker
Proxmox and the idle cycle use. Janus's own held-out gate still decides; this command never registers or serves.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from learning.cli import train_specialist


class Target:
    def __init__(self, name):
        self.name = name


def fake_environment(tmp_path, monkeypatch, *, outcome="trained", exit_code=0):
    built, ran = [], []

    def build_job(target, *, profiles, directory, stages, stamp, device, settings):
        built.append({"name": target.name, "stages": tuple(stages), "device": device, "settings": settings})
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / "job.json").write_text("{}", encoding="utf-8")
        return directory

    def run_worker(argv, **kwargs):
        ran.append({"argv": argv, "env": kwargs.get("env")})
        output = Path(argv[argv.index("learning.worker.stages") + 2])
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps({"outcome": outcome, "baseline": {"passed": 1, "total": 9},
                                                        "stages": []}), encoding="utf-8")
        return subprocess.CompletedProcess(argv, exit_code)

    monkeypatch.setattr(train_specialist, "build_specialist_job", build_job)
    monkeypatch.setattr(train_specialist, "specialist_targets",
                        lambda: {"account": Target("account"), "access": Target("access")})
    monkeypatch.setattr(train_specialist, "model_profiles", lambda: {})
    monkeypatch.setattr(train_specialist, "collect_candidate",
                        lambda run: {"adapter_directory": str(run / "final"), "final_stage": "sft"})
    monkeypatch.setattr(train_specialist.subprocess, "run", run_worker)
    return built, ran


def test_one_command_trains_any_specialist_with_the_shared_worker(tmp_path, monkeypatch, capsys):
    built, ran = fake_environment(tmp_path, monkeypatch)
    code = train_specialist.main(["account", "--device", "cpu", "--stages", "sft,dpo", "--work", str(tmp_path)])
    assert code == 0
    assert built == [{"name": "account", "stages": ("sft", "dpo"), "device": "cpu", "settings": {}}]
    worker = ran[0]
    assert worker["argv"][:3] == [sys.executable, "-m", "learning.worker.stages"]
    assert worker["env"]["CUDA_VISIBLE_DEVICES"] == "" and worker["env"]["HF_HUB_OFFLINE"] == "1"
    report = json.loads(capsys.readouterr().out)
    assert report["outcome"] == "trained" and report["candidate"].endswith("final")
    assert "never served" in report["next"]
    assert not any((tmp_path / "jobs").iterdir())         # the job folder (links to the base weights) is removed


def test_the_same_command_trains_another_specialist_by_name(tmp_path, monkeypatch, capsys):
    built, _ran = fake_environment(tmp_path, monkeypatch, outcome="no_improvement")
    assert train_specialist.main(["access", "--device", "cuda", "--work", str(tmp_path),
                                  "--settings", '{"sft": {"epochs": 2}}']) == 0
    assert built[0]["name"] == "access" and built[0]["stages"] == ("sft", "dpo")
    assert built[0]["settings"] == {"sft": {"epochs": 2}}
    report = json.loads(capsys.readouterr().out)
    assert report["outcome"] == "no_improvement" and report["candidate"] is None


def test_an_unknown_specialist_or_a_failed_worker_is_an_error(tmp_path, monkeypatch, capsys):
    fake_environment(tmp_path, monkeypatch, outcome="failed", exit_code=1)
    with pytest.raises(SystemExit):
        train_specialist.main(["nobody", "--device", "cpu", "--work", str(tmp_path)])
    assert train_specialist.main(["account", "--device", "cpu", "--work", str(tmp_path)]) == 1


def test_settings_must_be_a_json_object(tmp_path, monkeypatch):
    fake_environment(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        train_specialist.main(["account", "--device", "cpu", "--work", str(tmp_path), "--settings", "[1]"])


def test_the_old_per_specialist_trainers_are_gone():
    import importlib.util

    for name in ("learning.cli.train_account_specialist", "learning.cli.train_developer_shell_specialist"):
        assert importlib.util.find_spec(name) is None, name
    root = Path(__file__).resolve().parents[4]
    assert not (root / "scripts" / "specialist_sft_train.py").exists()
