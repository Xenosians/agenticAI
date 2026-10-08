import json
from pathlib import Path
import pytest
from learning.training.methods import validate_method_role
from learning.training.preference_training import train_preferences


@pytest.mark.parametrize("method,role,tuning", [("ppo","specialist","qlora"), ("dpo","hub","qlora"), ("rag","hub","full"), ("unknown","hub",None)])
def test_unsupported_combinations_fail_closed(method, role, tuning):
    with pytest.raises(ValueError): validate_method_role(method, role, tuning)


def test_preferences_never_train_without_opt_in():
    with pytest.raises(PermissionError): train_preferences(None, max_steps=1)


def test_ppo_training_is_independent_of_fixed_evaluations():
    root = Path(__file__).resolve().parents[4]
    training = [json.loads(line) for line in (root / "learning/config/ppo-training/router-training-v1.jsonl").read_text().splitlines()]
    heldout = []
    for path in (root / "learning/evaluation/evals").glob("*.jsonl"):
        heldout.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    assert not {r["case_id"] for r in training} & {r.get("case_id") for r in heldout}
    assert not {r["user_request"].strip().lower() for r in training} & {r.get("user_request", "").strip().lower() for r in heldout}
