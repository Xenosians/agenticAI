from learning.paths import REPOSITORY_ROOT
from learning.continual.automation.targets import (
    build_continual_target_coverage,
)


def test_target_coverage_exposes_current_limits():
    coverage = build_continual_target_coverage(
        hub_model_key="hub-main",
        agent_directory=REPOSITORY_ROOT / "subagents" / "agents",
    )
    by_target = {
        item.target_component: item
        for item in coverage
    }

    hub = by_target["hub"]
    assert hub.candidate_training_supported
    assert hub.behavior_training_supported
    assert hub.corpus_adaptation_supported

    developer = by_target["developer-specialist"]
    assert developer.training_path == "shared_hub_adapter_corpus"
    assert developer.candidate_training_supported
    assert not developer.behavior_training_supported
    assert not developer.auto_promotion_supported

    jira = by_target["jira-specialist"]
    assert jira.model_key == "hub-main"
    assert not jira.auto_promotion_supported

    account = by_target["account-specialist"]
    assert account.training_path == "separate_model_unwired"
