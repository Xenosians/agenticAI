from learning.paths import REPOSITORY_ROOT
from learning.continual.automation.targets import (
    build_continual_target_coverage,
)


def test_target_coverage_exposes_current_limits():
    coverage = build_continual_target_coverage(
        hub_model_key="hub-main",
        agent_directory=(
            REPOSITORY_ROOT
            / "subagents"
            / "agents"
        ),
    )

    by_target = {
        item.target_component:
            item

        for item
        in coverage
    }

    hub = (
        by_target[
            "hub"
        ]
    )

    assert (
        hub.model_key
        == "hub-main"
    )

    assert (
        hub.training_path
        == "phase5_hub"
    )

    assert (
        hub.candidate_training_supported
    )

    assert (
        hub.behavior_training_supported
    )

    assert (
        hub.corpus_adaptation_supported
    )


    # Promoted specialists no longer share the Hub logical model.
    # Their current continual target-specific training path remains
    # deliberately unwired until an explicit specialist trainer is
    # connected.

    developer = (
        by_target[
            "developer-specialist"
        ]
    )

    assert (
        developer.model_key
        != hub.model_key
    )

    assert (
        developer.training_path
        == "separate_model_unwired"
    )

    assert not (
        developer.candidate_training_supported
    )

    assert not (
        developer.behavior_training_supported
    )

    assert not (
        developer.auto_promotion_supported
    )


    jira = (
        by_target[
            "jira-specialist"
        ]
    )

    assert (
        jira.model_key
        != hub.model_key
    )

    assert (
        jira.training_path
        == "separate_model_unwired"
    )

    assert not (
        jira.auto_promotion_supported
    )


    account = (
        by_target[
            "account-specialist"
        ]
    )

    assert (
        account.model_key
        != hub.model_key
    )

    assert (
        account.training_path
        == "separate_model_unwired"
    )
