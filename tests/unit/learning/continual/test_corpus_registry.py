import json

from pathlib import Path

import pytest

from learning.continual.corpus_registry import (
    ContinualCorpusRegistry,
    CorpusRegistrySource,
    load_corpus_registry,
)


def test_registry_rejects_duplicate_source_ids():
    with pytest.raises(ValueError):
        ContinualCorpusRegistry(
            sources=[
                CorpusRegistrySource(
                    source_id="same",
                    provider="runtime",
                    target_component="hub",
                    objectives=["sft"],
                    trust="verified",
                ),
                CorpusRegistrySource(
                    source_id="same",
                    provider="runtime",
                    target_component="hub",
                    objectives=["sft"],
                    trust="verified",
                ),
            ]
        )


def test_static_dataset_cannot_be_direct_ppo_source():
    with pytest.raises(ValueError):
        CorpusRegistrySource(
            source_id="bad-ppo",
            provider="huggingface",
            dataset_id="example/example",
            target_component="developer-specialist",
            objectives=["ppo"],
            trust="curated",
            training_eligible=True,
            verified_reward=True,
        )


def test_ppo_requires_verified_runtime_reward():
    with pytest.raises(ValueError):
        CorpusRegistrySource(
            source_id="runtime-ppo",
            provider="runtime",
            target_component="developer-specialist",
            objectives=["ppo"],
            trust="verified",
            training_eligible=True,
            verified_reward=False,
        )


def test_evaluation_only_never_training_eligible():
    with pytest.raises(ValueError):
        CorpusRegistrySource(
            source_id="heldout",
            provider="local",
            local_path="/tmp/eval.jsonl",
            target_component="hub",
            objectives=["evaluation"],
            trust="verified",
            evaluation_only=True,
            training_eligible=True,
        )


def test_runtime_feedback_can_feed_multiple_objectives():
    source = CorpusRegistrySource(
        source_id="runtime-feedback",
        provider="runtime",
        target_component="dynamic",
        objectives=[
            "sft",
            "dpo",
            "ppo",
        ],
        trust="verified",
        enabled=True,
        training_eligible=True,
        verified_reward=True,
        materialization_mode="runtime",
    )

    assert source.enabled is True
    assert "sft" in source.objectives
    assert "dpo" in source.objectives
    assert "ppo" in source.objectives


def test_registry_loads_from_json(tmp_path: Path):
    path = (
        tmp_path
        / "registry.json"
    )

    path.write_text(
        json.dumps(
            {
                "schema":
                    "continual-corpus-registry.v1",

                "sources": [
                    {
                        "source_id":
                            "runtime-feedback",

                        "provider":
                            "runtime",

                        "target_component":
                            "hub",

                        "objectives": [
                            "sft"
                        ],

                        "trust":
                            "verified",

                        "enabled":
                            True,

                        "training_eligible":
                            True
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = load_corpus_registry(
        path
    )

    assert (
        len(
            registry.enabled_sources()
        )
        == 1
    )

    assert (
        registry
        .enabled_sources()[0]
        .source_id
        == "runtime-feedback"
    )