"""The networking specialist proposes governed tailnet changes, so it runs on the model that proposes them.

Measured on its own held-out cases (proposal only, nothing executed, 18 November 2026):
networking-base (untrained Qwen2.5 0.5B) 6/29 and it answered the live grant request with the
routing context instead of a tool call; hub-main 21/29 with both grant cases right.
"""
from pathlib import Path

from learning.training.specialist_sft import load_training_config
from subagents.core.definitions.loader import load_agent_definition

AGENT = Path("subagents/agents/networking-specialist.md")


def test_the_networking_specialist_runs_on_the_model_that_proposes_governed_changes():
    agent = load_agent_definition(AGENT)
    assert agent.model == "hub-main"


def test_training_starts_from_the_model_the_specialist_runs_on():
    config = load_training_config(Path("learning/config/specialists/networking.json"))
    assert config.base_model_key == load_agent_definition(AGENT).model
