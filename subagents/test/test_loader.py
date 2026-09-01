from pathlib import Path

import pytest

from subagents.core.loader import load_agent_definition

from subagents.core.loader import (
    load_agent_definition,
    load_agent_directory,  
)


def test_load_agent_definition(tmp_path: Path):
    agent_file = tmp_path / "test-agent.md"

    agent_file.write_text(
        """---
        name: account-specialist
        description: Handles account-related requests.
        tools:
        - account_status
        model: local-qwen
        max_steps: 3
        ---

        You are an account specialist.
        """,
        encoding="utf-8",
    )

    agent = load_agent_definition(agent_file)

    assert agent.name == "account-specialist"
    assert agent.description == "Handles account-related requests."
    assert agent.tools == ["account_status"]
    assert agent.model == "local-qwen"
    assert agent.max_steps == 3
    assert agent.system_prompt == "You are an account specialist."


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_agent_definition("does-not-exist.md")


def test_missing_front_matter_raises_error(tmp_path: Path):
    agent_file = tmp_path / "bad-agent.md"

    agent_file.write_text(
        "You are an invalid agent definition.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_agent_definition(agent_file)
        
def test_load_real_account_specialist():
    agent = load_agent_definition(
        "subagents/agents/account-specialist.md"
    )

    assert agent.name == "account-specialist"
    assert "account_status" in agent.tools
    assert "unlock_user" in agent.tools
    assert "reset_password" in agent.tools
    assert agent.model == "local-qwen"
    assert agent.max_steps == 3
    assert "ITSM account specialist" in agent.system_prompt

def test_load_agent_directory():
    agents = load_agent_directory(
        "subagents/agents"
    )

    assert len(agents) >= 1

    names = [
        agent.name
        for agent in agents
    ]

    assert "account-specialist" in names


def test_missing_agent_directory_raises_error():
    with pytest.raises(FileNotFoundError):
        load_agent_directory(
            "does-not-exist"
        )
        
def test_load_real_access_specialist():
    agent = load_agent_definition(
        "subagents/agents/access-specialist.md"
    )

    assert agent.name == "access-specialist"
    assert agent.tools == ["check_access"]
    assert agent.model == "local-qwen"
    assert agent.max_steps == 3
    assert "access-management specialist" in agent.system_prompt
    
def test_agent_directory_contains_expected_specialists():
    agents = load_agent_directory(
        "subagents/agents"
    )

    names = {
        agent.name
        for agent in agents
    }

    assert "account-specialist" in names
    assert "access-specialist" in names