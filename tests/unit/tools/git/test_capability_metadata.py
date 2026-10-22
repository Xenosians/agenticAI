from subagents.core.tooling.capabilities import (
    build_capability_spec,
)
from config import Settings
from services import git_repositories
import pytest


@pytest.mark.parametrize("aliases", [("ai", "backend", "frontend"), ("code", "service")])
def test_git_repository_capability_exposes_configured_aliases(monkeypatch, tmp_path, aliases):
    configured = {name: tmp_path / name for name in aliases}
    settings = Settings(_env_file=None, git_repositories=configured, git_default_repository=aliases[0])
    monkeypatch.setattr(git_repositories, "get_settings", lambda: settings)
    monkeypatch.setattr(git_repositories, "workspace_root", lambda: tmp_path)

    spec = (
        build_capability_spec(
            "workspace_git_log"
        )
    )

    repository_schema = (
        spec[
            "argument_schema"
        ][
            "repository"
        ]
    )

    assert (
        "enum"
        in repository_schema
    )

    values = (
        repository_schema[
            "enum"
        ]
    )

    assert set(values) == set(aliases)

    # Physical workspace paths must never become model-facing
    # repository values.
    for value in values:

        assert (
            "/"
            not in value
        )

        assert (
            "\\"
            not in value
        )
