from subagents.core.capabilities import (
    build_capability_spec,
)


def test_git_repository_capability_exposes_configured_aliases():

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

    assert (
        "ai"
        in values
    )

    assert (
        "backend"
        in values
    )

    assert (
        "frontend"
        in values
    )

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