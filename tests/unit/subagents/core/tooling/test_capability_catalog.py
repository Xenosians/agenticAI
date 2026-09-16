import pytest

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
    build_capability_spec,
    build_router_agent_spec,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)


# ============================================================
# FAKE CAPABILITY REGISTRY
# ============================================================


def fake_argument_values(
):

    return {
        "repository": [
            "ai",
            "backend",
            "frontend",
        ]
    }


def fake_tool_lookup(
    tool_name: str,
):

    tools = {
        "alpha_read": {
            "description":
                (
                    "Read alpha state."
                ),

            "parameters": {
                "identifier": {
                    "type":
                        "str",

                    "description":
                        "Exact identifier.",
                }
            },
        },

        "beta_search": {
            "description":
                (
                    "Search beta records."
                ),

            "parameters": {
                "query": {
                    "type":
                        "str",

                    "description":
                        "Search query.",
                }
            },
        },

        "repo_status": {
            "description":
                (
                    "Inspect one configured "
                    "repository."
                ),

            "parameters": {
                "repository": {
                    "type":
                        "str",

                    "description":
                        (
                            "Configured logical "
                            "repository identifier."
                        ),
                }
            },

            "argument_values_resolver":
                fake_argument_values,
        },
    }

    return (
        tools.get(
            tool_name
        )
    )


# ============================================================
# AGENT
# ============================================================


def example_agent(
) -> AgentDefinition:

    return (
        AgentDefinition(
            name=(
                "example-specialist"
            ),

            description=(
                "Handles example operations."
            ),

            model=(
                "example-model"
            ),

            tools=[
                "alpha_read",
                "beta_search",
            ],
        )
    )


# ============================================================
# BASIC CAPABILITY
# ============================================================


def test_capability_spec_contains_argument_schema():

    spec = (
        build_capability_spec(
            "alpha_read",

            tool_lookup=(
                fake_tool_lookup
            ),

            include_arguments=True,
        )
    )

    assert (
        spec[
            "name"
        ]
        == "alpha_read"
    )

    assert (
        spec[
            "description"
        ]
        == "Read alpha state."
    )

    assert (
        spec[
            "argument_schema"
        ][
            "identifier"
        ][
            "type"
        ]
        == "str"
    )


# ============================================================
# DYNAMIC ENUM
# ============================================================


def test_capability_spec_injects_runtime_enum():

    spec = (
        build_capability_spec(
            "repo_status",

            tool_lookup=(
                fake_tool_lookup
            ),

            include_arguments=True,
        )
    )

    assert (
        spec[
            "argument_schema"
        ][
            "repository"
        ][
            "enum"
        ]
        == [
            "ai",
            "backend",
            "frontend",
        ]
    )


def test_capability_enum_does_not_mutate_registry_schema():

    tool = (
        fake_tool_lookup(
            "repo_status"
        )
    )

    assert (
        "enum"
        not in tool[
            "parameters"
        ][
            "repository"
        ]
    )

    build_capability_spec(
        "repo_status",

        tool_lookup=(
            fake_tool_lookup
        ),

        include_arguments=True,
    )

    assert (
        "enum"
        not in tool[
            "parameters"
        ][
            "repository"
        ]
    )


# ============================================================
# ROUTER VIEW
# ============================================================


def test_router_spec_omits_argument_schema():

    spec = (
        build_router_agent_spec(
            example_agent(),

            tool_lookup=(
                fake_tool_lookup
            ),
        )
    )

    assert (
        spec[
            "name"
        ]
        == "example-specialist"
    )

    assert (
        len(
            spec[
                "capabilities"
            ]
        )
        == 2
    )

    assert (
        "argument_schema"
        not in spec[
            "capabilities"
        ][
            0
        ]
    )


# ============================================================
# ALLOWLIST
# ============================================================


def test_agent_catalog_exposes_only_allowed_capabilities():

    agent = (
        AgentDefinition(
            name=(
                "restricted-specialist"
            ),

            description=(
                "Restricted."
            ),

            model=(
                "example-model"
            ),

            tools=[
                "alpha_read",
            ],
        )
    )

    catalog = (
        build_agent_capability_catalog(
            agent,

            tool_lookup=(
                fake_tool_lookup
            ),
        )
    )

    assert [
        item[
            "name"
        ]

        for item
        in catalog
    ] == [
        "alpha_read",
    ]


# ============================================================
# UNKNOWN CAPABILITY
# ============================================================


def test_unknown_capability_is_rejected():

    agent = (
        AgentDefinition(
            name=(
                "bad-specialist"
            ),

            description=(
                "Bad."
            ),

            model=(
                "example-model"
            ),

            tools=[
                "does_not_exist",
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unknown capability"
        ),
    ):

        build_agent_capability_catalog(
            agent,

            tool_lookup=(
                fake_tool_lookup
            ),
        )


# ============================================================
# DUPLICATES
# ============================================================


def test_duplicate_capability_is_rejected():

    agent = (
        AgentDefinition(
            name=(
                "duplicate-specialist"
            ),

            description=(
                "Duplicate."
            ),

            model=(
                "example-model"
            ),

            tools=[
                "alpha_read",
                "alpha_read",
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate capability"
        ),
    ):

        build_agent_capability_catalog(
            agent,

            tool_lookup=(
                fake_tool_lookup
            ),
        )