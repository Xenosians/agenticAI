from subagents.core.loader import (
    load_agent_directory,
)

from subagents.core.tool_prompt import (
    build_worker_system_prompt,
)


# ============================================================
# HELPERS
# ============================================================


def agents_by_name(
):

    agents = (
        load_agent_directory(
            "subagents/agents"
        )
    )

    return {
        agent.name:
            agent

        for agent
        in agents
    }


# ============================================================
# RAW ROLE PROMPTS
# ============================================================


def test_role_prompts_do_not_hardcode_capability_names():

    agents = (
        agents_by_name()
    )

    names = [
        "developer-specialist",
        "ticket-specialist",
        "account-specialist",
        "access-specialist",
    ]

    for name in names:

        agent = (
            agents[
                name
            ]
        )

        raw_prompt = (
            agent.system_prompt
        )

        for tool_name in (
            agent.tools
        ):

            assert (
                tool_name
                not in raw_prompt
            ), (
                f"{name} hardcodes capability "
                f"'{tool_name}' in its role prompt."
            )


# ============================================================
# RUNTIME CAPABILITY INJECTION
# ============================================================


def test_runtime_prompt_contains_allowed_capabilities():

    agents = (
        agents_by_name()
    )

    for agent in (
        agents.values()
    ):

        prompt = (
            build_worker_system_prompt(
                agent
            )
        )

        for tool_name in (
            agent.tools
        ):

            assert (
                tool_name
                in prompt
            ), (
                f"Runtime prompt for "
                f"{agent.name} does not expose "
                f"capability '{tool_name}'."
            )


# ============================================================
# CAPABILITY BOUNDARY
# ============================================================


def test_worker_prompt_does_not_expose_other_agent_capabilities():

    agents = (
        agents_by_name()
    )

    ticket = (
        agents[
            "ticket-specialist"
        ]
    )

    developer = (
        agents[
            "developer-specialist"
        ]
    )

    ticket_prompt = (
        build_worker_system_prompt(
            ticket
        )
    )

    for tool_name in (
        developer.tools
    ):

        assert (
            tool_name
            not in ticket_prompt
        ), (
            "Ticket specialist runtime prompt "
            "unexpectedly exposes developer "
            f"capability '{tool_name}'."
        )


def test_account_prompt_does_not_expose_access_capability():

    agents = (
        agents_by_name()
    )

    account = (
        agents[
            "account-specialist"
        ]
    )

    access = (
        agents[
            "access-specialist"
        ]
    )

    prompt = (
        build_worker_system_prompt(
            account
        )
    )

    for tool_name in (
        access.tools
    ):

        assert (
            tool_name
            not in prompt
        ), (
            "Account specialist runtime prompt "
            "unexpectedly exposes access "
            f"capability '{tool_name}'."
        )


# ============================================================
# GENERIC PROTOCOL
# ============================================================


def test_generated_worker_prompt_uses_generic_capability_protocol():

    agents = (
        agents_by_name()
    )

    prompt = (
        build_worker_system_prompt(
            agents[
                "ticket-specialist"
            ]
        )
    )

    # --------------------------------------------------------
    # Generic capability catalog is present.
    # --------------------------------------------------------

    assert (
        "AVAILABLE CAPABILITIES:"
        in prompt
    )

    # --------------------------------------------------------
    # Runtime argument schemas are exposed.
    # --------------------------------------------------------

    assert (
        "argument_schema"
        in prompt
    )

    # --------------------------------------------------------
    # Original user request remains authoritative.
    # --------------------------------------------------------

    assert (
        "ORIGINAL USER REQUEST"
        in prompt
    )

    # --------------------------------------------------------
    # Router metadata must remain advisory rather than becoming
    # an authorization or grounding source.
    # --------------------------------------------------------

    assert (
        "Routing metadata is advisory"
        in prompt
    )

    # --------------------------------------------------------
    # Output stays constrained to the structured protocol.
    # --------------------------------------------------------

    assert (
        "RETURN ONLY THE JSON ARRAY"
        in prompt
    )


# ============================================================
# BOUNDED ARGUMENT PROTOCOL
# ============================================================


def test_generated_worker_prompt_understands_bounded_enum_values():

    agents = (
        agents_by_name()
    )

    prompt = (
        build_worker_system_prompt(
            agents[
                "developer-specialist"
            ]
        )
    )

    # The protocol must understand the generic enum mechanism.
    #
    # This is deliberately not Git-specific.
    assert (
        '"enum"'
        in prompt
    )

    assert (
        "exact value from that enum"
        in prompt
    )

    assert (
        "Never invent an enum value"
        in prompt
    )