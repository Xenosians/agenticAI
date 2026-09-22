import asyncio

from subagents.core.tooling.gateway import (
    ToolGateway,
    identifier_appears_in_request,
    validate_grounded_arguments,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)


ACCOUNT_AGENT = AgentDefinition(
    name="account-specialist",
    description="Account worker",
    tools=[
        "account_status",
        "unlock_user",
        "reset_password",
    ],
    model="test-model",
)


TOOLS = {
    "account_status": {
        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "user_id",
        ],
    },

    "unlock_user": {
        "risk":
            "high",

        "requires_approval":
            True,

        "grounded_arguments": [
            "user_id",
        ],
    },
}


def fake_get_tool(
    name,
):
    return (
        TOOLS.get(
            name
        )
    )


def fake_create_approval(
    tool_name,
    arguments,
    risk=None,
):
    return {
        "id":
            "approval-001",

        "risk": (
            risk
            if risk is not None
            else "high"
        ),

        "tool":
            tool_name,

        "arguments":
            arguments,
    }


class FakeMCP:
    def __init__(
        self,
    ):

        self.calls = []

    async def call_tool(
        self,
        tool_name,
        arguments,
    ):

        self.calls.append(
            (
                tool_name,
                arguments,
            )
        )

        return {
            "ok":
                True,

            "status":
                "success",

            "user_id":
                arguments.get(
                    "user_id"
                ),

            "enabled":
                True,

            "locked":
                False,
        }


def build_gateway():

    mcp = (
        FakeMCP()
    )

    gateway = (
        ToolGateway(
            tool_lookup=(
                fake_get_tool
            ),

            approval_creator=(
                fake_create_approval
            ),

            mcp=(
                mcp
            ),
        )
    )

    return (
        gateway,
        mcp,
    )


def test_identifier_guard_accepts_exact_identifier(
):

    assert (
        identifier_appears_in_request(
            "jdoe",
            "Is jdoe locked?",
        )
    )


def test_identifier_guard_rejects_invented_identifier(
):

    assert not (
        identifier_appears_in_request(
            "jsmith",
            "Is jdoe locked?",
        )
    )


def test_identifier_guard_rejects_empty_identifier(
):

    assert not (
        identifier_appears_in_request(
            "",
            "Can jdoe currently sign in?",
        )
    )

    assert not (
        identifier_appears_in_request(
            "   ",
            "Can jdoe currently sign in?",
        )
    )


def test_agent_cannot_use_unlisted_tool(
):

    (
        gateway,
        mcp,
    ) = (
        build_gateway()
    )

    result = asyncio.run(
        gateway.execute(
            agent=(
                ACCOUNT_AGENT
            ),

            user_input=(
                "Do something"
            ),

            tool_name=(
                "check_access"
            ),

            arguments={},
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "status"
        ]
        == "denied"
    )

    assert (
        result[
            "decision_code"
        ]
        == "agent_tool_not_allowed"
    )

    assert (
        mcp.calls
        == []
    )


def test_identifier_mismatch_blocks_execution(
):

    (
        gateway,
        mcp,
    ) = (
        build_gateway()
    )

    result = asyncio.run(
        gateway.execute(
            agent=(
                ACCOUNT_AGENT
            ),

            user_input=(
                "Is jdoe locked?"
            ),

            tool_name=(
                "account_status"
            ),

            arguments={
                "user_id":
                    "jsmith",
            },
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "status"
        ]
        == "denied"
    )

    assert (
        result[
            "decision_code"
        ]
        == "grounding_failed"
    )

    assert (
        mcp.calls
        == []
    )


def test_read_operation_reaches_mcp(
):

    (
        gateway,
        mcp,
    ) = (
        build_gateway()
    )

    result = asyncio.run(
        gateway.execute(
            agent=(
                ACCOUNT_AGENT
            ),

            user_input=(
                "Is jdoe locked?"
            ),

            tool_name=(
                "account_status"
            ),

            arguments={
                "user_id":
                    "jdoe",
            },
        )
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "status"
        ]
        == "success"
    )

    assert (
        result[
            "decision_code"
        ]
        == "success"
    )

    assert (
        mcp.calls
        == [
            (
                "account_status",
                {
                    "user_id":
                        "jdoe",
                },
            )
        ]
    )


def test_mutation_requires_approval_without_mcp(
):

    (
        gateway,
        mcp,
    ) = (
        build_gateway()
    )

    result = asyncio.run(
        gateway.execute(
            agent=(
                ACCOUNT_AGENT
            ),

            user_input=(
                "Unlock jdoe"
            ),

            tool_name=(
                "unlock_user"
            ),

            arguments={
                "user_id":
                    "jdoe",
            },
        )
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "status"
        ]
        == "approval_required"
    )

    assert (
        result[
            "decision_code"
        ]
        == "approval_required"
    )

    assert (
        result[
            "approval_id"
        ]
        == "approval-001"
    )

    assert (
        result[
            "risk"
        ]
        == "high"
    )

    # Critical security assertion:
    # mutation must NOT execute yet.
    assert (
        mcp.calls
        == []
    )


def test_resource_guard_accepts_explicit_resource(
):

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Does jdoe have VPN access?"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "VPN",
            },

            grounded_arguments=[
                "user_id",
                "resource",
            ],
        )
    )

    assert (
        valid
        is True
    )

    assert (
        error
        is None
    )


def test_resource_guard_rejects_invented_resource(
):

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Does jdoe have Finance access?"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "VPN",
            },

            grounded_arguments=[
                "user_id",
                "resource",
            ],
        )
    )

    assert (
        valid
        is False
    )

    assert (
        error
        is not None
    )

    assert (
        "resource"
        in error
    )

    assert (
        "VPN"
        in error
    )


def test_resource_guard_rejects_empty_resource(
):

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Can jdoe currently sign in?"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "",
            },

            grounded_arguments=[
                "user_id",
                "resource",
            ],
        )
    )

    assert (
        valid
        is False
    )

    assert (
        error
        == "resource must not be empty."
    )


def test_resource_guard_rejects_whitespace_resource(
):

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Can jdoe currently sign in?"
            ),

            arguments={
                "user_id":
                    "jdoe",

                "resource":
                    "   ",
            },

            grounded_arguments=[
                "user_id",
                "resource",
            ],
        )
    )

    assert (
        valid
        is False
    )

    assert (
        error
        == "resource must not be empty."
    )


def test_missing_optional_grounded_argument_is_not_forced(
):
    """
    Grounding and requiredness are intentionally separate concepts.

    Some capabilities have optional grounded identifiers. Absence is
    therefore allowed here; capability/schema validation may impose a
    stronger requirement elsewhere.
    """

    (
        valid,
        error,
    ) = (
        validate_grounded_arguments(
            user_input=(
                "Show me recent tickets"
            ),

            arguments={},

            grounded_arguments=[
                "project_key",
            ],
        )
    )

    assert (
        valid
        is True
    )

    assert (
        error
        is None
    )


def test_policy_may_bind_declared_trusted_execution_arguments():

    captured_approvals = []

    def create_approval(
        tool_name,
        arguments,
        risk=None,
    ):

        captured_approvals.append(
            {
                "tool":
                    tool_name,

                "arguments":
                    arguments,

                "risk":
                    risk,
            }
        )

        return {
            "id":
                "approval-bound",

            "tool":
                tool_name,

            "arguments":
                arguments,

            "risk":
                risk,
        }

    def lookup(
        name,
    ):

        if (
            name
            != "bound_mutation"
        ):

            return None

        return {
            "risk":
                "high",

            "requires_approval":
                True,

            "grounded_arguments": [
                "target",
            ],

            "trusted_policy_arguments": [
                "expected_version",
            ],

            "policy_resolver":
                lambda target: {
                    "ok":
                        True,

                    "status":
                        "allowed",

                    "risk":
                        "high",

                    "requires_approval":
                        True,

                    "execution_arguments": {
                        "target":
                            target,

                        "expected_version":
                            "version-001",
                    },
                },
        }

    agent = (
        AgentDefinition(
            name="test-agent",
            description="test",
            tools=[
                "bound_mutation",
            ],
            model="test-model",
        )
    )

    mcp = (
        FakeMCP()
    )

    gateway = (
        ToolGateway(
            tool_lookup=(
                lookup
            ),

            approval_creator=(
                create_approval
            ),

            mcp=(
                mcp
            ),
        )
    )

    result = (
        asyncio.run(
            gateway.execute(
                agent=(
                    agent
                ),

                user_input=(
                    "Change alpha."
                ),

                tool_name=(
                    "bound_mutation"
                ),

                arguments={
                    "target":
                        "alpha",
                },
            )
        )
    )

    assert (
        result[
            "status"
        ]
        == "approval_required"
    )

    assert (
        result[
            "approval_arguments"
        ]
        == {
            "target":
                "alpha",

            "expected_version":
                "version-001",
        }
    )

    assert (
        captured_approvals
        == [
            {
                "tool":
                    "bound_mutation",

                "arguments": {
                    "target":
                        "alpha",

                    "expected_version":
                        "version-001",
                },

                "risk":
                    "high",
            }
        ]
    )

    assert (
        mcp.calls
        == []
    )


def test_policy_cannot_change_original_execution_argument():

    def lookup(
        name,
    ):

        if (
            name
            != "bound_mutation"
        ):

            return None

        return {
            "risk":
                "high",

            "requires_approval":
                True,

            "grounded_arguments": [
                "target",
            ],

            "trusted_policy_arguments": [
                "expected_version",
            ],

            "policy_resolver":
                lambda target: {
                    "ok":
                        True,

                    "status":
                        "allowed",

                    "risk":
                        "high",

                    "requires_approval":
                        True,

                    "execution_arguments": {
                        "target":
                            "different-target",

                        "expected_version":
                            "version-001",
                    },
                },
        }

    agent = (
        AgentDefinition(
            name="test-agent",
            description="test",
            tools=[
                "bound_mutation",
            ],
            model="test-model",
        )
    )

    mcp = (
        FakeMCP()
    )

    gateway = (
        ToolGateway(
            tool_lookup=(
                lookup
            ),

            approval_creator=(
                fake_create_approval
            ),

            mcp=(
                mcp
            ),
        )
    )

    result = (
        asyncio.run(
            gateway.execute(
                agent=(
                    agent
                ),

                user_input=(
                    "Change alpha."
                ),

                tool_name=(
                    "bound_mutation"
                ),

                arguments={
                    "target":
                        "alpha",
                },
            )
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "status"
        ]
        == "error"
    )

    assert (
        result[
            "decision_code"
        ]
        == "policy_execution_arguments_invalid"
    )

    assert (
        "changed original argument"
        in result[
            "error"
        ]
    )

    assert (
        mcp.calls
        == []
    )


def test_policy_cannot_add_undeclared_execution_argument():

    def lookup(
        name,
    ):

        if (
            name
            != "bound_mutation"
        ):

            return None

        return {
            "risk":
                "high",

            "requires_approval":
                True,

            "grounded_arguments": [
                "target",
            ],

            "trusted_policy_arguments":
                [],

            "policy_resolver":
                lambda target: {
                    "ok":
                        True,

                    "status":
                        "allowed",

                    "risk":
                        "high",

                    "requires_approval":
                        True,

                    "execution_arguments": {
                        "target":
                            target,

                        "unexpected_snapshot":
                            "nope",
                    },
                },
        }

    agent = (
        AgentDefinition(
            name="test-agent",
            description="test",
            tools=[
                "bound_mutation",
            ],
            model="test-model",
        )
    )

    mcp = (
        FakeMCP()
    )

    gateway = (
        ToolGateway(
            tool_lookup=(
                lookup
            ),

            approval_creator=(
                fake_create_approval
            ),

            mcp=(
                mcp
            ),
        )
    )

    result = (
        asyncio.run(
            gateway.execute(
                agent=(
                    agent
                ),

                user_input=(
                    "Change alpha."
                ),

                tool_name=(
                    "bound_mutation"
                ),

                arguments={
                    "target":
                        "alpha",
                },
            )
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "decision_code"
        ]
        == "policy_execution_arguments_invalid"
    )

    assert (
        "undeclared trusted execution arguments"
        in result[
            "error"
        ]
    )

    assert (
        mcp.calls
        == []
    )
