from ldap3 import (
    MODIFY_REPLACE,
)

from services.directory import (
    LdapAccountLifecycleService,
    MockDirectoryService,
    build_account_lifecycle_service,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    SemanticIntent,
)

from subagents.core.orchestration.semantic_guard import (
    SemanticGuard,
)

from tools.account.mcp import (
    register_account_lifecycle_tools,
)

from tools.registry import (
    get_tool,
)


class FakeMCPServer:

    def __init__(
        self,
    ) -> None:

        self.functions = {}

    def tool(
        self,
    ):

        def decorator(
            function,
        ):

            self.functions[
                function.__name__
            ] = (
                function
            )

            return (
                function
            )

        return (
            decorator
        )


def test_mock_disable_and_enable_account():

    directory = (
        MockDirectoryService()
    )

    lifecycle = (
        build_account_lifecycle_service(
            directory
        )
    )

    disabled = (
        lifecycle
        .disable_user(
            "jdoe"
        )
    )

    assert disabled.ok is True
    assert disabled.status == "executed"
    assert disabled.changed is True
    assert disabled.enabled is False

    state = (
        directory
        .account_status(
            "jdoe"
        )
    )

    assert (
        state[
            "enabled"
        ]
        is False
    )

    enabled = (
        lifecycle
        .enable_user(
            "jdoe"
        )
    )

    assert enabled.ok is True
    assert enabled.changed is True
    assert enabled.enabled is True

    state = (
        directory
        .account_status(
            "jdoe"
        )
    )

    assert (
        state[
            "enabled"
        ]
        is True
    )


def test_mock_lifecycle_is_idempotent():

    directory = (
        MockDirectoryService()
    )

    lifecycle = (
        build_account_lifecycle_service(
            directory
        )
    )

    result = (
        lifecycle
        .enable_user(
            "jdoe"
        )
    )

    assert result.ok is True
    assert result.changed is False
    assert result.enabled is True

    first = (
        lifecycle
        .disable_user(
            "jdoe"
        )
    )

    second = (
        lifecycle
        .disable_user(
            "jdoe"
        )
    )

    assert first.changed is True
    assert second.changed is False
    assert second.enabled is False


def test_mock_lifecycle_unknown_user():

    lifecycle = (
        build_account_lifecycle_service(
            MockDirectoryService()
        )
    )

    result = (
        lifecycle
        .disable_user(
            "missing-user"
        )
    )

    assert result.ok is False
    assert result.status == "not_found"
    assert result.changed is False


class FakeEntry:

    entry_dn = (
        "CN=John Doe,"
        "CN=Users,"
        "DC=itsm,"
        "DC=local"
    )


class FakeConnection:

    def __init__(
        self,
        directory,
    ) -> None:

        self.directory = (
            directory
        )

        self.result = {}

        self.unbound = (
            False
        )

    def modify(
        self,
        entry_dn,
        changes,
    ):

        assert (
            entry_dn
            == FakeEntry.entry_dn
        )

        operations = (
            changes[
                "userAccountControl"
            ]
        )

        assert (
            operations[
                0
            ][
                0
            ]
            == MODIFY_REPLACE
        )

        value = (
            operations[
                0
            ][
                1
            ][
                0
            ]
        )

        self.directory.uac = (
            int(
                value
            )
        )

        return (
            True
        )

    def unbind(
        self,
    ):

        self.unbound = (
            True
        )


class FakeLdapDirectory:

    def __init__(
        self,
        *,
        uac: int,
    ) -> None:

        self.uac = (
            uac
        )

        self.connection = (
            FakeConnection(
                self
            )
        )

    def _connect_write(
        self,
    ):

        return (
            self.connection
        )

    def _find_user(
        self,
        connection,
        user_id,
    ):

        assert (
            user_id
            == "jdoe"
        )

        return (
            FakeEntry()
        )

    def _get_raw_integer_attribute(
        self,
        entry,
        attribute_name,
        default=0,
    ):

        assert (
            attribute_name
            == "userAccountControl"
        )

        return (
            self.uac
        )


def test_ldap_disable_preserves_other_uac_flags():

    # NORMAL_ACCOUNT = 0x0200
    directory = (
        FakeLdapDirectory(
            uac=0x0200
        )
    )

    lifecycle = (
        LdapAccountLifecycleService(
            directory
        )
    )

    result = (
        lifecycle
        .disable_user(
            "jdoe"
        )
    )

    assert result.ok is True
    assert result.changed is True
    assert result.enabled is False

    assert (
        directory.uac
        == 0x0202
    )

    assert (
        directory
        .connection
        .unbound
        is True
    )


def test_ldap_enable_clears_only_disable_flag():

    directory = (
        FakeLdapDirectory(
            uac=0x0202
        )
    )

    lifecycle = (
        LdapAccountLifecycleService(
            directory
        )
    )

    result = (
        lifecycle
        .enable_user(
            "jdoe"
        )
    )

    assert result.ok is True
    assert result.changed is True
    assert result.enabled is True

    assert (
        directory.uac
        == 0x0200
    )


def test_account_lifecycle_catalog_requires_approval():

    for tool_name in [
        "enable_user",
        "disable_user",
    ]:

        tool = (
            get_tool(
                tool_name
            )
        )

        assert (
            tool
            is not None
        )

        assert (
            tool[
                "risk"
            ]
            == "medium"
        )

        assert (
            tool[
                "requires_approval"
            ]
            is True
        )

        assert (
            tool[
                "grounded_arguments"
            ]
            == [
                "user_id",
            ]
        )


def test_account_lifecycle_mcp_updates_shared_state():

    directory = (
        MockDirectoryService()
    )

    lifecycle = (
        build_account_lifecycle_service(
            directory
        )
    )

    server = (
        FakeMCPServer()
    )

    register_account_lifecycle_tools(
        server,
        lifecycle,
    )

    assert (
        set(
            server.functions
        )
        == {
            "enable_user",
            "disable_user",
        }
    )

    result = (
        server.functions[
            "disable_user"
        ](
            user_id=(
                "jdoe"
            )
        )
    )

    assert result.ok is True
    assert result.changed is True
    assert result.enabled is False

    state = (
        directory
        .account_status(
            "jdoe"
        )
    )

    assert (
        state[
            "enabled"
        ]
        is False
    )


def test_semantic_guard_blocks_wrong_account_identity():

    agent = (
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Account specialist"
            ),

            model=(
                "test-model"
            ),

            tools=[
                "enable_user",
                "disable_user",
            ],
        )
    )

    intent = (
        SemanticIntent(
            summary=(
                "Disable jdoe."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "disable_user",
            ],

            forbidden_tools=[
                "enable_user",
            ],

            allowed_arguments={
                "user_id": [
                    "jdoe",
                ],
            },

            forbidden_arguments={
                "user_id": [
                    "alice",
                ],
            },

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        SemanticGuard()
        .evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "disable_user"
            ),

            arguments={
                "user_id":
                    "alice",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_argument_forbidden"
    )


def test_semantic_guard_blocks_enable_when_disable_requested():

    agent = (
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Account specialist"
            ),

            model=(
                "test-model"
            ),

            tools=[
                "enable_user",
                "disable_user",
            ],
        )
    )

    intent = (
        SemanticIntent(
            summary=(
                "Disable jdoe."
            ),

            effect=(
                "mutation"
            ),

            allowed_tools=[
                "disable_user",
            ],

            forbidden_tools=[
                "enable_user",
            ],

            allowed_arguments={
                "user_id": [
                    "jdoe",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )

    decision = (
        SemanticGuard()
        .evaluate(
            agent=(
                agent
            ),

            intent=(
                intent
            ),

            tool_name=(
                "enable_user"
            ),

            arguments={
                "user_id":
                    "jdoe",
            },
        )
    )

    assert (
        decision.allowed
        is False
    )

    assert (
        decision.decision_code
        == "semantic_tool_forbidden"
    )
