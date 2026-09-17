from services.directory import (
    MockDirectoryService,
    build_access_mutation_service,
)

from tools.access.mcp import (
    register_access_mutation_tools,
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
            ] = function

            return function

        return decorator


def test_access_mutation_tools_are_registered():

    server = (
        FakeMCPServer()
    )

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    register_access_mutation_tools(
        server,
        mutations,
    )

    assert (
        set(
            server.functions
        )
        == {
            "grant_access",
            "revoke_access",
        }
    )


def test_grant_access_mcp_updates_shared_directory_state():

    server = (
        FakeMCPServer()
    )

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    register_access_mutation_tools(
        server,
        mutations,
    )

    result = (
        server.functions[
            "grant_access"
        ](
            user_id=(
                "jdoe"
            ),

            resource=(
                "admin"
            ),
        )
    )

    assert result.ok is True
    assert result.changed is True
    assert result.has_access is True

    state = (
        directory
        .check_access(
            "jdoe",
            "admin",
        )
    )

    assert (
        state[
            "has_access"
        ]
        is True
    )
