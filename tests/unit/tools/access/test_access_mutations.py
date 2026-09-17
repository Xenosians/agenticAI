from services.directory import (
    MockDirectoryService,
    build_access_mutation_service,
)

from tools.registry import (
    get_tool,
)


def test_access_mutation_catalog_requires_approval():

    for tool_name in (
        "grant_access",
        "revoke_access",
    ):

        tool = (
            get_tool(
                tool_name
            )
        )

        assert tool is not None

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
                "resource",
            ]
        )


def test_mock_grant_access_changes_read_state():

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    before = (
        directory
        .check_access(
            "jdoe",
            "admin",
        )
    )

    assert (
        before[
            "has_access"
        ]
        is False
    )

    result = (
        mutations
        .grant_access(
            "jdoe",
            "admin",
        )
    )

    assert result.ok is True
    assert result.status == "executed"
    assert result.changed is True
    assert result.has_access is True

    after = (
        directory
        .check_access(
            "jdoe",
            "admin",
        )
    )

    assert (
        after[
            "has_access"
        ]
        is True
    )


def test_mock_grant_access_is_idempotent():

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    result = (
        mutations
        .grant_access(
            "jdoe",
            "vpn",
        )
    )

    assert result.ok is True
    assert result.changed is False
    assert result.has_access is True


def test_mock_revoke_access_changes_read_state():

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    before = (
        directory
        .check_access(
            "jdoe",
            "vpn",
        )
    )

    assert (
        before[
            "has_access"
        ]
        is True
    )

    result = (
        mutations
        .revoke_access(
            "jdoe",
            "vpn",
        )
    )

    assert result.ok is True
    assert result.status == "executed"
    assert result.changed is True
    assert result.has_access is False

    after = (
        directory
        .check_access(
            "jdoe",
            "vpn",
        )
    )

    assert (
        after[
            "has_access"
        ]
        is False
    )


def test_mock_revoke_access_is_idempotent():

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    result = (
        mutations
        .revoke_access(
            "jdoe",
            "admin",
        )
    )

    assert result.ok is True
    assert result.changed is False
    assert result.has_access is False


def test_mock_access_mutation_rejects_unknown_user():

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    result = (
        mutations
        .grant_access(
            "alice",
            "vpn",
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.changed is False


def test_mock_access_mutation_rejects_unknown_resource():

    directory = (
        MockDirectoryService()
    )

    mutations = (
        build_access_mutation_service(
            directory
        )
    )

    result = (
        mutations
        .grant_access(
            "jdoe",
            "production-root",
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.changed is False
