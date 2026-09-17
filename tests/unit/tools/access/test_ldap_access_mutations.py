from ldap3 import (
    MODIFY_ADD,
    MODIFY_DELETE,
)

from services.directory import (
    LdapAccessMutationService,
    LdapDirectoryService,
)


GROUP_DN = (
    "CN=VPN Users,"
    "OU=Groups,"
    "DC=itsm,"
    "DC=local"
)

USER_DN = (
    "CN=John Doe,"
    "CN=Users,"
    "DC=itsm,"
    "DC=local"
)


class FakeAttribute:

    def __init__(
        self,
        values,
    ) -> None:

        self.values = (
            list(
                values
            )
        )

    def __bool__(
        self,
    ) -> bool:

        return bool(
            self.values
        )


class FakeGroupEntry:

    def __init__(
        self,
        members,
    ) -> None:

        self.entry_dn = (
            GROUP_DN
        )

        self.member = (
            FakeAttribute(
                members
            )
        )


class FakeUserEntry:

    def __init__(
        self,
    ) -> None:

        self.entry_dn = (
            USER_DN
        )


class FakeConnection:

    def __init__(
        self,
        members=None,
    ) -> None:

        self.members = set(
            members
            or []
        )

        self.entries = []

        self.result = {}

        self.unbound = False

    def search(
        self,
        *,
        search_base,
        search_filter,
        attributes,
        search_scope=None,
    ):

        if (
            search_base
            == GROUP_DN
        ):

            self.entries = [
                FakeGroupEntry(
                    self.members
                ),
            ]

            return True

        if (
            search_base
            == (
                "DC=itsm,"
                "DC=local"
            )
        ):

            self.entries = [
                FakeGroupEntry(
                    self.members
                ),
            ]

            return True

        self.entries = []

        return True

    def modify(
        self,
        group_dn,
        changes,
    ):

        assert (
            group_dn
            == GROUP_DN
        )

        operations = (
            changes[
                "member"
            ]
        )

        operation = (
            operations[
                0
            ][
                0
            ]
        )

        values = (
            operations[
                0
            ][
                1
            ]
        )

        assert values == [
            USER_DN,
        ]

        if (
            operation
            == MODIFY_ADD
        ):

            self.members.add(
                USER_DN
            )

            return True

        if (
            operation
            == MODIFY_DELETE
        ):

            self.members.discard(
                USER_DN
            )

            return True

        raise AssertionError(
            "Unexpected LDAP modify operation."
        )

    def unbind(
        self,
    ):

        self.unbound = True


class FakeLdapDirectory(
    LdapDirectoryService
):

    def __init__(
        self,
        *,
        members=None,
    ) -> None:

        self.base_dn = (
            "DC=itsm,"
            "DC=local"
        )

        self.access_groups = {
            "vpn":
                "VPN Users",
        }

        self.connection = (
            FakeConnection(
                members=(
                    members
                )
            )
        )

    @staticmethod
    def _normalize_user_id(
        user_id: str,
    ) -> str:

        return (
            user_id.strip()
        )

    @staticmethod
    def _normalize_resource(
        resource: str,
    ) -> str:

        value = (
            resource
            .strip()
            .lower()
        )

        aliases = {
            "vpn":
                "vpn",

            "vpn access":
                "vpn",
        }

        return (
            aliases.get(
                value,
                value,
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
        user_id: str,
    ):

        assert (
            user_id
            == "jdoe"
        )

        return (
            FakeUserEntry()
        )


def test_ldap_grant_access_adds_group_membership():

    directory = (
        FakeLdapDirectory()
    )

    service = (
        LdapAccessMutationService(
            directory
        )
    )

    result = (
        service
        .grant_access(
            "jdoe",
            "VPN",
        )
    )

    assert result.ok is True
    assert result.status == "executed"
    assert result.changed is True
    assert result.has_access is True

    assert (
        USER_DN
        in directory
        .connection
        .members
    )

    assert (
        directory
        .connection
        .unbound
        is True
    )


def test_ldap_revoke_access_removes_group_membership():

    directory = (
        FakeLdapDirectory(
            members=[
                USER_DN,
            ]
        )
    )

    service = (
        LdapAccessMutationService(
            directory
        )
    )

    result = (
        service
        .revoke_access(
            "jdoe",
            "VPN",
        )
    )

    assert result.ok is True
    assert result.status == "executed"
    assert result.changed is True
    assert result.has_access is False

    assert (
        USER_DN
        not in directory
        .connection
        .members
    )
