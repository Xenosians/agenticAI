from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import (
    Any,
)

from ldap3 import (
    BASE,
    MODIFY_ADD,
    MODIFY_DELETE,
)

from ldap3.utils.conv import (
    escape_filter_chars,
)

from pydantic import (
    BaseModel,
)

from .base import (
    DirectoryService,
)

from .ldap import (
    LdapDirectoryService,
)

from .mock import (
    MockDirectoryService,
)


class AccessMutationResult(
    BaseModel
):
    """
    Provider-neutral result for governed access mutations.

    Authorization and approval happen before this service executes.
    """

    ok: bool

    status: str

    changed: bool = False

    user_id: (
        str | None
    ) = None

    resource: (
        str | None
    ) = None

    has_access: (
        bool | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class AccessMutationService(
    ABC
):
    @abstractmethod
    def grant_access(
        self,
        user_id: str,
        resource: str,
    ) -> AccessMutationResult:
        raise NotImplementedError

    @abstractmethod
    def revoke_access(
        self,
        user_id: str,
        resource: str,
    ) -> AccessMutationResult:
        raise NotImplementedError


class MockAccessMutationService(
    AccessMutationService
):
    """
    Mutates the exact state used by MockDirectoryService reads.
    """

    def __init__(
        self,
        directory: MockDirectoryService,
    ) -> None:

        if not isinstance(
            directory,
            MockDirectoryService,
        ):

            raise TypeError(
                "MockAccessMutationService requires "
                "MockDirectoryService."
            )

        self.directory = (
            directory
        )

    def _resolve(
        self,
        user_id: str,
        resource: str,
    ) -> tuple[
        str | None,
        str | None,
        str | None,
    ]:

        if not isinstance(
            user_id,
            str,
        ):

            return (
                None,
                None,
                "user_id must be a string.",
            )

        if not isinstance(
            resource,
            str,
        ):

            return (
                None,
                None,
                "resource must be a string.",
            )

        normalized_user = (
            self.directory
            ._normalize_user(
                user_id
            )
        )

        normalized_resource = (
            self.directory
            ._normalize_resource(
                resource
            )
        )

        if not normalized_user:

            return (
                None,
                None,
                "user_id must not be empty.",
            )

        if not normalized_resource:

            return (
                None,
                None,
                "resource must not be empty.",
            )

        if (
            normalized_user
            not in self.directory.users
        ):

            return (
                normalized_user,
                normalized_resource,
                (
                    f"User '{user_id}' "
                    "was not found."
                ),
            )

        known_resources = {
            value

            for value
            in (
                self.directory
                .resource_aliases
                .values()
            )

            if value
        }

        if (
            normalized_resource
            not in known_resources
        ):

            return (
                normalized_user,
                normalized_resource,
                (
                    "No access policy is configured "
                    f"for resource '{resource}'."
                ),
            )

        return (
            normalized_user,
            normalized_resource,
            None,
        )

    def grant_access(
        self,
        user_id: str,
        resource: str,
    ) -> AccessMutationResult:

        (
            normalized_user,
            normalized_resource,
            error,
        ) = (
            self._resolve(
                user_id,
                resource,
            )
        )

        if error is not None:

            return (
                AccessMutationResult(
                    ok=False,
                    status="denied",
                    changed=False,
                    user_id=(
                        normalized_user
                    ),
                    resource=(
                        normalized_resource
                    ),
                    error=(
                        error
                    ),
                )
            )

        assert normalized_user is not None
        assert normalized_resource is not None

        key = (
            normalized_user,
            normalized_resource,
        )

        if (
            self.directory
            .access
            .get(
                key
            )
            is True
        ):

            return (
                AccessMutationResult(
                    ok=True,
                    status="executed",
                    changed=False,
                    user_id=(
                        normalized_user
                    ),
                    resource=(
                        normalized_resource
                    ),
                    has_access=True,
                    message=(
                        f"User '{normalized_user}' "
                        "already has access to "
                        f"'{normalized_resource}'."
                    ),
                )
            )

        self.directory.access[
            key
        ] = True

        return (
            AccessMutationResult(
                ok=True,
                status="executed",
                changed=True,
                user_id=(
                    normalized_user
                ),
                resource=(
                    normalized_resource
                ),
                has_access=True,
                message=(
                    f"Access to '{normalized_resource}' "
                    f"was granted to '{normalized_user}'."
                ),
            )
        )

    def revoke_access(
        self,
        user_id: str,
        resource: str,
    ) -> AccessMutationResult:

        (
            normalized_user,
            normalized_resource,
            error,
        ) = (
            self._resolve(
                user_id,
                resource,
            )
        )

        if error is not None:

            return (
                AccessMutationResult(
                    ok=False,
                    status="denied",
                    changed=False,
                    user_id=(
                        normalized_user
                    ),
                    resource=(
                        normalized_resource
                    ),
                    error=(
                        error
                    ),
                )
            )

        assert normalized_user is not None
        assert normalized_resource is not None

        key = (
            normalized_user,
            normalized_resource,
        )

        if (
            self.directory
            .access
            .get(
                key
            )
            is not True
        ):

            self.directory.access[
                key
            ] = False

            return (
                AccessMutationResult(
                    ok=True,
                    status="executed",
                    changed=False,
                    user_id=(
                        normalized_user
                    ),
                    resource=(
                        normalized_resource
                    ),
                    has_access=False,
                    message=(
                        f"User '{normalized_user}' "
                        "already does not have access to "
                        f"'{normalized_resource}'."
                    ),
                )
            )

        self.directory.access[
            key
        ] = False

        return (
            AccessMutationResult(
                ok=True,
                status="executed",
                changed=True,
                user_id=(
                    normalized_user
                ),
                resource=(
                    normalized_resource
                ),
                has_access=False,
                message=(
                    f"Access to '{normalized_resource}' "
                    f"was revoked from '{normalized_user}'."
                ),
            )
        )


class LdapAccessMutationService(
    AccessMutationService
):
    """
    Active Directory group-membership command adapter.

    Logical resource identifiers are resolved through the existing
    trusted AD_ACCESS_GROUPS configuration owned by
    LdapDirectoryService.

    Models never provide group DNs.
    """

    def __init__(
        self,
        directory: LdapDirectoryService,
    ) -> None:

        if not isinstance(
            directory,
            LdapDirectoryService,
        ):

            raise TypeError(
                "LdapAccessMutationService requires "
                "LdapDirectoryService."
            )

        self.directory = (
            directory
        )

    def _resolve_group_dn(
        self,
        connection,
        configured_group: str,
    ) -> (
        str | None
    ):

        group = (
            configured_group
            .strip()
        )

        if not group:

            return None

        # --------------------------------------------------------
        # FULL DN CONFIGURATION
        # --------------------------------------------------------

        if (
            group.lower()
            .startswith(
                "cn="
            )
            and ","
            in group
        ):

            connection.search(
                search_base=(
                    group
                ),

                search_filter=(
                    "(objectClass=group)"
                ),

                search_scope=(
                    BASE
                ),

                attributes=[
                    "member",
                ],
            )

        # --------------------------------------------------------
        # SIMPLE GROUP NAME CONFIGURATION
        # --------------------------------------------------------

        else:

            safe_group = (
                escape_filter_chars(
                    group
                )
            )

            connection.search(
                search_base=(
                    self.directory
                    .base_dn
                ),

                search_filter=(
                    "(&(objectClass=group)"
                    f"(cn={safe_group}))"
                ),

                attributes=[
                    "member",
                ],
            )

        if not connection.entries:

            return None

        return (
            connection
            .entries[
                0
            ]
            .entry_dn
        )

    @staticmethod
    def _group_member_values(
        entry: Any,
    ) -> list[str]:

        try:

            attribute = (
                entry.member
            )

            if not attribute:

                return []

            values = (
                attribute.values
            )

            if not values:

                return []

            return [
                str(
                    value
                )

                for value
                in values
            ]

        except AttributeError:

            return []

    def _group_has_member(
        self,
        connection,
        group_dn: str,
        user_dn: str,
    ) -> bool:

        connection.search(
            search_base=(
                group_dn
            ),

            search_filter=(
                "(objectClass=group)"
            ),

            search_scope=(
                BASE
            ),

            attributes=[
                "member",
            ],
        )

        if not connection.entries:

            return False

        expected = (
            user_dn
            .strip()
            .casefold()
        )

        return any(
            str(
                value
            )
            .strip()
            .casefold()
            == expected

            for value
            in (
                self._group_member_values(
                    connection
                    .entries[
                        0
                    ]
                )
            )
        )

    def _context(
        self,
        connection,
        user_id: str,
        resource: str,
    ) -> tuple[
        str,
        str,
        Any | None,
        str | None,
        str | None,
    ]:

        normalized_user = (
            self.directory
            ._normalize_user_id(
                user_id
            )
        )

        normalized_resource = (
            self.directory
            ._normalize_resource(
                resource
            )
        )

        if not normalized_user:

            return (
                normalized_user,
                normalized_resource,
                None,
                None,
                "user_id must not be empty.",
            )

        if not normalized_resource:

            return (
                normalized_user,
                normalized_resource,
                None,
                None,
                "resource must not be empty.",
            )

        configured_group = (
            self.directory
            .access_groups
            .get(
                normalized_resource
            )
        )

        if configured_group is None:

            return (
                normalized_user,
                normalized_resource,
                None,
                None,
                (
                    "No Active Directory group is "
                    "configured for resource "
                    f"'{normalized_resource}'."
                ),
            )

        user_entry = (
            self.directory
            ._find_user(
                connection,
                normalized_user,
            )
        )

        if user_entry is None:

            return (
                normalized_user,
                normalized_resource,
                None,
                None,
                (
                    f"User '{normalized_user}' "
                    "was not found."
                ),
            )

        group_dn = (
            self._resolve_group_dn(
                connection,
                configured_group,
            )
        )

        if group_dn is None:

            return (
                normalized_user,
                normalized_resource,
                user_entry,
                None,
                (
                    "Configured Active Directory group "
                    f"'{configured_group}' was not found."
                ),
            )

        return (
            normalized_user,
            normalized_resource,
            user_entry,
            group_dn,
            None,
        )

    def grant_access(
        self,
        user_id: str,
        resource: str,
    ) -> AccessMutationResult:

        connection = None

        mutation_attempted = (
            False
        )

        try:

            connection = (
                self.directory
                ._connect_write()
            )

            (
                normalized_user,
                normalized_resource,
                user_entry,
                group_dn,
                error,
            ) = (
                self._context(
                    connection,
                    user_id,
                    resource,
                )
            )

            if error is not None:

                return (
                    AccessMutationResult(
                        ok=False,
                        status="denied",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        error=(
                            error
                        ),
                    )
                )

            assert user_entry is not None
            assert group_dn is not None

            user_dn = (
                user_entry.entry_dn
            )

            if (
                self._group_has_member(
                    connection,
                    group_dn,
                    user_dn,
                )
            ):

                return (
                    AccessMutationResult(
                        ok=True,
                        status="executed",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        has_access=True,
                        message=(
                            f"User '{normalized_user}' "
                            "already has access to "
                            f"'{normalized_resource}'."
                        ),
                    )
                )

            mutation_attempted = (
                True
            )

            success = (
                connection.modify(
                    group_dn,
                    {
                        "member": [
                            (
                                MODIFY_ADD,
                                [
                                    user_dn,
                                ],
                            )
                        ],
                    },
                )
            )

            if not success:

                return (
                    AccessMutationResult(
                        ok=False,
                        status="error",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        has_access=False,
                        error=(
                            "Active Directory access grant "
                            "failed: "
                            f"{connection.result}"
                        ),
                    )
                )

            if not (
                self._group_has_member(
                    connection,
                    group_dn,
                    user_dn,
                )
            ):

                return (
                    AccessMutationResult(
                        ok=False,
                        status="unknown",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        has_access=None,
                        error=(
                            "Active Directory accepted the "
                            "membership change, but the resulting "
                            "access state could not be verified."
                        ),
                    )
                )

            return (
                AccessMutationResult(
                    ok=True,
                    status="executed",
                    changed=True,
                    user_id=(
                        normalized_user
                    ),
                    resource=(
                        normalized_resource
                    ),
                    has_access=True,
                    message=(
                        f"Access to '{normalized_resource}' "
                        f"was granted to '{normalized_user}'."
                    ),
                )
            )

        except Exception as exc:

            return (
                AccessMutationResult(
                    ok=False,
                    status=(
                        "unknown"
                        if mutation_attempted
                        else "error"
                    ),
                    changed=False,
                    user_id=(
                        (
                            user_id.strip()
                        )
                        if isinstance(
                            user_id,
                            str,
                        )
                        else None
                    ),
                    resource=(
                        (
                            resource.strip()
                        )
                        if isinstance(
                            resource,
                            str,
                        )
                        else None
                    ),
                    has_access=None,
                    error=(
                        "Active Directory access grant failed: "
                        f"{exc}"
                    ),
                )
            )

        finally:

            if connection is not None:

                connection.unbind()

    def revoke_access(
        self,
        user_id: str,
        resource: str,
    ) -> AccessMutationResult:

        connection = None

        mutation_attempted = (
            False
        )

        try:

            connection = (
                self.directory
                ._connect_write()
            )

            (
                normalized_user,
                normalized_resource,
                user_entry,
                group_dn,
                error,
            ) = (
                self._context(
                    connection,
                    user_id,
                    resource,
                )
            )

            if error is not None:

                return (
                    AccessMutationResult(
                        ok=False,
                        status="denied",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        error=(
                            error
                        ),
                    )
                )

            assert user_entry is not None
            assert group_dn is not None

            user_dn = (
                user_entry.entry_dn
            )

            if not (
                self._group_has_member(
                    connection,
                    group_dn,
                    user_dn,
                )
            ):

                return (
                    AccessMutationResult(
                        ok=True,
                        status="executed",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        has_access=False,
                        message=(
                            f"User '{normalized_user}' "
                            "already does not have access to "
                            f"'{normalized_resource}'."
                        ),
                    )
                )

            mutation_attempted = (
                True
            )

            success = (
                connection.modify(
                    group_dn,
                    {
                        "member": [
                            (
                                MODIFY_DELETE,
                                [
                                    user_dn,
                                ],
                            )
                        ],
                    },
                )
            )

            if not success:

                return (
                    AccessMutationResult(
                        ok=False,
                        status="error",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        has_access=True,
                        error=(
                            "Active Directory access revoke "
                            "failed: "
                            f"{connection.result}"
                        ),
                    )
                )

            if (
                self._group_has_member(
                    connection,
                    group_dn,
                    user_dn,
                )
            ):

                return (
                    AccessMutationResult(
                        ok=False,
                        status="unknown",
                        changed=False,
                        user_id=(
                            normalized_user
                        ),
                        resource=(
                            normalized_resource
                        ),
                        has_access=None,
                        error=(
                            "Active Directory accepted the "
                            "membership change, but the resulting "
                            "access state could not be verified."
                        ),
                    )
                )

            return (
                AccessMutationResult(
                    ok=True,
                    status="executed",
                    changed=True,
                    user_id=(
                        normalized_user
                    ),
                    resource=(
                        normalized_resource
                    ),
                    has_access=False,
                    message=(
                        f"Access to '{normalized_resource}' "
                        f"was revoked from '{normalized_user}'."
                    ),
                )
            )

        except Exception as exc:

            return (
                AccessMutationResult(
                    ok=False,
                    status=(
                        "unknown"
                        if mutation_attempted
                        else "error"
                    ),
                    changed=False,
                    user_id=(
                        (
                            user_id.strip()
                        )
                        if isinstance(
                            user_id,
                            str,
                        )
                        else None
                    ),
                    resource=(
                        (
                            resource.strip()
                        )
                        if isinstance(
                            resource,
                            str,
                        )
                        else None
                    ),
                    has_access=None,
                    error=(
                        "Active Directory access revoke failed: "
                        f"{exc}"
                    ),
                )
            )

        finally:

            if connection is not None:

                connection.unbind()


def build_access_mutation_service(
    directory: DirectoryService,
) -> AccessMutationService:

    if isinstance(
        directory,
        MockDirectoryService,
    ):

        return (
            MockAccessMutationService(
                directory
            )
        )

    if isinstance(
        directory,
        LdapDirectoryService,
    ):

        return (
            LdapAccessMutationService(
                directory
            )
        )

    raise RuntimeError(
        "Unsupported DirectoryService implementation for "
        "access mutations: "
        f"{type(directory).__name__}"
    )
