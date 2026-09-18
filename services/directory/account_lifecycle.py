from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from ldap3 import (
    MODIFY_REPLACE,
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


ACCOUNT_DISABLE_FLAG = (
    0x0002
)


class AccountLifecycleResult(
    BaseModel
):
    ok: bool

    status: str

    changed: bool = False

    user_id: (
        str | None
    ) = None

    enabled: (
        bool | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


class AccountLifecycleService(
    ABC
):
    """
    Provider-neutral account enable / disable command boundary.

    Authorization and approval happen before this service is called.
    """

    @abstractmethod
    def enable_user(
        self,
        user_id: str,
    ) -> AccountLifecycleResult:
        raise NotImplementedError

    @abstractmethod
    def disable_user(
        self,
        user_id: str,
    ) -> AccountLifecycleResult:
        raise NotImplementedError


class MockAccountLifecycleService(
    AccountLifecycleService
):

    def __init__(
        self,
        directory: MockDirectoryService,
    ) -> None:

        self.directory = (
            directory
        )

    def _set_enabled(
        self,
        user_id: str,
        *,
        enabled: bool,
    ) -> AccountLifecycleResult:

        if not isinstance(
            user_id,
            str,
        ):

            return (
                AccountLifecycleResult(
                    ok=False,

                    status="denied",

                    enabled=None,

                    error=(
                        "user_id must be a string."
                    ),
                )
            )

        normalized_user = (
            self.directory
            ._normalize_user(
                user_id
            )
        )

        if not normalized_user:

            return (
                AccountLifecycleResult(
                    ok=False,

                    status="denied",

                    user_id=(
                        normalized_user
                    ),

                    enabled=None,

                    error=(
                        "user_id must not be empty."
                    ),
                )
            )

        user = (
            self.directory
            .users
            .get(
                normalized_user
            )
        )

        if user is None:

            return (
                AccountLifecycleResult(
                    ok=False,

                    status="not_found",

                    user_id=(
                        normalized_user
                    ),

                    enabled=None,

                    error=(
                        f"User '{normalized_user}' "
                        "was not found."
                    ),
                )
            )

        current_enabled = (
            bool(
                user[
                    "enabled"
                ]
            )
        )

        if (
            current_enabled
            == enabled
        ):

            state = (
                "enabled"
                if enabled
                else "disabled"
            )

            return (
                AccountLifecycleResult(
                    ok=True,

                    status="executed",

                    changed=False,

                    user_id=(
                        normalized_user
                    ),

                    enabled=(
                        enabled
                    ),

                    message=(
                        f"User '{normalized_user}' "
                        f"was already {state}."
                    ),
                )
            )

        user[
            "enabled"
        ] = (
            enabled
        )

        state = (
            "enabled"
            if enabled
            else "disabled"
        )

        return (
            AccountLifecycleResult(
                ok=True,

                status="executed",

                changed=True,

                user_id=(
                    normalized_user
                ),

                enabled=(
                    enabled
                ),

                message=(
                    f"User '{normalized_user}' "
                    f"was successfully {state}."
                ),
            )
        )

    def enable_user(
        self,
        user_id: str,
    ) -> AccountLifecycleResult:

        return (
            self._set_enabled(
                user_id,
                enabled=True,
            )
        )

    def disable_user(
        self,
        user_id: str,
    ) -> AccountLifecycleResult:

        return (
            self._set_enabled(
                user_id,
                enabled=False,
            )
        )


class LdapAccountLifecycleService(
    AccountLifecycleService
):
    """
    Active Directory account lifecycle adapter.

    Enabling and disabling operate only on the ACCOUNTDISABLE bit
    inside userAccountControl.

    Other userAccountControl flags are preserved.
    """

    def __init__(
        self,
        directory: LdapDirectoryService,
    ) -> None:

        self.directory = (
            directory
        )

    def _set_enabled(
        self,
        user_id: str,
        *,
        enabled: bool,
    ) -> AccountLifecycleResult:

        connection = None

        mutation_attempted = (
            False
        )

        normalized_user = (
            (
                user_id.strip()
            )
            if isinstance(
                user_id,
                str,
            )
            else ""
        )

        if not normalized_user:

            return (
                AccountLifecycleResult(
                    ok=False,

                    status="denied",

                    user_id=(
                        normalized_user
                        or None
                    ),

                    enabled=None,

                    error=(
                        "user_id must be a non-empty string."
                    ),
                )
            )

        try:

            connection = (
                self.directory
                ._connect_write()
            )

            entry = (
                self.directory
                ._find_user(
                    connection,
                    normalized_user,
                )
            )

            if entry is None:

                return (
                    AccountLifecycleResult(
                        ok=False,

                        status="not_found",

                        user_id=(
                            normalized_user
                        ),

                        enabled=None,

                        error=(
                            f"User '{normalized_user}' "
                            "was not found."
                        ),
                    )
                )

            current_uac = (
                self.directory
                ._get_raw_integer_attribute(
                    entry,
                    "userAccountControl",
                    0,
                )
            )

            currently_enabled = (
                not bool(
                    current_uac
                    & ACCOUNT_DISABLE_FLAG
                )
            )

            if (
                currently_enabled
                == enabled
            ):

                state = (
                    "enabled"
                    if enabled
                    else "disabled"
                )

                return (
                    AccountLifecycleResult(
                        ok=True,

                        status="executed",

                        changed=False,

                        user_id=(
                            normalized_user
                        ),

                        enabled=(
                            enabled
                        ),

                        message=(
                            f"User '{normalized_user}' "
                            f"was already {state}."
                        ),
                    )
                )

            if enabled:

                desired_uac = (
                    current_uac
                    & ~ACCOUNT_DISABLE_FLAG
                )

            else:

                desired_uac = (
                    current_uac
                    | ACCOUNT_DISABLE_FLAG
                )

            mutation_attempted = (
                True
            )

            success = (
                connection.modify(
                    entry.entry_dn,

                    {
                        "userAccountControl": [
                            (
                                MODIFY_REPLACE,
                                [
                                    desired_uac,
                                ],
                            )
                        ],
                    },
                )
            )

            if not success:

                return (
                    AccountLifecycleResult(
                        ok=False,

                        status="error",

                        changed=False,

                        user_id=(
                            normalized_user
                        ),

                        enabled=(
                            currently_enabled
                        ),

                        error=(
                            "Active Directory account "
                            "lifecycle mutation failed: "
                            f"{connection.result}"
                        ),
                    )
                )

            verified_entry = (
                self.directory
                ._find_user(
                    connection,
                    normalized_user,
                )
            )

            if verified_entry is None:

                return (
                    AccountLifecycleResult(
                        ok=False,

                        status="unknown",

                        changed=False,

                        user_id=(
                            normalized_user
                        ),

                        enabled=None,

                        error=(
                            "Active Directory accepted the "
                            "account mutation, but the user "
                            "could not be re-read for "
                            "verification."
                        ),
                    )
                )

            verified_uac = (
                self.directory
                ._get_raw_integer_attribute(
                    verified_entry,
                    "userAccountControl",
                    0,
                )
            )

            verified_enabled = (
                not bool(
                    verified_uac
                    & ACCOUNT_DISABLE_FLAG
                )
            )

            if (
                verified_enabled
                != enabled
            ):

                return (
                    AccountLifecycleResult(
                        ok=False,

                        status="unknown",

                        changed=False,

                        user_id=(
                            normalized_user
                        ),

                        enabled=(
                            verified_enabled
                        ),

                        error=(
                            "Active Directory accepted the "
                            "account mutation, but the resulting "
                            "enabled state could not be verified."
                        ),
                    )
                )

            state = (
                "enabled"
                if enabled
                else "disabled"
            )

            return (
                AccountLifecycleResult(
                    ok=True,

                    status="executed",

                    changed=True,

                    user_id=(
                        normalized_user
                    ),

                    enabled=(
                        enabled
                    ),

                    message=(
                        f"User '{normalized_user}' "
                        f"was successfully {state}."
                    ),
                )
            )

        except Exception as exc:

            return (
                AccountLifecycleResult(
                    ok=False,

                    status=(
                        "unknown"
                        if mutation_attempted
                        else "error"
                    ),

                    changed=False,

                    user_id=(
                        normalized_user
                    ),

                    enabled=None,

                    error=(
                        "Active Directory account lifecycle "
                        f"operation failed: {exc}"
                    ),
                )
            )

        finally:

            if (
                connection
                is not None
            ):

                connection.unbind()

    def enable_user(
        self,
        user_id: str,
    ) -> AccountLifecycleResult:

        return (
            self._set_enabled(
                user_id,
                enabled=True,
            )
        )

    def disable_user(
        self,
        user_id: str,
    ) -> AccountLifecycleResult:

        return (
            self._set_enabled(
                user_id,
                enabled=False,
            )
        )


def build_account_lifecycle_service(
    directory: DirectoryService,
) -> AccountLifecycleService:

    if isinstance(
        directory,
        MockDirectoryService,
    ):

        return (
            MockAccountLifecycleService(
                directory
            )
        )

    if isinstance(
        directory,
        LdapDirectoryService,
    ):

        return (
            LdapAccountLifecycleService(
                directory
            )
        )

    raise RuntimeError(
        "Unsupported DirectoryService implementation "
        "for account lifecycle mutations: "
        f"{type(directory).__name__}"
    )
