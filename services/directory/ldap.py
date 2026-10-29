from typing import Any

from ldap3 import (
    ALL,
    BASE,
    MODIFY_REPLACE,
    SUBTREE,
    Connection,
    Server,
)
from ldap3.protocol.formatters.formatters import (
    format_sid,
)
from ldap3.utils.conv import (
    escape_filter_chars,
)

from config import (
    Settings,
    get_settings,
)

from .base import (
    MAX_ACCOUNT_SEARCH_RESULTS,
    MAX_GROUP_MEMBERS,
    MAX_USER_GROUPS,
    DirectoryService,
    group_name_candidates,
    search_fragment_error,
)


LDAP_MATCHING_RULE_IN_CHAIN = "1.2.840.113556.1.4.1941"

PAGED_RESULTS_CONTROL = "1.2.840.113556.1.4.319"

CONNECT_TIMEOUT_SECONDS = 10
RECEIVE_TIMEOUT_SECONDS = 30
SEARCH_TIME_LIMIT_SECONDS = 10
SEARCH_PAGE_SIZE = 50

MAX_GROUP_NAME_LENGTH = 256

LDAP_SUCCESS = 0
LDAP_SIZE_LIMIT_EXCEEDED = 4
LDAP_NO_SUCH_OBJECT = 32

# Constructed by the domain controller on every read. Its UF_LOCKOUT bit is the
# authoritative current lockout state: it honours the lockout duration and is
# present even when lockoutTime was never written (Active Directory and Samba
# omit lockoutTime until an account's first lockout).
COMPUTED_ACCOUNT_CONTROL = "msDS-User-Account-Control-Computed"
UF_LOCKOUT = 0x0010

# Account state may never fall back to a guessed value.
SECURITY_STATE_ATTRIBUTES = frozenset(
    {"userAccountControl", "lockoutTime", COMPUTED_ACCOUNT_CONTROL}
)

# Only identity and state attributes; password and hash attributes are
# never requested.
ACCOUNT_SEARCH_ATTRIBUTES = [
    "sAMAccountName",
    "displayName",
    "mail",
    "userAccountControl",
    "lockoutTime",
    COMPUTED_ACCOUNT_CONTROL,
]

GROUP_ATTRIBUTES = [
    "cn",
    "sAMAccountName",
    "objectSid",
]

GROUP_MEMBER_ATTRIBUTES = [
    "sAMAccountName",
    "cn",
    "displayName",
    "objectClass",
    "memberOf",
]

MEMBERSHIP_ORDER = {
    "direct": 0,
    "primary": 1,
    "nested": 2,
}


class DirectoryReadError(Exception):
    """An LDAP read could not establish the requested directory facts."""


class LdapDirectoryService(
    DirectoryService
):
    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = (
            settings
            if settings is not None
            else get_settings()
        )

        # ========================================================
        # LDAP SERVER
        # ========================================================

        self.host = (
            self.settings.ad_host
        )

        self.port = (
            self.settings.resolved_ad_port
        )

        self.use_ssl = (
            self.settings.ad_use_ssl
        )

        self.base_dn = (
            self.settings.ad_base_dn
        )

        # ========================================================
        # READ-ONLY BIND
        # ========================================================

        self.bind_user = (
            self.settings
            .ad_read_bind_identity
        )

        self.bind_password = (
            self.settings
            .ad_bind_password
        )

        # ========================================================
        # WRITE / MUTATION BIND
        # ========================================================

        self.write_bind_user = (
            self.settings
            .ad_write_bind_identity
        )

        self.write_bind_password = (
            self.settings
            .ad_write_bind_password
        )

        # ========================================================
        # ACCESS GROUP MAPPING
        # ========================================================

        self.access_groups = (
            self._normalize_access_groups(
                self.settings
                .ad_access_groups
            )
        )

        self._validate_config()

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def _validate_config(
        self,
    ) -> None:
        missing = []

        if not self.host:
            missing.append(
                "AD_HOST"
            )

        if not self.bind_user:
            missing.append(
                "AD_BIND_USER or AD_BIND_DN"
            )

        if not self.bind_password:
            missing.append(
                "AD_BIND_PASSWORD"
            )

        if not self.base_dn:
            missing.append(
                "AD_BASE_DN"
            )

        if missing:
            raise RuntimeError(
                "Missing Active Directory configuration: "
                + ", ".join(
                    missing
                )
            )

    @staticmethod
    def _normalize_access_groups(
        groups: dict[str, str],
    ) -> dict[str, str]:
        return {
            str(key).strip().lower():
            str(value).strip()

            for key, value
            in groups.items()
        }

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_user_id(
        user_id: str,
    ) -> str:
        return user_id.strip()

    @staticmethod
    def _normalize_resource(
        resource: str,
    ) -> str:
        cleaned = (
            resource
            .strip()
            .lower()
        )

        aliases = {
            "vpn":
                "vpn",
            "vpn access":
                "vpn",
            "virtual private network":
                "vpn",
            "virtual private network access":
                "vpn",

            "admin":
                "admin",
            "admin access":
                "admin",
            "administrator":
                "admin",
            "administrator access":
                "admin",
        }

        return aliases.get(
            cleaned,
            cleaned,
        )

    # ============================================================
    # LDAP ATTRIBUTE HELPERS
    # ============================================================

    @staticmethod
    def _get_raw_integer_attribute(
        entry: Any,
        attribute_name: str,
        default: int | None = None,
    ) -> int:
        """
        Read a single nonnegative integer directly from LDAP raw_values.

        Security state must remain unknown when attributes are unreadable or
        malformed. Only optional metadata callers may explicitly supply a
        fallback; account state and mutation verification never do so.

        ldap3 may convert some Active Directory attributes,
        such as lockoutTime, into datetime objects when
        schema information is available.

        Active Directory stores lockoutTime as an Integer8
        / Windows FILETIME value, so using raw_values avoids
        ldap3's automatic conversion.
        """

        try:
            attribute = entry[
                attribute_name
            ]

            raw_values = (
                attribute.raw_values
                if attribute is not None
                else None
            )

            if not raw_values or len(raw_values) != 1:
                raise ValueError("Expected one integer value.")

            raw_value = (
                raw_values[0]
            )

            if isinstance(
                raw_value,
                bytes,
            ):
                raw_value = (
                    raw_value.decode(
                        "ascii"
                    )
                )

            if type(raw_value) is str:
                if not raw_value or not raw_value.isascii() or not raw_value.isdecimal():
                    raise ValueError("Expected decimal integer bytes.")
                raw_value = int(raw_value)
            if type(raw_value) is not int or not 0 <= raw_value <= 2**63 - 1:
                raise ValueError("Expected a nonnegative LDAP integer.")
            if attribute_name in {"userAccountControl", COMPUTED_ACCOUNT_CONTROL} and raw_value > 2**32 - 1:
                raise ValueError("Invalid account control bit field.")
            return raw_value

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            AttributeError,
            UnicodeDecodeError,
        ):
            if default is not None and attribute_name not in SECURITY_STATE_ATTRIBUTES:
                return default
            raise DirectoryReadError(
                f"Active Directory did not return a readable {attribute_name} attribute."
            ) from None

    @classmethod
    def _account_flags(
        cls,
        entry: Any,
    ) -> tuple[bool, bool]:
        """
        Return (enabled, locked) for one account entry.
        """

        user_account_control = (
            cls._get_raw_integer_attribute(
                entry,
                "userAccountControl",
            )
        )

        # ACCOUNTDISABLE flag = 0x0002
        enabled = not bool(
            user_account_control
            & 0x0002
        )

        return (
            enabled,
            cls._locked_state(entry),
        )

    @staticmethod
    def _has_attribute(
        entry: Any,
        attribute_name: str,
    ) -> bool:
        try:
            attribute = entry[attribute_name]
        except (KeyError, AttributeError):
            return False
        return bool(getattr(attribute, "raw_values", None))

    @classmethod
    def _locked_state(
        cls,
        entry: Any,
    ) -> bool:
        """
        Current lockout state, from directory evidence only.

        The constructed UF_LOCKOUT bit decides when the directory returns it.
        A directory without that attribute falls back to an explicitly
        returned lockoutTime. With neither, the state is unknown and
        DirectoryReadError is raised; it is never assumed to be unlocked.
        """

        if cls._has_attribute(entry, COMPUTED_ACCOUNT_CONTROL):
            return bool(
                cls._get_raw_integer_attribute(
                    entry,
                    COMPUTED_ACCOUNT_CONTROL,
                )
                & UF_LOCKOUT
            )

        return (
            cls._get_raw_integer_attribute(
                entry,
                "lockoutTime",
            )
            > 0
        )

    @staticmethod
    def _text_attribute(
        entry: Any,
        attribute_name: str,
    ) -> str | None:
        try:
            value = (
                entry[
                    attribute_name
                ]
                .value
            )

        except (
            KeyError,
            IndexError,
            AttributeError,
        ):
            return None

        if isinstance(
            value,
            list,
        ):
            value = (
                value[0]
                if value
                else None
            )

        if value is None:
            return None

        text = (
            str(value)
            .strip()
        )

        return text or None

    @staticmethod
    def _list_attribute(
        entry: Any,
        attribute_name: str,
    ) -> list[str]:
        try:
            values = (
                entry[
                    attribute_name
                ]
                .values
            )

        except (
            KeyError,
            IndexError,
            AttributeError,
        ):
            return []

        return [
            str(value)

            for value
            in values or []
        ]

    @staticmethod
    def _object_sid(
        entry: Any,
    ) -> str | None:
        try:
            raw_values = (
                entry[
                    "objectSid"
                ]
                .raw_values
            )

        except (
            KeyError,
            IndexError,
            AttributeError,
        ):
            return None

        if not raw_values:
            return None

        raw_value = (
            raw_values[0]
        )

        if isinstance(
            raw_value,
            bytes,
        ):
            raw_value = (
                format_sid(
                    raw_value
                )
            )

        # format_sid returns already-textual SIDs unchanged, as bytes.
        if isinstance(
            raw_value,
            bytes,
        ):
            raw_value = (
                raw_value.decode(
                    "ascii",
                    "replace",
                )
            )

        sid = (
            str(raw_value)
            .strip()
        )

        return (
            sid
            if sid.startswith("S-")
            else None
        )

    # ============================================================
    # CONNECTIONS
    # ============================================================

    def _create_server(
        self,
    ) -> Server:
        from services.lua_adapters.host_boundaries import inspect_link

        inspect_link("ldap", f"{'ldaps' if self.use_ssl else 'ldap'}://{self.host}:{self.port}", operation="bind")
        return Server(
            self.host,
            port=self.port,
            use_ssl=self.use_ssl,
            get_info=ALL,
            connect_timeout=(
                CONNECT_TIMEOUT_SECONDS
            ),
        )

    def _connect(
        self,
    ) -> Connection:
        """
        Read-only directory connection.
        """

        server = (
            self._create_server()
        )

        return Connection(
            server,
            user=self.bind_user,
            password=self.bind_password,
            auto_bind=True,
            receive_timeout=(
                RECEIVE_TIMEOUT_SECONDS
            ),
        )

    def _connect_write(
        self,
    ) -> Connection:
        """
        Connection used only for approved mutations.
        """

        if not self.write_bind_user:
            raise RuntimeError(
                "AD_WRITE_BIND_USER or "
                "AD_WRITE_BIND_DN is not configured."
            )

        if not self.write_bind_password:
            raise RuntimeError(
                "AD_WRITE_BIND_PASSWORD "
                "is not configured."
            )

        server = (
            self._create_server()
        )

        return Connection(
            server,
            user=self.write_bind_user,
            password=(
                self.write_bind_password
            ),
            auto_bind=True,
        )

    # ============================================================
    # USER LOOKUP
    # ============================================================

    def _find_user(
        self,
        connection: Connection,
        user_id: str,
    ):
        normalized_user = (
            self._normalize_user_id(
                user_id
            )
        )

        safe_user_id = (
            escape_filter_chars(
                normalized_user
            )
        )

        ldap_filter = (
            "(&(objectClass=user)"
            f"(sAMAccountName={safe_user_id}))"
        )

        connection.search(
            search_base=self.base_dn,
            search_filter=ldap_filter,
            attributes=[
                "sAMAccountName",
                "userAccountControl",
                "lockoutTime",
                COMPUTED_ACCOUNT_CONTROL,
                "memberOf",
                "objectSid",
                "primaryGroupID",
            ],
        )

        if not connection.entries:
            return None

        return connection.entries[0]

    # ============================================================
    # ACCOUNT STATUS
    # ============================================================

    def account_status(
        self,
        user_id: str,
    ) -> dict:
        connection = None

        normalized_user = (
            self._normalize_user_id(
                user_id
            )
        )

        try:
            connection = (
                self._connect()
            )

            entry = (
                self._find_user(
                    connection,
                    normalized_user,
                )
            )

            if entry is None:
                return {
                    "ok":
                        False,
                    "user_id":
                        normalized_user,
                    "enabled":
                        None,
                    "locked":
                        None,
                    "error": (
                        f"User '{normalized_user}' "
                        "not found."
                    ),
                }

            enabled, locked = (
                self._account_flags(
                    entry
                )
            )

            return {
                "ok":
                    True,
                "user_id":
                    normalized_user,
                "enabled":
                    enabled,
                "locked":
                    locked,
                "error":
                    None,
            }

        except Exception as exc:
            return {
                "ok":
                    False,
                "user_id":
                    normalized_user,
                "enabled":
                    None,
                "locked":
                    None,
                "error": (
                    "Active Directory query failed: "
                    f"{exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    # ============================================================
    # ACCESS CHECK
    # ============================================================

    def check_access(
        self,
        user_id: str,
        resource: str,
    ) -> dict:
        connection = None

        normalized_user = (
            self._normalize_user_id(
                user_id
            )
        )

        normalized_resource = (
            self._normalize_resource(
                resource
            )
        )

        required_group = (
            self.access_groups.get(
                normalized_resource
            )
        )

        if required_group is None:
            return {
                "ok":
                    False,
                "user_id":
                    normalized_user,
                "resource":
                    normalized_resource,
                "has_access":
                    None,
                "error": (
                    "No Active Directory group is "
                    "configured for resource "
                    f"'{normalized_resource}'."
                ),
            }

        try:
            connection = (
                self._connect()
            )

            entry = (
                self._find_user(
                    connection,
                    normalized_user,
                )
            )

            if entry is None:
                return {
                    "ok":
                        False,
                    "user_id":
                        normalized_user,
                    "resource":
                        normalized_resource,
                    "has_access":
                        None,
                    "error": (
                        f"User '{normalized_user}' "
                        "not found."
                    ),
                }

            memberships = []

            try:
                memberships = (
                    entry.memberOf.values
                    if entry.memberOf
                    else []
                )

            except AttributeError:
                memberships = []

            required_group_lower = (
                required_group
                .strip()
                .lower()
            )

            has_access = any(
                self._group_matches(
                    group_dn,
                    required_group_lower,
                )

                for group_dn
                in memberships
            )

            return {
                "ok":
                    True,
                "user_id":
                    normalized_user,
                "resource":
                    normalized_resource,
                "has_access":
                    has_access,
                "error":
                    None,
            }

        except Exception as exc:
            return {
                "ok":
                    False,
                "user_id":
                    normalized_user,
                "resource":
                    normalized_resource,
                "has_access":
                    None,
                "error": (
                    "Active Directory query failed: "
                    f"{exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    # ============================================================
    # GROUP MATCH
    # ============================================================

    @staticmethod
    def _group_matches(
        group_dn: str,
        required_group: str,
    ) -> bool:
        group_dn_lower = (
            str(group_dn)
            .strip()
            .lower()
        )

        required_group_lower = (
            required_group
            .strip()
            .lower()
        )

        return (
            group_dn_lower
            .startswith(
                f"cn={required_group_lower},"
            )
        )

    # ============================================================
    # ACCOUNT UNLOCK
    # ============================================================

    def unlock_user(
        self,
        user_id: str,
    ) -> dict:
        connection = None
        mutation_attempted = False
        mutation_accepted = False

        normalized_user = (
            self._normalize_user_id(
                user_id
            )
        )

        try:
            connection = (
                self._connect_write()
            )

            entry = (
                self._find_user(
                    connection,
                    normalized_user,
                )
            )

            if entry is None:
                return {
                    "ok":
                        False,
                    "status":
                        "error",
                    "changed":
                        False,
                    "user_id":
                        normalized_user,
                    "message":
                        None,
                    "error": (
                        f"User '{normalized_user}' "
                        "not found."
                    ),
                }

            # ----------------------------------------------------
            # Nothing to change
            # ----------------------------------------------------

            if not self._locked_state(entry):
                return {
                    "ok":
                        True,
                    "status":
                        "executed",
                    "changed":
                        False,
                    "user_id":
                        normalized_user,
                    "message": (
                        f"User '{normalized_user}' "
                        "is already unlocked."
                    ),
                    "verification_ok": True,
                    "retry_safe": True,
                    "error":
                        None,
                }

            user_dn = (
                entry.entry_dn
            )

            # ----------------------------------------------------
            # Unlock
            # ----------------------------------------------------

            mutation_attempted = True
            success = (
                connection.modify(
                    user_dn,
                    {
                        "lockoutTime": [
                            (
                                MODIFY_REPLACE,
                                ["0"],
                            )
                        ],
                    },
                )
            )

            if not success:
                return {
                    "ok":
                        False,
                    "status":
                        "error",
                    "changed":
                        False,
                    "user_id":
                        normalized_user,
                    "message":
                        None,
                    "error": (
                        "Active Directory unlock "
                        "failed: "
                        f"{connection.result}"
                    ),
                }

            mutation_accepted = True
            # ----------------------------------------------------
            # Verify mutation
            # ----------------------------------------------------

            verification_entry = (
                self._find_user(
                    connection,
                    normalized_user,
                )
            )

            if verification_entry is None:
                return {
                    "ok":
                        False,
                    "status":
                        "outcome_unknown",
                    "changed":
                        None,
                    "mutation_performed": True,
                    "retry_safe": False,
                    "user_id":
                        normalized_user,
                    "message":
                        None,
                    "error": (
                        "Unable to verify account "
                        "after unlock."
                    ),
                }

            # The write must be visible: lockoutTime reads back as the 0 that
            # was written, and the directory no longer reports a lockout.
            new_lockout_time = (
                self
                ._get_raw_integer_attribute(
                    verification_entry,
                    "lockoutTime",
                )
            )

            if new_lockout_time != 0 or self._locked_state(verification_entry):
                return {
                    "ok":
                        False,
                    "status":
                        "outcome_unknown",
                    "changed":
                        None,
                    "mutation_performed": True,
                    "retry_safe": False,
                    "user_id":
                        normalized_user,
                    "message":
                        None,
                    "error": (
                        "LDAP modification returned "
                        "success, but account is still "
                        "reported as locked."
                    ),
                }

            return {
                "ok":
                    True,
                "status":
                    "executed",
                "changed":
                    True,
                "mutation_performed": True,
                "verification_ok": True,
                "retry_safe": False,
                "user_id":
                    normalized_user,
                "message": (
                    f"User '{normalized_user}' "
                    "was successfully unlocked."
                ),
                "error":
                    None,
            }

        except Exception as exc:
            return {
                "ok":
                    False,
                "status":
                    "outcome_unknown" if mutation_attempted else "error",
                "changed":
                    None if mutation_attempted else False,
                "mutation_performed": True if mutation_accepted else None if mutation_attempted else False,
                "retry_safe": not mutation_attempted,
                "user_id":
                    normalized_user,
                "message":
                    None,
                "error": (
                    "Active Directory unlock failed: "
                    f"{exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    # ============================================================
    # DIRECTORY READS: SEARCH, GROUPS, MEMBERS
    # ============================================================

    def resource_group(
        self,
        resource: str,
    ) -> tuple[str, str | None]:
        normalized_resource = (
            self._normalize_resource(
                resource
            )
        )

        return (
            normalized_resource,
            self.access_groups.get(
                normalized_resource
            ),
        )

    def _mapped_resources(
        self,
        group_dn: str,
    ) -> list[str]:
        """
        Logical resources whose configured group is this group.
        """

        resources = []

        for (
            resource,
            configured,
        ) in sorted(
            self.access_groups.items()
        ):
            value = (
                configured
                .strip()
            )

            if (
                value.lower()
                .startswith("cn=")
                and "," in value
            ):
                matches = (
                    value.casefold()
                    == group_dn.strip().casefold()
                )

            else:
                matches = (
                    self._group_matches(
                        group_dn,
                        value,
                    )
                )

            if matches:
                resources.append(
                    resource
                )

        return resources

    @staticmethod
    def _raise_for_result(
        connection: Connection,
    ) -> int:
        code = (
            connection.result
            .get("result")
        )

        if code in (
            LDAP_SUCCESS,
            LDAP_SIZE_LIMIT_EXCEEDED,
            LDAP_NO_SUCH_OBJECT,
        ):
            return code

        raise DirectoryReadError(
            "LDAP search failed: "
            f"{connection.result.get('description')}"
        )

    def _paged_search(
        self,
        connection: Connection,
        *,
        search_filter: str,
        attributes: list[str],
        limit: int,
        search_base: str | None = None,
        search_scope: str = SUBTREE,
    ) -> tuple[list[Any], bool]:
        """
        Read at most `limit` entries with the paged-results control.

        Returns the entries and whether more entries exist.
        """

        entries: list[Any] = []

        cookie = None

        while True:
            connection.search(
                search_base=(
                    search_base
                    or self.base_dn
                ),
                search_filter=(
                    search_filter
                ),
                search_scope=(
                    search_scope
                ),
                attributes=(
                    attributes
                ),
                time_limit=(
                    SEARCH_TIME_LIMIT_SECONDS
                ),
                paged_size=(
                    SEARCH_PAGE_SIZE
                ),
                paged_cookie=(
                    cookie
                ),
            )

            code = (
                self._raise_for_result(
                    connection
                )
            )

            if code == LDAP_NO_SUCH_OBJECT:
                return [], False

            entries.extend(
                connection.entries
            )

            if len(entries) > limit:
                return (
                    entries[:limit],
                    True,
                )

            if code == LDAP_SIZE_LIMIT_EXCEEDED:
                return entries, True

            cookie = (
                connection.result
                .get("controls", {})
                .get(PAGED_RESULTS_CONTROL, {})
                .get("value", {})
                .get("cookie")
            )

            if not cookie:
                return entries, False

    def _read_failure(
        self,
        result: dict,
        exc: Exception,
    ) -> dict:
        return {
            **result,
            "ok": False,
            "error": (
                "Active Directory query failed: "
                f"{exc}"
            ),
            "error_code": "provider_error",
        }

    def _entry_name(
        self,
        entry: Any,
    ) -> str:
        return (
            self._text_attribute(
                entry,
                "sAMAccountName",
            )
            or self._text_attribute(
                entry,
                "cn",
            )
            or str(entry.entry_dn)
        )

    def _primary_group(
        self,
        connection: Connection,
        user_entry: Any,
    ) -> Any | None:
        """
        The primary group (usually Domain Users) is not listed in
        memberOf or member, so it is resolved from the account SID.
        """

        sid = (
            self._object_sid(
                user_entry
            )
        )

        rid = (
            self._get_raw_integer_attribute(
                user_entry,
                "primaryGroupID",
                0,
            )
        )

        if not sid or rid <= 0:
            return None

        group_sid = (
            sid.rsplit("-", 1)[0]
            + f"-{rid}"
        )

        entries, _ = (
            self._paged_search(
                connection,
                search_filter=(
                    "(&(objectClass=group)"
                    f"(objectSid={escape_filter_chars(group_sid)}))"
                ),
                attributes=(
                    GROUP_ATTRIBUTES
                ),
                limit=1,
            )
        )

        return (
            entries[0]
            if entries
            else None
        )

    def _group_item(
        self,
        entry: Any,
        membership: str,
    ) -> dict:
        return {
            "name": self._entry_name(
                entry
            ),
            "membership": membership,
            "resources": self._mapped_resources(
                str(entry.entry_dn)
            ),
        }

    def account_search(
        self,
        query: str,
    ) -> dict:
        fragment = (
            query.strip()
            if isinstance(query, str)
            else ""
        )

        result = {
            "ok": False,
            "query": fragment,
            "accounts": [],
            "count": 0,
            "truncated": False,
            "error": None,
            "error_code": None,
        }

        invalid = (
            search_fragment_error(
                fragment
            )
        )

        if invalid is not None:
            return {
                **result,
                "error": invalid,
                "error_code": "invalid_argument",
            }

        safe_fragment = (
            escape_filter_chars(
                fragment
            )
        )

        ldap_filter = (
            "(&(objectClass=user)"
            "(!(objectClass=computer))"
            f"(|(sAMAccountName=*{safe_fragment}*)"
            f"(displayName=*{safe_fragment}*)"
            f"(mail=*{safe_fragment}*)))"
        )

        connection = None

        try:
            connection = (
                self._connect()
            )

            entries, truncated = (
                self._paged_search(
                    connection,
                    search_filter=(
                        ldap_filter
                    ),
                    attributes=(
                        ACCOUNT_SEARCH_ATTRIBUTES
                    ),
                    limit=(
                        MAX_ACCOUNT_SEARCH_RESULTS
                    ),
                )
            )

            accounts = []

            for entry in entries:
                enabled, locked = (
                    self._account_flags(
                        entry
                    )
                )

                accounts.append(
                    {
                        "username": self._entry_name(
                            entry
                        ),
                        "display_name": self._text_attribute(
                            entry,
                            "displayName",
                        ),
                        "email": self._text_attribute(
                            entry,
                            "mail",
                        ),
                        "enabled": enabled,
                        "locked": locked,
                    }
                )

            accounts.sort(
                key=lambda item: (
                    item["username"]
                    .casefold()
                )
            )

            return {
                **result,
                "ok": True,
                "accounts": accounts,
                "count": len(accounts),
                "truncated": truncated,
            }

        except Exception as exc:
            return self._read_failure(
                result,
                exc,
            )

        finally:
            if connection is not None:
                connection.unbind()

    def user_groups(
        self,
        user_id: str,
    ) -> dict:
        normalized_user = (
            self._normalize_user_id(
                user_id
            )
            if isinstance(user_id, str)
            else ""
        )

        result = {
            "ok": False,
            "user_id": normalized_user,
            "groups": [],
            "count": 0,
            "truncated": False,
            "error": None,
            "error_code": None,
        }

        if not normalized_user:
            return {
                **result,
                "error": "user_id must not be empty.",
                "error_code": "invalid_argument",
            }

        connection = None

        try:
            connection = (
                self._connect()
            )

            entry = (
                self._find_user(
                    connection,
                    normalized_user,
                )
            )

            if entry is None:
                # _find_user ignores the result code; a failing
                # directory must not read as a missing account.
                if (
                    connection.result
                    .get("result")
                    != LDAP_SUCCESS
                ):
                    raise DirectoryReadError(
                        "LDAP search failed: "
                        f"{connection.result.get('description')}"
                    )

                return {
                    **result,
                    "error": (
                        f"User '{normalized_user}' "
                        "not found."
                    ),
                    "error_code": "not_found",
                }

            direct = {
                dn.strip().casefold()

                for dn
                in self._list_attribute(
                    entry,
                    "memberOf",
                )
            }

            chain, truncated = (
                self._paged_search(
                    connection,
                    search_filter=(
                        "(&(objectClass=group)"
                        f"(member:{LDAP_MATCHING_RULE_IN_CHAIN}:="
                        f"{escape_filter_chars(str(entry.entry_dn))}))"
                    ),
                    attributes=(
                        GROUP_ATTRIBUTES
                    ),
                    limit=(
                        MAX_USER_GROUPS
                    ),
                )
            )

            groups: dict[str, tuple[Any, str]] = {}

            for group in chain:
                key = (
                    str(group.entry_dn)
                    .strip()
                    .casefold()
                )

                groups[key] = (
                    group,
                    "direct"
                    if key in direct
                    else "nested",
                )

            primary = (
                self._primary_group(
                    connection,
                    entry,
                )
            )

            if primary is not None:
                groups[
                    str(primary.entry_dn)
                    .strip()
                    .casefold()
                ] = (
                    primary,
                    "primary",
                )

                # Groups that contain the primary group are nested memberships too.
                via_primary, more = (
                    self._paged_search(
                        connection,
                        search_filter=(
                            "(&(objectClass=group)"
                            f"(member:{LDAP_MATCHING_RULE_IN_CHAIN}:="
                            f"{escape_filter_chars(str(primary.entry_dn))}))"
                        ),
                        attributes=(
                            GROUP_ATTRIBUTES
                        ),
                        limit=(
                            MAX_USER_GROUPS
                        ),
                    )
                )

                truncated = (
                    truncated
                    or more
                )

                for group in via_primary:
                    groups.setdefault(
                        str(group.entry_dn)
                        .strip()
                        .casefold(),
                        (
                            group,
                            "nested",
                        ),
                    )

            items = sorted(
                (
                    self._group_item(
                        group,
                        membership,
                    )

                    for group, membership
                    in groups.values()
                ),
                key=lambda item: (
                    MEMBERSHIP_ORDER[
                        item["membership"]
                    ],
                    item["name"].casefold(),
                ),
            )

            if len(items) > MAX_USER_GROUPS:
                items = items[:MAX_USER_GROUPS]
                truncated = True

            return {
                **result,
                "ok": True,
                "user_id": (
                    self._text_attribute(
                        entry,
                        "sAMAccountName",
                    )
                    or normalized_user
                ),
                "groups": items,
                "count": len(items),
                "truncated": truncated,
            }

        except Exception as exc:
            return self._read_failure(
                result,
                exc,
            )

        finally:
            if connection is not None:
                connection.unbind()

    def _find_group(
        self,
        connection: Connection,
        name: str,
        *,
        configured: bool = False,
    ) -> list[Any]:
        """
        Groups named exactly `name`, or the entry of a group DN.

        Configured resource groups match by CN only, like the access
        check and access mutations; a user-supplied name may also be
        the group's account name.
        """

        if (
            name.lower()
            .startswith("cn=")
            and "," in name
        ):
            entries, _ = (
                self._paged_search(
                    connection,
                    search_base=name,
                    search_scope=BASE,
                    search_filter=(
                        "(objectClass=group)"
                    ),
                    attributes=(
                        GROUP_ATTRIBUTES
                    ),
                    limit=1,
                )
            )

            return entries

        safe_name = (
            escape_filter_chars(
                name
            )
        )

        name_filter = (
            f"(cn={safe_name})"
            if configured
            else (
                f"(|(cn={safe_name})"
                f"(sAMAccountName={safe_name}))"
            )
        )

        entries, _ = (
            self._paged_search(
                connection,
                search_filter=(
                    "(&(objectClass=group)"
                    f"{name_filter})"
                ),
                attributes=(
                    GROUP_ATTRIBUTES
                ),
                limit=2,
            )
        )

        return entries

    @staticmethod
    def _member_type(
        entry: Any,
    ) -> str:
        classes = {
            str(value).casefold()

            for value
            in (
                LdapDirectoryService
                ._list_attribute(
                    entry,
                    "objectClass",
                )
            )
        }

        for object_class in (
            "computer",
            "group",
            "user",
            "contact",
        ):
            if object_class in classes:
                return object_class

        return "other"

    def group_members(
        self,
        group: str,
    ) -> dict:
        requested = (
            group.strip()
            if isinstance(group, str)
            else ""
        )

        result = {
            "ok": False,
            "group": requested,
            "resources": [],
            "members": [],
            "count": 0,
            "truncated": False,
            "error": None,
            "error_code": None,
        }

        if (
            not requested
            or len(requested)
            > MAX_GROUP_NAME_LENGTH
        ):
            return {
                **result,
                "error": (
                    "group must be a group or resource name of 1 to "
                    f"{MAX_GROUP_NAME_LENGTH} characters."
                ),
                "error_code": "invalid_argument",
            }

        connection = None

        try:
            connection = (
                self._connect()
            )

            entries: list[Any] = []

            for candidate in group_name_candidates(
                requested
            ):
                normalized_resource, configured = (
                    self.resource_group(
                        candidate
                    )
                )

                if configured is not None:
                    entries = (
                        self._find_group(
                            connection,
                            configured,
                            configured=True,
                        )
                    )

                    if not entries:
                        return {
                            **result,
                            "resources": [
                                normalized_resource,
                            ],
                            "error": (
                                f"The group '{configured}' configured "
                                f"for resource '{normalized_resource}' "
                                "was not found in the directory."
                            ),
                            "error_code": "not_found",
                        }

                    break

                entries = (
                    self._find_group(
                        connection,
                        candidate,
                    )
                )

                if entries:
                    break

            if not entries:
                return {
                    **result,
                    "error": (
                        f"No group or configured resource named "
                        f"'{requested}' was found."
                    ),
                    "error_code": "not_found",
                }

            if len(entries) > 1:
                return {
                    **result,
                    "error": (
                        f"More than one group is named '{requested}'; "
                        "use the group's account name."
                    ),
                    "error_code": "ambiguous",
                }

            group_entry = (
                entries[0]
            )

            group_dn = (
                str(group_entry.entry_dn)
            )

            members, truncated = (
                self._paged_search(
                    connection,
                    search_filter=(
                        f"(memberOf:{LDAP_MATCHING_RULE_IN_CHAIN}:="
                        f"{escape_filter_chars(group_dn)})"
                    ),
                    attributes=(
                        GROUP_MEMBER_ATTRIBUTES
                    ),
                    limit=(
                        MAX_GROUP_MEMBERS
                    ),
                )
            )

            group_key = (
                group_dn
                .strip()
                .casefold()
            )

            listed: dict[str, dict] = {}

            for member in members:
                direct = any(
                    dn.strip().casefold()
                    == group_key

                    for dn
                    in self._list_attribute(
                        member,
                        "memberOf",
                    )
                )

                listed[
                    str(member.entry_dn)
                    .strip()
                    .casefold()
                ] = self._member_item(
                    member,
                    "direct"
                    if direct
                    else "nested",
                )

            sid = (
                self._object_sid(
                    group_entry
                )
            )

            budget = (
                MAX_GROUP_MEMBERS
                - len(listed)
            )

            if (
                sid
                and not truncated
                and budget >= 0
            ):
                # Accounts whose primary group this is are not in member.
                primary_members, more = (
                    self._paged_search(
                        connection,
                        search_filter=(
                            "(primaryGroupID="
                            f"{escape_filter_chars(sid.rsplit('-', 1)[1])})"
                        ),
                        attributes=(
                            GROUP_MEMBER_ATTRIBUTES
                        ),
                        limit=budget,
                    )
                )

                truncated = more

                for member in primary_members:
                    listed.setdefault(
                        str(member.entry_dn)
                        .strip()
                        .casefold(),
                        self._member_item(
                            member,
                            "primary",
                        ),
                    )

            items = sorted(
                listed.values(),
                key=lambda item: (
                    MEMBERSHIP_ORDER[
                        item["membership"]
                    ],
                    item["name"].casefold(),
                ),
            )

            return {
                **result,
                "ok": True,
                "group": self._entry_name(
                    group_entry
                ),
                "resources": self._mapped_resources(
                    group_dn
                ),
                "members": items,
                "count": len(items),
                "truncated": truncated,
            }

        except Exception as exc:
            return self._read_failure(
                result,
                exc,
            )

        finally:
            if connection is not None:
                connection.unbind()

    def _member_item(
        self,
        entry: Any,
        membership: str,
    ) -> dict:
        return {
            "name": self._entry_name(
                entry
            ),
            "display_name": self._text_attribute(
                entry,
                "displayName",
            ),
            "type": self._member_type(
                entry
            ),
            "membership": membership,
        }
