import json
from typing import Any

from ldap3 import (
    ALL,
    MODIFY_REPLACE,
    Connection,
    Server,
)
from ldap3.utils.conv import escape_filter_chars

from config import (
    get_bool_env,
    get_env,
)

from .base import DirectoryService


class LdapDirectoryService(DirectoryService):

    def __init__(self):
        # =================================================
        # LDAP SERVER
        # =================================================

        self.host = get_env("AD_HOST")
        self.base_dn = get_env("AD_BASE_DN")

        self.use_ssl = get_bool_env(
            "AD_USE_SSL",
            True,
        )

        default_port = (
            "636"
            if self.use_ssl
            else "389"
        )

        self.port = int(
            get_env(
                "AD_PORT",
                default_port,
            )
            or default_port
        )

        # =================================================
        # READ-ONLY BIND
        # =================================================

        self.bind_user = (
            get_env("AD_BIND_USER")
            or get_env("AD_BIND_DN")
        )

        self.bind_password = get_env(
            "AD_BIND_PASSWORD"
        )

        # =================================================
        # WRITE / MUTATION BIND
        # =================================================

        self.write_bind_user = (
            get_env("AD_WRITE_BIND_USER")
            or get_env("AD_WRITE_BIND_DN")
        )

        self.write_bind_password = get_env(
            "AD_WRITE_BIND_PASSWORD"
        )

        # =================================================
        # ACCESS GROUP MAPPING
        # =================================================

        self.access_groups = (
            self._load_access_groups()
        )

        self._validate_config()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    def _validate_config(self):
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
                + ", ".join(missing)
            )

    def _load_access_groups(self) -> dict:
        raw = (
            get_env(
                "AD_ACCESS_GROUPS",
                "{}",
            )
            or "{}"
        )

        try:
            groups = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "AD_ACCESS_GROUPS must be valid JSON."
            ) from exc

        if not isinstance(groups, dict):
            raise RuntimeError(
                "AD_ACCESS_GROUPS must be a JSON object."
            )

        return {
            str(key).strip().lower():
            str(value).strip()
            for key, value in groups.items()
        }

    # =====================================================
    # NORMALIZATION
    # =====================================================

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
            "vpn": "vpn",
            "vpn access": "vpn",
            "virtual private network": "vpn",
            "virtual private network access": "vpn",

            "admin": "admin",
            "admin access": "admin",
            "administrator": "admin",
            "administrator access": "admin",
        }

        return aliases.get(
            cleaned,
            cleaned,
        )

    # =====================================================
    # LDAP ATTRIBUTE HELPERS
    # =====================================================

    @staticmethod
    def _get_raw_integer_attribute(
        entry: Any,
        attribute_name: str,
        default: int = 0,
    ) -> int:
        """
        Read an integer directly from LDAP raw_values.

        ldap3 may convert some Active Directory attributes,
        such as lockoutTime, into datetime objects when
        schema information is available.

        Active Directory stores lockoutTime as an Integer8
        / Windows FILETIME value, so using raw_values avoids
        ldap3's automatic conversion.
        """

        try:
            attribute = entry[attribute_name]

            raw_values = (
                attribute.raw_values
                if attribute is not None
                else None
            )

            if not raw_values:
                return default

            raw_value = raw_values[0]

            if raw_value is None:
                return default

            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode(
                    "utf-8"
                )

            return int(raw_value)

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            AttributeError,
            UnicodeDecodeError,
        ):
            return default

    # =====================================================
    # CONNECTIONS
    # =====================================================

    def _create_server(
        self,
    ) -> Server:
        return Server(
            self.host,
            port=self.port,
            use_ssl=self.use_ssl,
            get_info=ALL,
        )

    def _connect(
        self,
    ) -> Connection:
        """
        Read-only directory connection.
        """

        server = self._create_server()

        return Connection(
            server,
            user=self.bind_user,
            password=self.bind_password,
            auto_bind=True,
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

        server = self._create_server()

        return Connection(
            server,
            user=self.write_bind_user,
            password=self.write_bind_password,
            auto_bind=True,
        )

    # =====================================================
    # USER LOOKUP
    # =====================================================

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
                "memberOf",
            ],
        )

        if not connection.entries:
            return None

        return connection.entries[0]

    # =====================================================
    # ACCOUNT STATUS
    # =====================================================

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
            connection = self._connect()

            entry = self._find_user(
                connection,
                normalized_user,
            )

            if entry is None:
                return {
                    "ok": False,
                    "user_id": normalized_user,
                    "enabled": None,
                    "locked": None,
                    "error": (
                        f"User '{normalized_user}' "
                        "not found."
                    ),
                }

            user_account_control = (
                self._get_raw_integer_attribute(
                    entry,
                    "userAccountControl",
                    0,
                )
            )

            lockout_time = (
                self._get_raw_integer_attribute(
                    entry,
                    "lockoutTime",
                    0,
                )
            )

            # ACCOUNTDISABLE flag = 0x0002
            enabled = not bool(
                user_account_control
                & 0x0002
            )

            locked = (
                lockout_time > 0
            )

            return {
                "ok": True,
                "user_id": normalized_user,
                "enabled": enabled,
                "locked": locked,
                "error": None,
            }

        except Exception as exc:
            return {
                "ok": False,
                "user_id": normalized_user,
                "enabled": None,
                "locked": None,
                "error": (
                    "Active Directory query failed: "
                    f"{exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    # =====================================================
    # ACCESS CHECK
    # =====================================================

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
                "ok": False,
                "user_id": normalized_user,
                "resource": normalized_resource,
                "has_access": None,
                "error": (
                    "No Active Directory group is "
                    "configured for resource "
                    f"'{normalized_resource}'."
                ),
            }

        try:
            connection = self._connect()

            entry = self._find_user(
                connection,
                normalized_user,
            )

            if entry is None:
                return {
                    "ok": False,
                    "user_id": normalized_user,
                    "resource": normalized_resource,
                    "has_access": None,
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
                for group_dn in memberships
            )

            return {
                "ok": True,
                "user_id": normalized_user,
                "resource": normalized_resource,
                "has_access": has_access,
                "error": None,
            }

        except Exception as exc:
            return {
                "ok": False,
                "user_id": normalized_user,
                "resource": normalized_resource,
                "has_access": None,
                "error": (
                    "Active Directory query failed: "
                    f"{exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    # =====================================================
    # GROUP MATCH
    # =====================================================

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

        return group_dn_lower.startswith(
            f"cn={required_group_lower},"
        )

    # =====================================================
    # ACCOUNT UNLOCK
    # =====================================================

    def unlock_user(
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
                self._connect_write()
            )

            entry = self._find_user(
                connection,
                normalized_user,
            )

            if entry is None:
                return {
                    "ok": False,
                    "status": "error",
                    "changed": False,
                    "user_id": normalized_user,
                    "message": None,
                    "error": (
                        f"User '{normalized_user}' "
                        "not found."
                    ),
                }

            lockout_time = (
                self._get_raw_integer_attribute(
                    entry,
                    "lockoutTime",
                    0,
                )
            )

            # -------------------------------------------------
            # Nothing to change
            # -------------------------------------------------

            if lockout_time == 0:
                return {
                    "ok": True,
                    "status": "executed",
                    "changed": False,
                    "user_id": normalized_user,
                    "message": (
                        f"User '{normalized_user}' "
                        "is already unlocked."
                    ),
                    "error": None,
                }

            user_dn = entry.entry_dn

            # -------------------------------------------------
            # Unlock
            # -------------------------------------------------

            success = connection.modify(
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

            if not success:
                return {
                    "ok": False,
                    "status": "error",
                    "changed": False,
                    "user_id": normalized_user,
                    "message": None,
                    "error": (
                        "Active Directory unlock "
                        "failed: "
                        f"{connection.result}"
                    ),
                }

            # -------------------------------------------------
            # Verify mutation
            # -------------------------------------------------

            verification_entry = (
                self._find_user(
                    connection,
                    normalized_user,
                )
            )

            if verification_entry is None:
                return {
                    "ok": False,
                    "status": "error",
                    "changed": False,
                    "user_id": normalized_user,
                    "message": None,
                    "error": (
                        "Unable to verify account "
                        "after unlock."
                    ),
                }

            new_lockout_time = (
                self._get_raw_integer_attribute(
                    verification_entry,
                    "lockoutTime",
                    0,
                )
            )

            if new_lockout_time != 0:
                return {
                    "ok": False,
                    "status": "error",
                    "changed": False,
                    "user_id": normalized_user,
                    "message": None,
                    "error": (
                        "LDAP modification returned "
                        "success, but account is still "
                        "reported as locked."
                    ),
                }

            return {
                "ok": True,
                "status": "executed",
                "changed": True,
                "user_id": normalized_user,
                "message": (
                    f"User '{normalized_user}' "
                    "was successfully unlocked."
                ),
                "error": None,
            }

        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "changed": False,
                "user_id": normalized_user,
                "message": None,
                "error": (
                    "Active Directory unlock failed: "
                    f"{exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    # =====================================================
    # PASSWORD RESET
    # =====================================================

    def reset_password(
        self,
        user_id: str,
    ) -> dict:
        normalized_user = (
            self._normalize_user_id(
                user_id
            )
        )

        return {
            "ok": False,
            "status": "error",
            "changed": False,
            "user_id": normalized_user,
            "message": None,
            "error": (
                "Real Active Directory password "
                "reset is not enabled yet."
            ),
        }