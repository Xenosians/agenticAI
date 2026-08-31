import json
import os

from ldap3 import (
    ALL,
    Connection,
    Server,
)
from ldap3.utils.conv import escape_filter_chars

from .base import DirectoryService


class LdapDirectoryService(DirectoryService):
    def __init__(self):
        self.host = os.environ.get("AD_HOST")
        self.bind_user = os.environ.get("AD_BIND_USER")
        self.bind_password = os.environ.get("AD_BIND_PASSWORD")
        self.base_dn = os.environ.get("AD_BASE_DN")

        self.use_ssl = (
            os.environ.get(
                "AD_USE_SSL",
                "true",
            ).lower()
            == "true"
        )

        self.port = int(
            os.environ.get(
                "AD_PORT",
                "636" if self.use_ssl else "389",
            )
        )

        self.access_groups = self._load_access_groups()

        self._validate_config()

    def _validate_config(self):
        required = {
            "AD_HOST": self.host,
            "AD_BIND_USER": self.bind_user,
            "AD_BIND_PASSWORD": self.bind_password,
            "AD_BASE_DN": self.base_dn,
        }

        missing = [
            name
            for name, value in required.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Missing Active Directory configuration: "
                + ", ".join(missing)
            )

    def _load_access_groups(self) -> dict:
        """
        Example environment value:

        AD_ACCESS_GROUPS={
            "vpn": "VPN-Users",
            "admin": "Domain Admins"
        }
        """

        raw = os.environ.get(
            "AD_ACCESS_GROUPS",
            "{}",
        )

        try:
            groups = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "AD_ACCESS_GROUPS must be valid JSON."
            ) from exc

        return {
            str(key).strip().lower():
            str(value).strip()
            for key, value in groups.items()
        }

    def _connect(self) -> Connection:
        server = Server(
            self.host,
            port=self.port,
            use_ssl=self.use_ssl,
            get_info=ALL,
        )

        connection = Connection(
            server,
            user=self.bind_user,
            password=self.bind_password,
            auto_bind=True,
        )

        return connection

    def _find_user(
        self,
        connection: Connection,
        user_id: str,
    ):
        safe_user_id = escape_filter_chars(
            user_id.strip()
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

    def account_status(
        self,
        user_id: str,
    ) -> dict:
        connection = None

        try:
            connection = self._connect()

            entry = self._find_user(
                connection,
                user_id,
            )

            if entry is None:
                return {
                    "ok": False,
                    "error": (
                        f"User '{user_id}' not found."
                    ),
                }

            user_account_control = int(
                entry.userAccountControl.value or 0
            )

            lockout_time = int(
                entry.lockoutTime.value or 0
            )

            # ACCOUNTDISABLE flag = 0x0002
            enabled = not bool(
                user_account_control & 0x0002
            )

            locked = lockout_time > 0

            return {
                "ok": True,
                "user_id": user_id,
                "enabled": enabled,
                "locked": locked,
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": (
                    f"Active Directory query failed: {exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    def check_access(
        self,
        user_id: str,
        resource: str,
    ) -> dict:
        normalized_resource = (
            resource.strip().lower()
        )

        required_group = self.access_groups.get(
            normalized_resource
        )

        if required_group is None:
            return {
                "ok": False,
                "user_id": user_id,
                "resource": normalized_resource,
                "has_access": None,
                "error": (
                    "No Active Directory group is "
                    f"configured for resource "
                    f"'{resource}'."
                ),
            }

        connection = None

        try:
            connection = self._connect()

            entry = self._find_user(
                connection,
                user_id,
            )

            if entry is None:
                return {
                    "ok": False,
                    "user_id": user_id,
                    "resource": normalized_resource,
                    "has_access": None,
                    "error": (
                        f"User '{user_id}' not found."
                    ),
                }

            memberships = (
                entry.memberOf.values
                if entry.memberOf
                else []
            )

            required_group_lower = (
                required_group.lower()
            )

            has_access = any(
                self._group_matches(
                    dn,
                    required_group_lower,
                )
                for dn in memberships
            )

            return {
                "ok": True,
                "user_id": user_id,
                "resource": normalized_resource,
                "has_access": has_access,
                "error": None,
            }

        except Exception as exc:
            return {
                "ok": False,
                "user_id": user_id,
                "resource": normalized_resource,
                "has_access": None,
                "error": (
                    f"Active Directory query failed: {exc}"
                ),
            }

        finally:
            if connection is not None:
                connection.unbind()

    @staticmethod
    def _group_matches(
        group_dn: str,
        required_group: str,
    ) -> bool:
        """
        Example DN:

        CN=VPN-Users,OU=Groups,DC=example,DC=com
        """

        group_dn_lower = str(
            group_dn
        ).lower()

        return group_dn_lower.startswith(
            f"cn={required_group},"
        )

    def unlock_user(
        self,
        user_id: str,
    ) -> dict:
        """
        Real mutations are intentionally disabled
        during the first LDAP integration phase.
        """

        return {
            "ok": False,
            "status": "error",
            "changed": False,
            "user_id": user_id,
            "error": (
                "Real Active Directory account "
                "unlock is not enabled yet."
            ),
        }

    def reset_password(
        self,
        user_id: str,
    ) -> dict:
        """
        Real mutations are intentionally disabled
        during the first LDAP integration phase.
        """

        return {
            "ok": False,
            "status": "error",
            "changed": False,
            "user_id": user_id,
            "error": (
                "Real Active Directory password "
                "reset is not enabled yet."
            ),
        }