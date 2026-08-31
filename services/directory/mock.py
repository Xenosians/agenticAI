from .base import DirectoryService


class MockDirectoryService(DirectoryService):

    def __init__(self):
        self.users = {
            "jdoe": {
                "enabled": True,
                "locked": False,
                "password_reset_count": 0,
            },
            "asmith": {
                "enabled": True,
                "locked": True,
                "password_reset_count": 0,
            },
            "disabled": {
                "enabled": False,
                "locked": False,
                "password_reset_count": 0,
            },
        }

        self.access = {
            ("jdoe", "vpn"): True,
            ("jdoe", "admin"): False,
            ("asmith", "vpn"): True,
            ("asmith", "admin"): True,
        }

        self.resource_aliases = {
            "vpn": "vpn",
            "vpn access": "vpn",
            "virtual private network": "vpn",
            "virtual private network access": "vpn",

            "admin": "admin",
            "admin access": "admin",
            "administrator": "admin",
            "administrator access": "admin",
        }

    def _normalize_user(
        self,
        user_id: str,
    ) -> str:
        return user_id.strip().lower()

    def _normalize_resource(
        self,
        resource: str,
    ) -> str:
        cleaned = resource.strip().lower()

        return self.resource_aliases.get(
            cleaned,
            cleaned,
        )

    def account_status(
        self,
        user_id: str,
    ) -> dict:

        normalized_user = self._normalize_user(
            user_id
        )

        user = self.users.get(
            normalized_user
        )

        if user is None:
            return {
                "ok": False,
                "error": (
                    f"User '{user_id}' not found."
                ),
            }

        return {
            "ok": True,
            "user_id": normalized_user,
            "enabled": user["enabled"],
            "locked": user["locked"],
        }

    def check_access(
        self,
        user_id: str,
        resource: str,
    ) -> dict:

        normalized_user = self._normalize_user(
            user_id
        )

        normalized_resource = (
            self._normalize_resource(
                resource
            )
        )

        key = (
            normalized_user,
            normalized_resource,
        )

        if key not in self.access:
            return {
                "ok": False,
                "user_id": normalized_user,
                "resource": normalized_resource,
                "has_access": None,
                "error": (
                    "No access information available "
                    f"for user '{user_id}' and "
                    f"resource '{resource}'."
                ),
            }

        return {
            "ok": True,
            "user_id": normalized_user,
            "resource": normalized_resource,
            "has_access": self.access[key],
            "error": None,
        }

    def unlock_user(
        self,
        user_id: str,
    ) -> dict:

        normalized_user = self._normalize_user(
            user_id
        )

        user = self.users.get(
            normalized_user
        )

        if user is None:
            return {
                "ok": False,
                "status": "error",
                "error": (
                    f"User '{user_id}' not found."
                ),
            }

        if not user["locked"]:
            return {
                "ok": True,
                "status": "executed",
                "changed": False,
                "user_id": normalized_user,
                "message": (
                    f"User '{normalized_user}' "
                    "was already unlocked."
                ),
            }

        user["locked"] = False

        return {
            "ok": True,
            "status": "executed",
            "changed": True,
            "user_id": normalized_user,
            "message": (
                f"User '{normalized_user}' "
                "was successfully unlocked."
            ),
        }

    def reset_password(
        self,
        user_id: str,
    ) -> dict:

        normalized_user = self._normalize_user(
            user_id
        )

        user = self.users.get(
            normalized_user
        )

        if user is None:
            return {
                "ok": False,
                "status": "error",
                "error": (
                    f"User '{user_id}' not found."
                ),
            }

        user["password_reset_count"] += 1

        return {
            "ok": True,
            "status": "executed",
            "changed": True,
            "user_id": normalized_user,
            "password_reset_count": (
                user["password_reset_count"]
            ),
            "message": (
                f"Password for '{normalized_user}' "
                "was successfully reset."
            ),
        }