from .base import (
    MAX_ACCOUNT_SEARCH_RESULTS,
    MAX_GROUP_MEMBERS,
    DirectoryService,
    group_name_candidates,
    search_fragment_error,
)


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

        # Access state lives in self.access, so mock group membership is
        # derived from it and stays consistent with grant/revoke.
        self.resource_groups = {
            "vpn": "VPN-Users",
            "admin": "ITSM-Admins",
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
            self.resource_groups.get(
                normalized_resource
            ),
        )

    def _display_name(
        self,
        user: dict,
    ) -> str | None:
        names = [
            user.get("given_name"),
            user.get("family_name"),
        ]

        text = " ".join(
            name

            for name
            in names

            if name
        )

        return text or None

    def account_search(
        self,
        query: str,
    ) -> dict:
        fragment = (
            query.strip()
            if isinstance(query, str)
            else ""
        )

        invalid = (
            search_fragment_error(
                fragment
            )
        )

        if invalid is not None:
            return {
                "ok": False,
                "query": fragment,
                "accounts": [],
                "count": 0,
                "truncated": False,
                "error": invalid,
                "error_code": "invalid_argument",
            }

        needle = (
            fragment.casefold()
        )

        accounts = []

        for username, user in sorted(
            self.users.items()
        ):
            display_name = (
                self._display_name(
                    user
                )
            )

            values = [
                username,
                display_name or "",
                user.get("email") or "",
            ]

            if any(
                needle in value.casefold()

                for value
                in values
            ):
                accounts.append(
                    {
                        "username": username,
                        "display_name": display_name,
                        "email": user.get("email"),
                        "enabled": user["enabled"],
                        "locked": user["locked"],
                    }
                )

        return {
            "ok": True,
            "query": fragment,
            "accounts": accounts[
                :MAX_ACCOUNT_SEARCH_RESULTS
            ],
            "count": min(
                len(accounts),
                MAX_ACCOUNT_SEARCH_RESULTS,
            ),
            "truncated": (
                len(accounts)
                > MAX_ACCOUNT_SEARCH_RESULTS
            ),
            "error": None,
            "error_code": None,
        }

    def user_groups(
        self,
        user_id: str,
    ) -> dict:
        normalized_user = (
            self._normalize_user(
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

        if normalized_user not in self.users:
            return {
                **result,
                "error": (
                    f"User '{user_id}' not found."
                ),
                "error_code": "not_found",
            }

        groups = sorted(
            (
                {
                    "name": self.resource_groups[resource],
                    "membership": "direct",
                    "resources": [resource],
                }

                for (user, resource), granted
                in self.access.items()

                if user == normalized_user
                and granted
                and resource in self.resource_groups
            ),
            key=lambda item: item["name"].casefold(),
        )

        return {
            **result,
            "ok": True,
            "groups": groups,
            "count": len(groups),
        }

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

        if not requested:
            return {
                **result,
                "error": "group must not be empty.",
                "error_code": "invalid_argument",
            }

        resource, name = None, None

        for candidate in group_name_candidates(
            requested
        ):
            resource, name = (
                self.resource_group(
                    candidate
                )
            )

            if name is None:
                resource, name = next(
                    (
                        (key, value)

                        for key, value
                        in self.resource_groups.items()

                        if value.casefold()
                        == candidate.casefold()
                    ),
                    (None, None),
                )

            if name is not None:
                break

        if name is None:
            return {
                **result,
                "error": (
                    f"No group or configured resource named "
                    f"'{requested}' was found."
                ),
                "error_code": "not_found",
            }

        members = [
            {
                "name": user,
                "display_name": self._display_name(
                    self.users.get(user, {})
                ),
                "type": "user",
                "membership": "direct",
            }

            for (user, key), granted
            in sorted(self.access.items())

            if key == resource
            and granted
        ]

        return {
            **result,
            "ok": True,
            "group": name,
            "resources": [resource],
            "members": members[:MAX_GROUP_MEMBERS],
            "count": min(
                len(members),
                MAX_GROUP_MEMBERS,
            ),
            "truncated": (
                len(members)
                > MAX_GROUP_MEMBERS
            ),
        }
