from abc import ABC, abstractmethod


MAX_ACCOUNT_SEARCH_RESULTS = 25
MAX_USER_GROUPS = 200
MAX_GROUP_MEMBERS = 100

MIN_SEARCH_FRAGMENT_LENGTH = 2
MAX_SEARCH_FRAGMENT_LENGTH = 64

# Existing capabilities that would clear a blocking reason. An access
# explanation only names them; it never executes them.
ACCESS_FIX_TOOLS = {
    "account_disabled": "enable_user",
    "account_locked": "unlock_user",
    "not_a_member": "grant_access",
}

# Reasons that prove there is no access; the others leave it open.
BLOCKING_ACCESS_REASONS = {
    "account_not_found",
    "account_disabled",
    "account_locked",
    "resource_group_missing",
    "not_a_member",
}

GRANTING_MEMBERSHIPS = {
    "direct",
    "nested",
    "primary",
}


def search_fragment_error(
    fragment: str,
) -> str | None:
    if not fragment:
        return "query must not be empty."

    if (
        len(fragment)
        < MIN_SEARCH_FRAGMENT_LENGTH
    ):
        # A single character matches nearly every account.
        return (
            "query must contain at least "
            f"{MIN_SEARCH_FRAGMENT_LENGTH} characters."
        )

    if (
        len(fragment)
        > MAX_SEARCH_FRAGMENT_LENGTH
    ):
        return (
            "query must contain at most "
            f"{MAX_SEARCH_FRAGMENT_LENGTH} characters."
        )

    return None


def group_name_candidates(
    requested: str,
) -> list[str]:
    """
    Names to try for a requested group, most literal first.
    """

    candidates = [
        requested,
    ]

    if (
        requested.casefold()
        .endswith(" group")
    ):
        # "the vpn group" names the resource, not a group called so.
        candidates.append(
            requested[: -len(" group")]
            .strip()
        )

    return [
        candidate

        for candidate
        in candidates

        if candidate
    ]


class DirectoryService(ABC):

    @abstractmethod
    def account_status(
        self,
        user_id: str,
    ) -> dict:
        pass

    @abstractmethod
    def check_access(
        self,
        user_id: str,
        resource: str,
    ) -> dict:
        pass

    @abstractmethod
    def unlock_user(
        self,
        user_id: str,
    ) -> dict:
        pass

    # Concrete, so partial fakes of this interface keep working.

    def account_search(
        self,
        query: str,
    ) -> dict:
        raise NotImplementedError

    def user_groups(
        self,
        user_id: str,
    ) -> dict:
        raise NotImplementedError

    def group_members(
        self,
        group: str,
    ) -> dict:
        raise NotImplementedError

    def resource_group(
        self,
        resource: str,
    ) -> tuple[str, str | None]:
        """
        Return the normalized logical resource and its configured
        directory group, or None when the resource is not mapped.
        """

        raise NotImplementedError

    # ============================================================
    # ACCESS EXPLANATION
    # ============================================================

    def access_explain(
        self,
        user_id: str,
        resource: str,
    ) -> dict:
        """
        Explain one user's access to one logical resource from the
        other reads of this service; no directory logic of its own.
        """

        user = (
            user_id.strip()
            if isinstance(user_id, str)
            else ""
        )

        requested = (
            resource.strip()
            if isinstance(resource, str)
            else ""
        )

        result = {
            "ok": False,
            "user_id": user,
            "resource": requested,
            "group": None,
            "has_access": None,
            "checks": {},
            "reasons": [],
            "error": None,
            "error_code": None,
        }

        if not user or not requested:
            return {
                **result,
                "error": (
                    "user_id and resource must not be empty."
                ),
                "error_code": "invalid_argument",
            }

        normalized_resource, group = (
            self.resource_group(
                requested
            )
        )

        result.update(
            resource=normalized_resource,
            group=group,
        )

        membership = (
            self.user_groups(
                user
            )
        )

        exists = bool(
            membership.get("ok")
        )

        if (
            not exists
            and membership.get("error_code")
            != "not_found"
        ):
            return {
                **result,
                "error": membership.get("error"),
                "error_code": (
                    membership.get("error_code")
                    or "provider_error"
                ),
            }

        account = (
            membership.get("user_id")
            or user
        )

        result["user_id"] = account

        checks = {
            "account_exists": exists,
            "enabled": None,
            "locked": None,
            "resource_mapped": group is not None,
            "membership": None,
        }

        reasons = []

        def add_reason(
            code: str,
            message: str,
        ) -> None:
            reasons.append(
                {
                    "code": code,
                    "message": message,
                    "suggested_tool": (
                        ACCESS_FIX_TOOLS.get(code)
                    ),
                }
            )

        if not exists:
            add_reason(
                "account_not_found",
                f"Account '{account}' was not found "
                "in the directory.",
            )

        else:
            status = (
                self.account_status(
                    account
                )
            )

            if not status.get("ok"):
                return {
                    **result,
                    "error": status.get("error"),
                    "error_code": "provider_error",
                }

            checks["enabled"] = status.get("enabled")
            checks["locked"] = status.get("locked")

            if checks["enabled"] is False:
                add_reason(
                    "account_disabled",
                    f"Account '{account}' is disabled.",
                )

            if checks["locked"] is True:
                add_reason(
                    "account_locked",
                    f"Account '{account}' is locked.",
                )

        if group is None:
            add_reason(
                "resource_not_mapped",
                "No directory group is configured for "
                f"resource '{normalized_resource}'.",
            )

        elif exists:
            kinds = [
                item.get("membership")

                for item
                in membership.get("groups", [])

                if normalized_resource
                in item.get("resources", [])
            ]

            if kinds:
                checks["membership"] = (
                    "direct"
                    if "direct" in kinds
                    else kinds[0]
                )

            elif membership.get("truncated"):
                # The group list was capped, so a missing group proves nothing.
                checks["membership"] = "unknown"

                add_reason(
                    "membership_unknown",
                    f"'{account}' belongs to more groups than one "
                    f"read returns; membership in '{group}' could "
                    "not be confirmed.",
                )

            else:
                lookup = (
                    self.group_members(
                        normalized_resource
                    )
                )

                if lookup.get("ok"):
                    checks["membership"] = "none"

                    add_reason(
                        "not_a_member",
                        f"'{account}' is not a member of '{group}', "
                        "directly or through nested groups.",
                    )

                elif (
                    lookup.get("error_code")
                    == "not_found"
                ):
                    add_reason(
                        "resource_group_missing",
                        f"The group '{group}' configured for "
                        f"'{normalized_resource}' does not exist "
                        "in the directory.",
                    )

                else:
                    return {
                        **result,
                        "error": lookup.get("error"),
                        "error_code": (
                            lookup.get("error_code")
                            or "provider_error"
                        ),
                    }

        codes = {
            reason["code"]

            for reason
            in reasons
        }

        if codes & BLOCKING_ACCESS_REASONS:
            has_access = False

        elif codes:
            has_access = None

        else:
            has_access = (
                checks["membership"]
                in GRANTING_MEMBERSHIPS
            )

        return {
            **result,
            "ok": True,
            "has_access": has_access,
            "checks": checks,
            "reasons": reasons,
        }
