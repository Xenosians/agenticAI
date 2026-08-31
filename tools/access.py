RESOURCE_ALIASES = {
    "vpn": "vpn",
    "vpn access": "vpn",
    "virtual private network": "vpn",
    "virtual private network access": "vpn",

    "admin": "admin",
    "admin access": "admin",
    "administrator": "admin",
    "administrator access": "admin",
}


MOCK_ACCESS = {
    ("jdoe", "vpn"): True,
    ("jdoe", "admin"): False,
    ("asmith", "vpn"): True,
    ("asmith", "admin"): True,
}


def normalize_resource(resource: str) -> str:
    """
    Convert natural-language resource names into
    canonical internal resource identifiers.
    """

    cleaned = resource.strip().lower()

    return RESOURCE_ALIASES.get(
        cleaned,
        cleaned,
    )


def check_access(
    user_id: str,
    resource: str,
) -> dict:
    """
    Check whether a user has access to a resource.

    Currently mocked.
    Later this implementation can be replaced
    by real infrastructure/AD queries.
    """

    normalized_user = user_id.strip().lower()
    normalized_resource = normalize_resource(resource)

    key = (
        normalized_user,
        normalized_resource,
    )

    if key not in MOCK_ACCESS:
        return {
            "ok": False,
            "user_id": normalized_user,
            "resource": normalized_resource,
            "has_access": None,
            "error": (
                "No access information available for "
                f"user '{user_id}' and resource '{resource}'."
            ),
        }

    return {
        "ok": True,
        "user_id": normalized_user,
        "resource": normalized_resource,
        "has_access": MOCK_ACCESS[key],
        "error": None,
    }