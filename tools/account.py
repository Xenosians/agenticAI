def account_status(user_id: str) -> dict:
    """
    Check the status of an Active Directory account.

    This is currently a mock implementation.
    Later this function will talk to Active Directory through MCP.
    """

    mock_users = {
        "jdoe": {
            "enabled": True,
            "locked": False,
        },
        "asmith": {
            "enabled": True,
            "locked": True,
        },
        "disabled": {
            "enabled": False,
            "locked": False,
        },
    }

    user = mock_users.get(user_id.lower())

    if user is None:
        return {
            "ok": False,
            "error": f"User '{user_id}' not found.",
        }

    return {
        "ok": True,
        "user_id": user_id,
        "enabled": user["enabled"],
        "locked": user["locked"],
    }


def unlock_user(user_id: str) -> dict:
    """
    Propose an account unlock.

    IMPORTANT:
    This function does NOT actually unlock anything yet.
    The real implementation will require approval.
    """

    mock_users = {
        "jdoe",
        "asmith",
        "disabled",
    }

    if user_id.lower() not in mock_users:
        return {
            "ok": False,
            "error": f"User '{user_id}' not found.",
        }

    return {
        "ok": True,
        "user_id": user_id,
        "action": "unlock_user",
        "status": "proposed",
        "message": f"Unlock of '{user_id}' requires human approval.",
    }


def reset_password(user_id: str) -> dict:
    """
    Propose a password reset.

    IMPORTANT:
    This function does NOT actually reset anything yet.
    The real implementation will require approval.
    """

    mock_users = {
        "jdoe",
        "asmith",
        "disabled",
    }

    if user_id.lower() not in mock_users:
        return {
            "ok": False,
            "error": f"User '{user_id}' not found.",
        }

    return {
        "ok": True,
        "user_id": user_id,
        "action": "reset_password",
        "status": "proposed",
        "message": f"Password reset for '{user_id}' requires human approval.",
    }