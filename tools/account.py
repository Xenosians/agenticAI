from services.directory import get_directory_service


def account_status(
    user_id: str,
) -> dict:
    directory = get_directory_service()

    return directory.account_status(
        user_id
    )


def unlock_user(
    user_id: str,
) -> dict:
    directory = get_directory_service()

    return directory.unlock_user(
        user_id
    )


def reset_password(
    user_id: str,
) -> dict:
    directory = get_directory_service()

    return directory.reset_password(
        user_id
    )