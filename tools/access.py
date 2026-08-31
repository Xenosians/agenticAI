from services.directory import get_directory_service


def check_access(
    user_id: str,
    resource: str,
) -> dict:
    directory = get_directory_service()

    return directory.check_access(
        user_id,
        resource,
    )