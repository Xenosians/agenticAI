from config import get_env

from .ldap import LdapDirectoryService
from .mock import MockDirectoryService


_directory_service = None


def get_directory_service():
    global _directory_service

    if _directory_service is not None:
        return _directory_service

    backend = (
        get_env(
            "DIRECTORY_BACKEND",
            "mock",
        )
        or "mock"
    ).strip().lower()

    if backend == "mock":
        _directory_service = MockDirectoryService()

    elif backend == "ldap":
        _directory_service = LdapDirectoryService()

    else:
        raise RuntimeError(
            f"Unsupported DIRECTORY_BACKEND: {backend}"
        )

    return _directory_service