from config import get_settings

from .base import DirectoryService
from .ldap import LdapDirectoryService
from .mock import MockDirectoryService


_directory_service: (
    DirectoryService | None
) = None


def get_directory_service(
) -> DirectoryService:
    global _directory_service

    if _directory_service is not None:
        return _directory_service

    settings = get_settings()

    backend = (
        settings.directory_backend
    )

    if backend == "mock":
        _directory_service = (
            MockDirectoryService()
        )

    elif backend == "ldap":
        _directory_service = (
            LdapDirectoryService(
                settings=settings,
            )
        )

    else:
        raise RuntimeError(
            "Unsupported directory backend: "
            f"{backend}"
        )

    return _directory_service