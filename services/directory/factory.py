from config import (
    Settings,
)

from .base import (
    DirectoryService,
)

from .ldap import (
    LdapDirectoryService,
)

from .mock import (
    MockDirectoryService,
)


def build_directory_service(
    settings: Settings,
) -> DirectoryService:
    """
    Build one directory-service implementation from validated
    application settings.

    This function does not cache or globally own the service.

    The caller is responsible for lifecycle ownership.
    """

    backend = (
        settings
        .directory_backend
    )

    if backend == "unconfigured":
        raise RuntimeError(
            "DIRECTORY_BACKEND is not configured. Select ldap for real directory "
            "operations, or explicitly select mock for a simulated demo/test."
        )

    if (
        backend
        == "mock"
    ):
        return (
            MockDirectoryService()
        )

    if (
        backend
        == "ldap"
    ):
        return (
            LdapDirectoryService(
                settings=(
                    settings
                ),
            )
        )

    raise RuntimeError(
        "Unsupported directory backend: "
        f"{backend}"
    )
