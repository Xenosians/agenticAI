from __future__ import annotations

from services.git_repositories import (
    GitRepositoryTarget,
    build_git_repository_registry,
)


def available_workspace_repositories(
) -> list[str]:
    """
    Return configured logical repository identifiers available to
    workspace capabilities.

    Workspace discovery intentionally shares the canonical logical
    repository registry with governed Git.

    This prevents separate aliases such as:

        backend
        backend-repo
        backend-workspace

    from drifting into different meanings.
    """

    registry = (
        build_git_repository_registry()
    )

    return (
        registry.available()
    )


def workspace_repository_aliases(
) -> dict[
    str,
    list[str],
]:
    """
    Configured repositories with the folder names people call them,
    for example "ai" with "agenticaipersonal" and "itsm-agent".

    Folder names only name configured repositories; they never add one.
    """

    return (
        build_git_repository_registry()
        .aliases()
    )


def resolve_workspace_repository(
    repository: (
        str
        | None
    ) = None,
) -> GitRepositoryTarget:
    """
    Resolve one logical repository identifier into trusted runtime
    state.

    The model may choose only a configured logical identifier.

    It never supplies or controls the resulting filesystem root.
    """

    registry = (
        build_git_repository_registry()
    )

    return (
        registry.resolve(
            repository
        )
    )
