from __future__ import annotations

from services.workspace_repositories import (
    available_workspace_repositories,
)


WORKSPACE_REPOSITORY_PARAMETER = {
    "type":
        "str",

    "description": (
        "Configured logical repository identifier such as "
        "ai, backend, or frontend. Set it only when the user "
        "names a repository; omit it otherwise so the server's "
        "configured default repository applies. Never guess one. "
        "Do NOT place a repository identifier inside relative_path."
    ),
}


def resolve_workspace_argument_values(
) -> dict[
    str,
    list[str],
]:
    """
    Expose only trusted logical repository identifiers.

    These values are model-facing bounded metadata only.

    They do not authorize filesystem access and do not replace
    workspace confinement, SemanticGuard, or ToolGateway policy.
    """

    return {
        "repository":
            available_workspace_repositories(),
    }
