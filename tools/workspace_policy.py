from pathlib import Path


ALLOWED_TEXT_SUFFIXES = {
    ".css",
    ".ex",
    ".exs",
    ".html",
    ".js",
    ".json",
    ".md",
    ".nim",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


ALLOWED_EXACT_TEXT_FILES = {
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README",
}


DENIED_NAME_FRAGMENTS = {
    "credential",
    "password",
    "secret",
    "token",
}


DENIED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def is_sensitive_name(
    name: str,
) -> bool:
    """
    Return True when a path component looks like a sensitive
    runtime/configuration file that discovery should not expose.
    """

    normalized = (
        name.strip().lower()
    )

    if not normalized:
        return False

    # The tracked example file is intentionally safe to expose.
    if normalized == ".env.example":
        return False

    if (
        normalized == ".env"
        or normalized.startswith(
            ".env."
        )
    ):
        return True

    return any(
        fragment in normalized
        for fragment
        in DENIED_NAME_FRAGMENTS
    )


def validate_discoverable_path(
    relative_path: str,
) -> tuple[
    bool,
    str | None,
]:
    """
    Validate whether a workspace-relative path may be surfaced
    through discovery tools.

    This is metadata/discovery policy only. Existence and
    workspace-boundary checks are handled separately.
    """

    path = Path(
        relative_path
    )

    for part in path.parts:
        normalized = (
            part.lower()
        )

        if (
            normalized
            in DENIED_DIRECTORY_NAMES
        ):
            return (
                False,
                (
                    "The requested path is blocked "
                    "by the workspace discovery policy."
                ),
            )

        if is_sensitive_name(
            part
        ):
            return (
                False,
                (
                    "The requested path is blocked "
                    "by the sensitive-file policy."
                ),
            )

    return True, None


def validate_readable_text_path(
    relative_path: str,
) -> tuple[
    bool,
    str | None,
]:
    """
    Validate whether a discovered workspace file may be read as
    UTF-8 source/text content.
    """

    (
        discoverable,
        discoverable_error,
    ) = validate_discoverable_path(
        relative_path
    )

    if not discoverable:
        return (
            False,
            discoverable_error,
        )

    path = Path(
        relative_path
    )

    if (
        path.name
        in ALLOWED_EXACT_TEXT_FILES
    ):
        return True, None

    suffix = (
        path.suffix.lower()
    )

    if (
        suffix
        not in ALLOWED_TEXT_SUFFIXES
    ):
        return (
            False,
            (
                "The requested file type is not "
                "allowed by the current text-read policy."
            ),
        )

    return True, None