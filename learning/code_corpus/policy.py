from __future__ import annotations

from pathlib import Path


ALLOWED_SUFFIXES = {
    ".py",
    ".ex",
    ".exs",
    ".nim",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
}

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".runtime",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "deps",
    "_build",
    "build",
    "dist",
    "coverage",
    "htmlcov",
    "Models",
    "models",
}

EXCLUDED_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}

PROJECT_MARKDOWN_ALLOWED_PREFIXES = (
    "subagents/agents/",
    "subagents/prompts/",
    "docs/",
)


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ex": "elixir",
    ".exs": "elixir",
    ".nim": "nim",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}


def normalize_relative_path(
    path: Path,
) -> str:
    return path.as_posix().lstrip("./")


def detect_language(
    path: Path,
) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(
        path.suffix.lower()
    )


def should_include_file(
    *,
    relative_path: Path,
    source_kind: str,
) -> bool:
    normalized = normalize_relative_path(
        relative_path
    )

    if any(
        part in EXCLUDED_DIRECTORY_NAMES
        for part in relative_path.parts
    ):
        return False

    if relative_path.name in EXCLUDED_FILE_NAMES:
        return False

    suffix = relative_path.suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        return False

    if (
        source_kind == "project"
        and suffix == ".md"
        and not normalized.startswith(
            PROJECT_MARKDOWN_ALLOWED_PREFIXES
        )
    ):
        return False

    if normalized.endswith(
        (".min.js", ".min.css")
    ):
        return False

    return True
