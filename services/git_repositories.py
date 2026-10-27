from __future__ import annotations

import re

from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

from config import (
    get_settings,
)

from services.process_runner import (
    workspace_root,
)


REPOSITORY_ALIAS_PATTERN = re.compile(
    r"^[a-z][a-z0-9_-]{0,31}$"
)


@dataclass(
    frozen=True
)
class GitRepositoryTarget:
    """
    One trusted logical repository target.

    `name` is model-visible.

    `path` is trusted runtime state and is never selected directly
    by the model.
    """

    name: str
    path: Path


class GitRepositoryRegistry:
    """
    Resolve bounded logical repository identifiers into trusted
    repository paths.

    The model may select a configured logical identifier.

    The model cannot:
    - provide a filesystem path;
    - escape the configured workspace;
    - register another repository;
    - redirect Git execution elsewhere.
    """

    def __init__(
        self,
        *,
        root: Path,
        repositories: dict[
            str,
            Path,
        ],
        default_repository: str,
    ) -> None:

        self.root = (
            root
            .expanduser()
            .resolve()
        )

        if not (
            self.root.is_dir()
        ):
            raise ValueError(
                "Git workspace root "
                "does not exist."
            )

        if not isinstance(
            repositories,
            dict,
        ):
            raise ValueError(
                "Git repository configuration "
                "must be a mapping."
            )

        normalized: dict[
            str,
            Path,
        ] = {}

        for (
            raw_name,
            raw_path,
        ) in repositories.items():

            name = (
                self._normalize_name(
                    raw_name
                )
            )

            if name in normalized:
                raise ValueError(
                    "Duplicate Git repository "
                    f"identifier '{name}'."
                )

            if isinstance(
                raw_path,
                str,
            ):
                configured_path = (
                    Path(
                        raw_path
                    )
                )

            elif isinstance(
                raw_path,
                Path,
            ):
                configured_path = (
                    raw_path
                )

            else:
                raise ValueError(
                    "Configured Git repository "
                    f"'{name}' has an invalid path."
                )

            normalized[
                name
            ] = (
                configured_path
            )

        if not normalized:
            raise ValueError(
                "At least one Git repository "
                "must be configured."
            )

        default_name = (
            self._normalize_name(
                default_repository
            )
        )

        if (
            default_name
            not in normalized
        ):
            raise ValueError(
                "Default Git repository "
                f"'{default_name}' is not configured."
            )

        self.repositories = (
            normalized
        )

        self.default_repository = (
            default_name
        )

    @staticmethod
    def _normalize_name(
        value: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Repository identifier "
                "must be a string."
            )

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:
            raise ValueError(
                "Repository identifier "
                "must not be empty."
            )

        if (
            REPOSITORY_ALIAS_PATTERN
            .fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                "Repository identifier "
                "contains unsupported characters."
            )

        return (
            normalized
        )

    def available(
        self,
    ) -> list[str]:

        return (
            sorted(
                self.repositories
                .keys()
            )
        )

    def aliases(
        self,
    ) -> dict[
        str,
        list[str],
    ]:
        """
        Folder names that identify exactly one configured repository.

        People name a repository by its folders, for example
        "agenticaiPersonal" or "itsm-agent" for the repository
        configured as "ai" at agenticaiPersonal/itsm-agent. Each folder
        on a configured relative path is an alias of that repository,
        unless it is a canonical name itself or lies on the path of
        another repository too.

        A repository inside a hidden folder (for example a runtime
        workspace under ".runtime") is internal: its surrounding folder
        names are generic words such as "workspace", so it is named only
        by its configured name.

        Aliases name configured repositories only; they never add one.
        """

        folders: dict[
            str,
            list[str],
        ] = {}

        for (
            name,
            configured_path,
        ) in self.repositories.items():

            requested = (
                configured_path
                .expanduser()
            )

            candidate = (
                requested.resolve()
                if requested.is_absolute()
                else (
                    self.root
                    / requested
                ).resolve()
            )

            # A path outside the workspace is refused by resolve(); it
            # contributes no names either.
            parts = (
                candidate.relative_to(
                    self.root
                ).parts
                if candidate.is_relative_to(
                    self.root
                )
                else ()
            )

            folders[name] = (
                []
                if any(
                    part.startswith(".")
                    for part in parts
                )
                else [
                    part.lower()
                    for part in parts
                    if REPOSITORY_ALIAS_PATTERN.fullmatch(part.lower())
                ]
            )

        owners: dict[
            str,
            set[str],
        ] = {}

        for name, names in folders.items():
            for alias in names:
                owners.setdefault(alias, set()).add(name)

        return {
            name: sorted({
                alias
                for alias in folders[name]
                if owners[alias] == {name}
                and alias not in self.repositories
            })
            for name in self.available()
        }

    def resolve(
        self,
        repository: (
            str | None
        ) = None,
    ) -> GitRepositoryTarget:

        if repository is None:
            name = (
                self.default_repository
            )

        else:
            name = (
                self._normalize_name(
                    repository
                )
            )

            if name not in self.repositories:
                name = next(
                    (
                        canonical
                        for canonical, aliases in self.aliases().items()
                        if name in aliases
                    ),
                    name,
                )

        configured_path = (
            self.repositories.get(
                name
            )
        )

        if configured_path is None:
            available = (
                ", ".join(
                    self.available()
                )
            )

            raise ValueError(
                f"Unknown Git repository '{name}'. "
                f"Available repositories: {available}."
            )

        requested = (
            configured_path
            .expanduser()
        )

        if requested.is_absolute():
            candidate = (
                requested.resolve()
            )

        else:
            candidate = (
                self.root
                / requested
            ).resolve()

        if (
            candidate != self.root
            and self.root
            not in candidate.parents
        ):
            raise ValueError(
                "Configured Git repository "
                "escapes the approved workspace."
            )

        if not (
            candidate.is_dir()
        ):
            raise ValueError(
                f"Configured Git repository '{name}' "
                "does not exist."
            )

        git_marker = (
            candidate
            / ".git"
        )

        if not (
            git_marker.exists()
        ):
            raise ValueError(
                f"Configured Git repository '{name}' "
                "is not a Git repository."
            )

        return (
            GitRepositoryTarget(
                name=name,
                path=candidate,
            )
        )


def build_git_repository_registry(
) -> GitRepositoryRegistry:
    """
    Build the runtime Git repository registry from validated
    application configuration.
    """

    settings = (
        get_settings()
    )

    return (
        GitRepositoryRegistry(
            root=(
                workspace_root()
            ),

            repositories=(
                settings
                .git_repositories
            ),

            default_repository=(
                settings
                .git_default_repository
            ),
        )
    )


def resolve_git_repository(
    repository: (
        str | None
    ) = None,
) -> GitRepositoryTarget:

    registry = (
        build_git_repository_registry()
    )

    return (
        registry.resolve(
            repository
        )
    )