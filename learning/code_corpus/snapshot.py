from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from learning.code_corpus.policy import (
    detect_language,
    should_include_file,
)
from learning.code_corpus.types import (
    SourceSnapshot,
)


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _run_git(
    *,
    root: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [
                "git",
                "-C",
                str(root),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def _git_snapshot(
    *,
    root: Path,
    source_kind: str,
    logical_repository: str,
    allow_dirty: bool,
) -> SourceSnapshot | None:
    head = _run_git(
        root=root,
        args=[
            "rev-parse",
            "--verify",
            "HEAD",
        ],
    )

    if (
        head is None
        or head.returncode != 0
    ):
        return None

    commit_sha = head.stdout.strip()

    if not commit_sha:
        return None

    status = _run_git(
        root=root,
        args=[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
    )

    if (
        status is None
        or status.returncode != 0
    ):
        raise ValueError(
            "Unable to inspect Git working-tree state."
        )

    changed = [
        line
        for line in status.stdout.splitlines()
        if line.strip()
    ]

    dirty = bool(
        changed
    )

    training_eligible = (
        not dirty
        or allow_dirty
    )

    return SourceSnapshot(
        source_kind=source_kind,
        logical_repository=logical_repository,
        root=str(
            root
        ),
        revision_kind="git",
        revision_id=commit_sha,
        git_commit_sha=commit_sha,
        git_dirty=dirty,
        git_changed_path_count=len(
            changed
        ),
        training_eligible=training_eligible,
    )


def _content_snapshot(
    *,
    root: Path,
    source_kind: str,
    logical_repository: str,
    allow_unversioned: bool,
) -> SourceSnapshot:
    records: list[str] = []

    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
    ):
        relative = path.relative_to(
            root
        )

        if not should_include_file(
            relative_path=relative,
            source_kind=source_kind,
        ):
            continue

        if detect_language(
            relative
        ) is None:
            continue

        records.append(
            relative.as_posix()
            + "\x1f"
            + _sha256_file(
                path
            )
        )

    material = "\n".join(
        records
    ).encode(
        "utf-8"
    )

    revision_id = hashlib.sha256(
        material
    ).hexdigest()

    return SourceSnapshot(
        source_kind=source_kind,
        logical_repository=logical_repository,
        root=str(
            root
        ),
        revision_kind="content_snapshot",
        revision_id=revision_id,
        git_commit_sha=None,
        git_dirty=False,
        git_changed_path_count=0,
        training_eligible=allow_unversioned,
    )


def resolve_source_snapshot(
    *,
    root: Path,
    source_kind: str,
    logical_repository: str,
    allow_dirty: bool = False,
    allow_unversioned: bool = False,
) -> SourceSnapshot:
    resolved = root.expanduser().resolve()

    if not resolved.is_dir():
        raise ValueError(
            f"Corpus source root does not exist: {resolved}"
        )

    git = _git_snapshot(
        root=resolved,
        source_kind=source_kind,
        logical_repository=logical_repository,
        allow_dirty=allow_dirty,
    )

    if git is not None:
        return git

    return _content_snapshot(
        root=resolved,
        source_kind=source_kind,
        logical_repository=logical_repository,
        allow_unversioned=allow_unversioned,
    )
