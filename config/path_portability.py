from __future__ import annotations

import os
import re

from pathlib import (
    Path,
    PurePosixPath,
    PureWindowsPath,
)


_WSL_DRIVE_PATH = re.compile(
    r"^/mnt/([A-Za-z])(?:/(.*))?$"
)

_WINDOWS_DRIVE_PATH = re.compile(
    r"^([A-Za-z]):(?:/(.*))?$"
)


def translate_portable_path_text(
    value: str | Path,
    *,
    target_os: str | None = None,
) -> str:
    """
    Translate the two repository path forms that can legitimately
    refer to the same physical Windows-hosted artifact.

    Windows:
        /mnt/c/project/example
        -> C:\\project\\example

    WSL/Linux:
        C:\\project\\example
        -> /mnt/c/project/example

    The serialized artifact itself is not rewritten. Translation
    happens only at the runtime boundary, preserving immutable
    manifest hashes and provenance.
    """

    raw = str(
        value
    ).strip()

    if not raw:
        raise ValueError(
            "Portable path cannot be empty."
        )

    normalized = (
        raw.replace(
            "\\",
            "/",
        )
    )

    os_name = (
        target_os
        if target_os is not None
        else os.name
    )

    if os_name == "nt":

        match = (
            _WSL_DRIVE_PATH
            .match(
                normalized
            )
        )

        if match is not None:

            drive = (
                match.group(1)
                .upper()
            )

            remainder = (
                match.group(2)
                or ""
            )

            parts = [
                part

                for part
                in remainder.split("/")

                if part
            ]

            return str(
                PureWindowsPath(
                    f"{drive}:\\",
                    *parts,
                )
            )

    else:

        match = (
            _WINDOWS_DRIVE_PATH
            .match(
                normalized
            )
        )

        if match is not None:

            drive = (
                match.group(1)
                .lower()
            )

            remainder = (
                match.group(2)
                or ""
            )

            parts = [
                part

                for part
                in remainder.split("/")

                if part
            ]

            return str(
                PurePosixPath(
                    "/mnt",
                    drive,
                    *parts,
                )
            )

    return raw


def resolve_portable_path(
    value: str | Path,
    *,
    base: Path | None = None,
) -> Path:
    """
    Return a native Path for the current process.

    Relative paths are resolved against `base` when supplied.
    Cross-platform absolute paths are translated before pathlib
    performs native resolution.
    """

    translated = (
        translate_portable_path_text(
            value
        )
    )

    path = (
        Path(
            translated
        )
        .expanduser()
    )

    if not path.is_absolute():

        anchor = (
            base
            if base is not None
            else Path.cwd()
        )

        path = (
            anchor
            / path
        )

    return path.resolve()
