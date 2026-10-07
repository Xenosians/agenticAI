#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys

from pathlib import Path


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if (
    str(REPOSITORY_ROOT)
    not in sys.path
):
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


from config import get_settings


def main() -> None:
    settings = get_settings()

    print("Model serving cache")
    print("===================")
    print(
        "shared_base_enabled:",
        settings.model_shared_base_enabled,
    )
    print(
        "artifact_cache_enabled:",
        settings.model_artifact_cache_enabled,
    )
    print(
        "artifact_cache_root:",
        settings.model_artifact_cache_root,
    )
    print(
        "materialize_quantized:",
        settings.model_artifact_cache_materialize_quantized,
    )
    print(
        "prefetch:",
        settings.model_artifact_cache_prefetch,
    )
    print(
        "max_loaded_models:",
        settings.model_max_loaded_models,
    )
    print(
        "pin_hub:",
        settings.model_pin_hub,
    )

    index = (
        settings
        .model_artifact_cache_root
        / "index.sqlite3"
    )

    print(
        "index:",
        index,
    )

    if not index.is_file():
        print("artifacts: none yet")
        return

    with sqlite3.connect(
        index
    ) as conn:
        rows = conn.execute(
            """
            SELECT
                cache_key,
                state,
                cache_path,
                last_used_at,
                error
            FROM model_artifacts
            ORDER BY last_used_at DESC
            """
        ).fetchall()

    print(
        "artifacts:",
        len(rows),
    )

    for (
        cache_key,
        state,
        cache_path,
        _last_used_at,
        error,
    ) in rows:
        print(
            "-",
            cache_key[:16],
            state,
            cache_path,
        )

        if error:
            print(
                "  error:",
                error,
            )


if __name__ == "__main__":
    main()
