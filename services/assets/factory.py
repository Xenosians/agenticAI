from config import (
    Settings,
)

from .base import (
    AssetService,
)

from .mock import (
    MockAssetService,
)
from .sqlite import SQLiteAssetService


def build_asset_service(
    settings: Settings,
) -> AssetService:
    """
    Build one asset inventory provider from validated
    application configuration.

    Model-facing capabilities depend only on AssetService.

    Provider selection is deployment configuration rather than
    orchestration or business-logic state.
    """

    backend = (
        settings
        .asset_backend
    )

    if backend == "mock":

        return (
            MockAssetService()
        )

    if backend == "sqlite":
        if settings.asset_sqlite_path is None:
            raise RuntimeError("ASSET_SQLITE_PATH is required for the SQLite asset backend.")
        return SQLiteAssetService(settings.asset_sqlite_path)

    raise RuntimeError(
        "Unsupported asset backend: "
        f"{backend}"
    )
