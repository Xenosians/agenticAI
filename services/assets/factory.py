from config import (
    Settings,
)

from .base import (
    AssetService,
)

from .mock import (
    MockAssetService,
)


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

    raise RuntimeError(
        "Unsupported asset backend: "
        f"{backend}"
    )
