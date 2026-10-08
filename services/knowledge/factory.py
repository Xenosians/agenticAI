from config import (
    Settings,
)

from .base import (
    KnowledgeService,
)

from .mock import (
    MockKnowledgeService,
)


def build_knowledge_service(
    settings: Settings,
) -> KnowledgeService:
    """
    Build one trusted knowledge provider from validated
    application configuration.

    Model-facing knowledge capabilities remain independent from
    the underlying storage or retrieval provider.
    """

    backend = (
        settings
        .knowledge_backend
    )

    if backend == "mock":

        return (
            MockKnowledgeService()
        )

    if backend == "local":
        from pathlib import Path
        from .local import LocalKnowledgeService
        if not settings.knowledge_snapshot_path or not settings.knowledge_snapshot_sha256:
            raise RuntimeError("Local knowledge requires a reviewed snapshot path and SHA256.")
        return LocalKnowledgeService(Path(settings.knowledge_snapshot_path), settings.knowledge_snapshot_sha256)

    raise RuntimeError(
        "Unsupported knowledge backend: "
        f"{backend}"
    )
