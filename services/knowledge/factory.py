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

    raise RuntimeError(
        "Unsupported knowledge backend: "
        f"{backend}"
    )
