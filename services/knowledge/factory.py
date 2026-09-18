from .base import (
    KnowledgeService,
)

from .mock import (
    MockKnowledgeService,
)


def build_knowledge_service(
) -> KnowledgeService:
    """
    Build the current trusted knowledge provider.

    The model-facing capability contract remains independent from
    the underlying storage or retrieval implementation.

    Future providers may use local files, an indexed corpus, RAG,
    Confluence, SharePoint, or another trusted knowledge source.
    """

    return (
        MockKnowledgeService()
    )
