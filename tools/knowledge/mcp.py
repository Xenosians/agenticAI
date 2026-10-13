from pydantic import ValidationError

from mcp.server import (
    MCPServer,
)

from services.knowledge import (
    KnowledgeDocument as KnowledgeDocumentMCPResult,
    KnowledgeLookupResult as KnowledgeLookupMCPResult,
    KnowledgeSearchHit as KnowledgeSearchHitMCPResult,
    KnowledgeSearchQuery,
    KnowledgeSearchResult as KnowledgeSearchMCPResult,
    KnowledgeService,
)


def register_knowledge_tools(
    server: MCPServer,
    knowledge: KnowledgeService,
) -> None:

    @server.tool()
    def knowledge_search(
        query: str,
        kind: str | None = None,
        limit: int | None = None,
    ) -> KnowledgeSearchMCPResult:

        try:

            search_query = (
                KnowledgeSearchQuery(
                    query=query,

                    kind=kind,

                    limit=(
                        10
                        if limit is None
                        else limit
                    ),
                )
            )

        except ValidationError:

            return (
                KnowledgeSearchMCPResult(
                    ok=False,

                    status="denied",

                    error=(
                        "Invalid knowledge search query."
                    ),
                )
            )

        result = (
            knowledge
            .search(
                search_query
            )
        )

        return (
            KnowledgeSearchMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def knowledge_get(
        document_id: str,
    ) -> KnowledgeLookupMCPResult:

        result = (
            knowledge
            .get_knowledge(
                document_id
            )
        )

        return (
            KnowledgeLookupMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def runbook_get(
        runbook_id: str,
    ) -> KnowledgeLookupMCPResult:

        result = (
            knowledge
            .get_runbook(
                runbook_id
            )
        )

        return (
            KnowledgeLookupMCPResult(
                **result.model_dump()
            )
        )
