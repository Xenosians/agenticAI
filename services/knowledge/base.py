from abc import (
    ABC,
    abstractmethod,
)

from .types import (
    KnowledgeLookupResult,
    KnowledgeSearchQuery,
    KnowledgeSearchResult,
)


class KnowledgeService(
    ABC
):

    @abstractmethod
    def search(
        self,
        query: KnowledgeSearchQuery,
    ) -> KnowledgeSearchResult:
        raise NotImplementedError

    @abstractmethod
    def get_knowledge(
        self,
        document_id: str,
    ) -> KnowledgeLookupResult:
        raise NotImplementedError

    @abstractmethod
    def get_runbook(
        self,
        runbook_id: str,
    ) -> KnowledgeLookupResult:
        raise NotImplementedError
