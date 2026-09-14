from abc import (
    ABC,
    abstractmethod,
)

from .types import (
    TicketCommentsResult,
    TicketHistoryResult,
    TicketLookupResult,
    TicketSearchQuery,
    TicketSearchResult,
)


class TicketService(
    ABC
):

    @abstractmethod
    def get_ticket(
        self,
        ticket_key: str,
    ) -> TicketLookupResult:
        raise NotImplementedError

    @abstractmethod
    def search_tickets(
        self,
        query: TicketSearchQuery,
    ) -> TicketSearchResult:
        raise NotImplementedError

    @abstractmethod
    def get_ticket_history(
        self,
        ticket_key: str,
        *,
        limit: int = 20,
    ) -> TicketHistoryResult:
        raise NotImplementedError

    @abstractmethod
    def get_ticket_comments(
        self,
        ticket_key: str,
        *,
        limit: int = 20,
    ) -> TicketCommentsResult:
        raise NotImplementedError