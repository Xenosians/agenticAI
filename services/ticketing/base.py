from abc import (
    ABC,
    abstractmethod,
)

from .types import (
    TicketAssignedResult,
    TicketCommentsResult,
    TicketHistoryResult,
    TicketLinksResult,
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

    @abstractmethod
    def get_ticket_links(
        self,
        ticket_key: str,
    ) -> TicketLinksResult:
        raise NotImplementedError

    @abstractmethod
    def get_assigned_tickets(
        self,
        assignee: str,
        *,
        status: str | None = None,
        limit: int = 10,
    ) -> TicketAssignedResult:
        raise NotImplementedError