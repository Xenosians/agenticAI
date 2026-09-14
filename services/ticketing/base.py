from abc import (
    ABC,
    abstractmethod,
)

from .types import (
    TicketLookupResult,
)


class TicketService(
    ABC
):

    @abstractmethod
    def get_ticket(
        self,
        ticket_key: str,
    ) -> TicketLookupResult:
        """
        Retrieve one ticket by its provider-visible identifier.
        """
        raise NotImplementedError
