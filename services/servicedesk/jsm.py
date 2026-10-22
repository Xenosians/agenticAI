"""Jira Service Management provider: service desks, request types, SLAs, queues and requests.

Reads use the Service Management REST API (``/rest/servicedeskapi``) with the
same Jira site and credentials as the ticketing provider. Raising a request is
prepared read-only before approval (desk, request type and its required fields
are resolved and frozen), revalidated before the single POST, and verified by
reading the created request back.
"""
from __future__ import annotations

import re
from functools import partial
from typing import Any
from urllib.parse import quote

import httpx

from services.ticketing.jira import JIRA_TICKET_KEY_PATTERN, JiraTicketService

from .types import (
    QueueListResult,
    RequestType,
    RequestTypeField,
    ServiceDesk,
    ServiceDeskQueue,
    ServiceRequestResult,
    SlaMetric,
    SlaStatusResult,
)

API = "/rest/servicedeskapi"
PAGE_SIZE = 50
# Desks and request types are scanned for at most this many pages.
MAX_LOOKUP_PAGES = 10
MAX_QUEUES = 50
MAX_SLA_METRICS = 50
MAX_NAME_LENGTH = 255
MAX_SUMMARY_LENGTH = 255
MAX_DESCRIPTION_LENGTH = 5000
MAX_DETAIL_LENGTH = 300
MAX_FRIENDLY_LENGTH = 50
MAX_LISTED_CHOICES = 10
# The request fields this provider fills; any other required field blocks a request.
FILLABLE_FIELDS = frozenset({"summary", "description"})
OPERATION = "service_request_create"
NOT_CONFIGURED = (
    "Jira Service Management is not configured. It uses the Jira ticketing "
    "settings: TICKETING_BACKEND=jira, JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN."
)
# Codes for a refused request, as opposed to a provider failure.
REFUSAL_CODES = frozenset({
    "ambiguous",
    "invalid_argument",
    "not_allowed",
    "permission_denied",
    "plan_required",
    "unsupported_fields",
})
I18N_KEY_PATTERN = re.compile(r"[a-z0-9_]+(\.[a-z0-9_]+)+", re.IGNORECASE)


class ServiceDeskError(Exception):
    """A service desk failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    @property
    def status(self) -> str:
        if self.code in {"not_found", "rate_limited"}:
            return self.code
        return "denied" if self.code in REFUSAL_CODES else "error"


def _exact(value: object, field_name: str, max_length: int) -> str:
    """User-supplied text, kept exactly as given."""

    if not isinstance(value, str) or not value.strip():
        raise ServiceDeskError("invalid_argument", f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ServiceDeskError("invalid_argument", f"{field_name} must not have leading or trailing whitespace.")
    if len(value) > max_length:
        raise ServiceDeskError("invalid_argument", f"{field_name} exceeds {max_length} characters.")
    return value


def _request_key(value: object) -> str:
    key = _exact(value, "request_key", MAX_NAME_LENGTH)
    if JIRA_TICKET_KEY_PATTERN.fullmatch(key) is None:
        raise ServiceDeskError("invalid_argument", "request_key must be a request key such as SUP-12.")
    return key.upper()


def _provider_text(value: object, field_name: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_NAME_LENGTH:
        raise ServiceDeskError("provider_error", f"Jira returned an invalid {field_name}.")
    return value


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _same_text(current: object, expected: str) -> bool:
    return isinstance(current, str) and current.replace("\r\n", "\n").strip() == expected.replace("\r\n", "\n").strip()


def _response_detail(response: httpx.Response) -> str:
    """The provider's own explanation, when it is readable text."""

    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""

    messages = [payload.get("errorMessage")]
    messages.extend(payload.get("errorMessages") or [])
    errors = payload.get("errors")
    if isinstance(errors, dict):
        messages.extend(f"{name}: {value}" for name, value in errors.items() if isinstance(value, str))
    readable = [
        message.strip()
        for message in messages
        if isinstance(message, str) and message.strip() and not I18N_KEY_PATTERN.fullmatch(message.strip())
    ]
    return " ".join(readable)[:MAX_DETAIL_LENGTH]


def _refusal(response: httpx.Response, action: str, not_found: str | None = None) -> ServiceDeskError | None:
    """Map a non-success response; ``None`` means the request succeeded."""

    code = response.status_code
    if 200 <= code < 300:
        return None

    detail = _response_detail(response)
    suffix = f" {detail}" if detail else ""
    if code == 402:
        return ServiceDeskError("plan_required", f"The Jira plan does not allow this service desk {action}.{suffix}")
    if code == 404:
        return ServiceDeskError("not_found", not_found or f"The service desk {action} target was not found.{suffix}")
    if code in {401, 403}:
        return ServiceDeskError(
            "permission_denied",
            f"Jira refused the service desk {action} for the configured account (HTTP {code}).{suffix}",
        )
    if code == 429:
        return ServiceDeskError("rate_limited", f"Jira rate limited the service desk {action}.")
    if code in {400, 409, 422}:
        return ServiceDeskError("invalid_argument", f"Jira rejected the service desk {action} (HTTP {code}).{suffix}")
    return ServiceDeskError("provider_error", f"Jira returned HTTP {code} during the service desk {action}.")


def _friendly(cycle: dict[str, Any] | None, field_name: str) -> str | None:
    duration = (cycle or {}).get(field_name)
    value = duration.get("friendly") if isinstance(duration, dict) else None
    return value.strip()[:MAX_FRIENDLY_LENGTH] if isinstance(value, str) and value.strip() else None


def _metric(item: dict[str, Any]) -> SlaMetric:
    ongoing = item.get("ongoingCycle") if isinstance(item.get("ongoingCycle"), dict) else None
    completed = [cycle for cycle in item.get("completedCycles") or [] if isinstance(cycle, dict)]
    cycle = ongoing or (completed[-1] if completed else None)

    if ongoing is not None:
        state = "paused" if ongoing.get("paused") is True else "ongoing"
    elif completed:
        state = "completed"
    else:
        state = "not_started"

    return SlaMetric(
        name=_provider_text(item.get("name"), "SLA name"),
        state=state,
        breached=cycle is not None and cycle.get("breached") is True,
        goal=_friendly(cycle, "goalDuration"),
        elapsed=_friendly(cycle, "elapsedTime"),
        remaining=_friendly(cycle, "remainingTime"),
        completed_cycles=len(completed),
    )


def _choices(items: list[dict[str, Any]], field_name: str) -> str:
    names = sorted({str(item[field_name]) for item in items if item.get(field_name)})
    return ", ".join(names[:MAX_LISTED_CHOICES])


class ServiceDeskService:
    """Jira Service Management reads and governed request creation."""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # The Jira ticket client validates the site URL and credentials and turns
        # Jira Cloud's silent anonymous fallback into an authentication failure.
        self._jira = JiraTicketService(
            base_url=base_url,
            email=email,
            api_token=api_token,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        self.client = self._jira.client

    def close(self) -> None:
        self._jira.close()

    def __enter__(self) -> ServiceDeskService:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------ HTTP

    def _get(
        self,
        path: str,
        *,
        action: str,
        params: dict[str, Any] | None = None,
        not_found: str | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.client.get(path, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ServiceDeskError(
                "provider_error",
                f"The service desk {action} request failed ({type(exc).__name__}).",
            ) from exc

        refusal = _refusal(response, action, not_found)
        if refusal is not None:
            raise refusal

        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceDeskError("provider_error", f"Jira returned invalid JSON during the service desk {action}.") from exc
        if not isinstance(payload, dict):
            raise ServiceDeskError("provider_error", f"Jira returned invalid data during the service desk {action}.")
        return payload

    def _page(
        self,
        path: str,
        *,
        action: str,
        params: dict[str, Any] | None = None,
        not_found: str | None = None,
        start: int = 0,
        limit: int = PAGE_SIZE,
    ) -> tuple[list[dict[str, Any]], bool]:
        """One page of values and whether the provider has more."""

        payload = self._get(path, action=action, params={**(params or {}), "start": start, "limit": limit}, not_found=not_found)
        values = payload.get("values")
        if not isinstance(values, list):
            raise ServiceDeskError("provider_error", f"Jira returned an invalid list during the service desk {action}.")
        return [item for item in values if isinstance(item, dict)], payload.get("isLastPage") is False

    def _all(self, path: str, *, action: str) -> tuple[list[dict[str, Any]], bool]:
        """Every value across pages, bounded; the flag tells whether the scan stopped early."""

        items: list[dict[str, Any]] = []
        for _ in range(MAX_LOOKUP_PAGES):
            page, more = self._page(path, action=action, start=len(items))
            items.extend(page)
            if not more or not page:
                return items, False
        return items, True

    # ------------------------------------------------------------ lookups

    def resolve_service_desk(self, service_desk: object) -> ServiceDesk:
        """One service desk by exact project key or name, compared case-insensitively."""

        wanted = _exact(service_desk, "service_desk", MAX_NAME_LENGTH)
        desks, partial_scan = self._all(f"{API}/servicedesk", action="lookup")
        folded = wanted.casefold()
        matches = (
            [desk for desk in desks if str(desk.get("projectKey") or "").casefold() == folded]
            or [desk for desk in desks if str(desk.get("projectName") or "").casefold() == folded]
        )

        if len(matches) > 1:
            raise ServiceDeskError("ambiguous", f"More than one service desk is named {wanted!r}; use its project key.")
        if not matches:
            known = _choices(desks, "projectKey")
            raise ServiceDeskError(
                "not_found",
                f"No service desk has the project key or name {wanted!r}."
                + (f" Service desks: {known}." if known else "")
                + (" Only the first service desks were searched." if partial_scan else ""),
            )

        raw = matches[0]
        return ServiceDesk(
            id=_provider_text(raw.get("id"), "service desk ID"),
            project_id=_provider_text(raw.get("projectId"), "service desk project ID"),
            project_key=_provider_text(raw.get("projectKey"), "service desk project key"),
            project_name=_provider_text(raw.get("projectName"), "service desk name"),
        )

    def resolve_request_type(self, desk: ServiceDesk, request_type: object) -> RequestType:
        """One request type of the desk by exact name, compared case-insensitively."""

        wanted = _exact(request_type, "request_type", MAX_NAME_LENGTH)
        types, partial_scan = self._all(f"{API}/servicedesk/{quote(desk.id, safe='')}/requesttype", action="request type lookup")
        matches = [item for item in types if str(item.get("name") or "").casefold() == wanted.casefold()]

        if len(matches) > 1:
            raise ServiceDeskError("ambiguous", f"More than one request type in {desk.project_key} is named {wanted!r}.")
        if not matches:
            known = _choices(types, "name")
            raise ServiceDeskError(
                "not_found",
                f"The {desk.project_key} service desk has no request type named {wanted!r}."
                + (f" Request types: {known}." if known else "")
                + (" Only the first request types were searched." if partial_scan else ""),
            )

        raw = matches[0]
        return RequestType(
            id=_provider_text(raw.get("id"), "request type ID"),
            name=_provider_text(raw.get("name"), "request type name"),
            can_create=raw.get("canCreateRequest") is not False,
        )

    def request_type_fields(self, desk: ServiceDesk, request_type: RequestType) -> list[RequestTypeField]:
        payload = self._get(
            f"{API}/servicedesk/{quote(desk.id, safe='')}/requesttype/{quote(request_type.id, safe='')}/field",
            action="request type field lookup",
        )
        raw_fields = payload.get("requestTypeFields")
        if not isinstance(raw_fields, list):
            raise ServiceDeskError("provider_error", "Jira returned invalid request type fields.")

        fields = []
        for raw in raw_fields:
            if not isinstance(raw, dict):
                continue
            field_id = _provider_text(raw.get("fieldId"), "request field ID")
            name = raw.get("name")
            fields.append(RequestTypeField(
                field_id=field_id,
                name=name.strip()[:MAX_NAME_LENGTH] if isinstance(name, str) and name.strip() else field_id,
                required=raw.get("required") is True,
            ))
        return fields

    # ------------------------------------------------------------ requests

    def _request_target(
        self,
        service_desk: object,
        request_type: object,
        summary: object,
        description: object,
    ) -> tuple[ServiceDesk, RequestType]:
        """Validate the exact request and resolve where it goes; reads only."""

        _exact(summary, "summary", MAX_SUMMARY_LENGTH)
        if description is not None:
            _exact(description, "description", MAX_DESCRIPTION_LENGTH)

        desk = self.resolve_service_desk(service_desk)
        found = self.resolve_request_type(desk, request_type)
        if not found.can_create:
            raise ServiceDeskError(
                "not_allowed",
                f"The configured account cannot raise {found.name!r} requests in the {desk.project_key} service desk.",
            )

        fields = self.request_type_fields(desk, found)
        blocking = [field.name for field in fields if field.required and field.field_id not in FILLABLE_FIELDS]
        if blocking:
            raise ServiceDeskError(
                "unsupported_fields",
                f"The {found.name!r} request type also requires {', '.join(blocking)}, which this tool cannot "
                "fill. Choose another request type or use the customer portal.",
            )

        field_ids = {field.field_id for field in fields}
        if "summary" not in field_ids:
            raise ServiceDeskError("unsupported_fields", f"The {found.name!r} request type has no summary field.")
        if description is None and any(field.required and field.field_id == "description" for field in fields):
            raise ServiceDeskError("invalid_argument", f"The {found.name!r} request type requires a description.")
        if description is not None and "description" not in field_ids:
            raise ServiceDeskError("invalid_argument", f"The {found.name!r} request type has no description field.")
        return desk, found

    def prepare_request(
        self,
        *,
        service_desk: object,
        request_type: object,
        summary: object,
        description: object = None,
    ) -> dict[str, str]:
        """Read-only preparation: resolve and freeze the desk and request type before approval."""

        desk, found = self._request_target(service_desk, request_type, summary, description)
        return {
            "expected_service_desk_id": desk.id,
            "expected_request_type_id": found.id,
        }

    def create_request(
        self,
        *,
        service_desk: object,
        request_type: object,
        summary: object,
        description: object = None,
        expected_service_desk_id: str | None = None,
        expected_request_type_id: str | None = None,
    ) -> ServiceRequestResult:
        """Raise one approved request once, then verify it by reading it back."""

        result = partial(
            ServiceRequestResult,
            operation=OPERATION,
            service_desk=service_desk if isinstance(service_desk, str) else None,
            request_type=request_type if isinstance(request_type, str) else None,
            summary=summary if isinstance(summary, str) else None,
        )

        if not all(isinstance(value, str) and value for value in (expected_service_desk_id, expected_request_type_id)):
            return result(
                ok=False,
                status="denied",
                error="An approved service request needs the service desk and request type resolved before approval.",
            )

        try:
            desk, found = self._request_target(service_desk, request_type, summary, description)
        except ServiceDeskError as exc:
            return result(ok=False, status=exc.status, error=str(exc), error_code=exc.code)

        if (desk.id, found.id) != (expected_service_desk_id, expected_request_type_id):
            return result(
                ok=False,
                status="denied",
                error="The service desk or request type changed after approval. The approved request will not be raised.",
            )

        target = {"service_desk": desk.project_key, "request_type": found.name}
        values = {"summary": summary}
        if description is not None:
            values["description"] = description

        try:
            response = self.client.post(
                f"{API}/request",
                json={"serviceDeskId": desk.id, "requestTypeId": found.id, "requestFieldValues": values},
            )
        except (httpx.TimeoutException, httpx.TransportError):
            return result(
                ok=False,
                status="outcome_unknown",
                mutation_performed=None,
                error="The service request hit an ambiguous transport failure. Automatic retry is disabled.",
                **target,
            )

        refusal = _refusal(response, "request")
        if refusal is not None:
            if refusal.code == "provider_error":
                return result(
                    ok=False,
                    status="outcome_unknown",
                    mutation_performed=None,
                    error=f"{refusal} The outcome is not trusted and automatic retry is disabled.",
                    **target,
                )
            return result(ok=False, status=refusal.status, error=str(refusal), error_code=refusal.code, **target)

        try:
            created = response.json()
        except ValueError:
            created = None
        key = created.get("issueKey") if isinstance(created, dict) else None
        if not isinstance(key, str) or JIRA_TICKET_KEY_PATTERN.fullmatch(key) is None:
            return result(
                ok=False,
                status="outcome_unknown",
                mutation_performed=True,
                error="Jira accepted the request but did not identify it for verification.",
                **target,
            )

        try:
            current = self._get(f"{API}/request/{quote(key, safe='')}", action="request read-back")
        except ServiceDeskError:
            current = None

        if not self._matches(current, key=key, desk=desk, request_type=found, values=values):
            return result(
                ok=False,
                status="outcome_unknown",
                mutation_performed=True,
                ticket_key=key,
                error=f"Jira created {key}, but reading it back did not confirm the approved desk, request type and text.",
                **target,
            )

        status = (current.get("currentStatus") or {}).get("status")
        return result(
            ok=True,
            status="success",
            changed=True,
            mutation_performed=True,
            verification_ok=True,
            ticket_key=key,
            request_status=status if isinstance(status, str) else None,
            message=f"Raised {key} ({found.name}) in the {desk.project_name} service desk and verified it.",
            **target,
        )

    @staticmethod
    def _matches(
        current: dict[str, Any] | None,
        *,
        key: str,
        desk: ServiceDesk,
        request_type: RequestType,
        values: dict[str, str],
    ) -> bool:
        if not isinstance(current, dict):
            return False
        identity = (str(current.get("issueKey")), str(current.get("serviceDeskId")), str(current.get("requestTypeId")))
        if identity != (key, desk.id, request_type.id):
            return False
        fields = {
            item.get("fieldId"): item.get("value")
            for item in current.get("requestFieldValues") or []
            if isinstance(item, dict)
        }
        return all(_same_text(fields.get(name), value) for name, value in values.items())

    # ------------------------------------------------------------ SLAs and queues

    def sla_status(self, request_key: object) -> SlaStatusResult:
        try:
            key = _request_key(request_key)
        except ServiceDeskError as exc:
            return SlaStatusResult(ok=False, status=exc.status, error=str(exc), error_code=exc.code)

        try:
            items, more = self._page(
                f"{API}/request/{quote(key, safe='')}/sla",
                action="SLA lookup",
                limit=MAX_SLA_METRICS,
                not_found=f"{key} was not found, or it is not a service desk request.",
            )
            metrics = [_metric(item) for item in items]
        except ServiceDeskError as exc:
            return SlaStatusResult(ok=False, status=exc.status, request_key=key, error=str(exc), error_code=exc.code)

        return SlaStatusResult(ok=True, status="success", request_key=key, metrics=metrics, count=len(metrics), truncated=more)

    def queue_list(self, service_desk: object) -> QueueListResult:
        try:
            desk = self.resolve_service_desk(service_desk)
            items, more = self._page(
                f"{API}/servicedesk/{quote(desk.id, safe='')}/queue",
                action="queue lookup",
                params={"includeCount": "true"},
                limit=MAX_QUEUES,
            )
            queues = [
                ServiceDeskQueue(
                    id=_provider_text(item.get("id"), "queue ID"),
                    name=_provider_text(item.get("name"), "queue name"),
                    issue_count=_count(item.get("issueCount")),
                    jql=item.get("jql") if isinstance(item.get("jql"), str) else None,
                )
                for item in items
            ]
        except ServiceDeskError as exc:
            return QueueListResult(
                ok=False,
                status=exc.status,
                service_desk=service_desk if isinstance(service_desk, str) else None,
                error=str(exc),
                error_code=exc.code,
            )

        return QueueListResult(
            ok=True,
            status="success",
            service_desk=desk.project_key,
            service_desk_name=desk.project_name,
            queues=queues,
            count=len(queues),
            truncated=more,
        )


def build_service_desk_service(settings: Any) -> ServiceDeskService | None:
    """The service desk provider, or ``None`` when Jira ticketing is not configured."""

    if settings.ticketing_backend != "jira":
        return None
    if not all((settings.jira_base_url, settings.jira_email, settings.jira_api_token)):
        return None
    return ServiceDeskService(
        base_url=settings.require_jira_base_url(),
        email=settings.require_jira_email(),
        api_token=settings.require_jira_api_token(),
        timeout_seconds=settings.jira_http_timeout_seconds,
    )
