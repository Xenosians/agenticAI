from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from config import Settings


ATLASSIAN_API_ORIGIN = "https://api.atlassian.com"
DEFAULT_PAGE_LIMIT = 25
MAX_PAGE_LIMIT = 50
_ALLOWED_PATH_PREFIXES = ("/admin/",)


@dataclass(frozen=True)
class AtlassianProviderError(Exception):
    """Sanitized provider error. Never stores response bodies or secrets."""

    code: str
    message: str
    status_code: int | None = None
    retry_after_seconds: int | None = None

    def __str__(self) -> str:
        return self.message


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def bounded_limit(value: int | None, *, default: int = DEFAULT_PAGE_LIMIT) -> int:
    if value is None:
        return default
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > MAX_PAGE_LIMIT
    ):
        raise ValueError(
            f"limit must be an integer between 1 and {MAX_PAGE_LIMIT}."
        )
    return value


def extract_next_cursor(value: object) -> str | None:
    """Parse a cursor without ever following provider-supplied links."""
    value = _optional_string(value)
    if value is None:
        return None

    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        cursors = parse_qs(parsed.query).get("cursor")
        if cursors and cursors[0]:
            return cursors[0]
        return None

    if any(marker in value for marker in ("/", "?", "#")):
        return None

    return value


def response_next_cursor(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    links = payload.get("links")
    if not isinstance(links, dict):
        return None
    return extract_next_cursor(links.get("next"))


def _retry_after_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        result = int(value.strip())
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def error_from_status(
    *,
    operation: str,
    status_code: int,
    retry_after: str | None = None,
) -> AtlassianProviderError:
    if status_code == 400:
        code = "bad_request"
        message = f"Atlassian rejected the {operation} request."
    elif status_code == 401:
        code = "authentication_failed"
        message = "Atlassian authentication failed."
    elif status_code == 403:
        code = "authorization_failed"
        message = "Atlassian authorization denied this operation."
    elif status_code == 404:
        code = "not_found"
        message = "The requested Atlassian resource was not found."
    elif status_code == 409:
        code = "conflict"
        message = f"Atlassian reported a conflict during {operation}."
    elif status_code == 429:
        return AtlassianProviderError(
            code="rate_limited",
            message="Atlassian rate-limited the request.",
            status_code=status_code,
            retry_after_seconds=_retry_after_seconds(retry_after),
        )
    elif status_code >= 500:
        code = "provider_unavailable"
        message = f"Atlassian returned HTTP {status_code} during {operation}."
    else:
        code = "provider_error"
        message = f"Atlassian returned HTTP {status_code} during {operation}."

    return AtlassianProviderError(
        code=code,
        message=message,
        status_code=status_code,
    )


@dataclass(frozen=True)
class AtlassianCredentialBroker:
    """
    Trusted credential broker.

    Secrets may be used to construct provider authentication, but secret
    values are never returned by model-visible status methods.
    """

    admin_api_key: str | None = None
    default_org_id: str | None = None
    jira_email: str | None = None
    jira_api_token: str | None = None

    def admin_headers(self) -> dict[str, str]:
        secret = _optional_string(self.admin_api_key)
        if secret is None:
            raise AtlassianProviderError(
                code="credential_not_configured",
                message="ATLASSIAN_ADMIN_API_KEY is not configured.",
            )
        return {
            "Authorization": f"Bearer {secret}",
            "Accept": "application/json",
        }

    def resolve_org_id(self, supplied_org_id: str | None) -> str:
        value = _optional_string(
            supplied_org_id if supplied_org_id is not None else self.default_org_id
        )
        if value is None:
            raise AtlassianProviderError(
                code="organization_not_configured",
                message=(
                    "No Atlassian organization ID was supplied and "
                    "ATLASSIAN_ORG_ID is not configured."
                ),
            )
        return value

    def redacted_status(self) -> dict[str, Any]:
        admin_configured = _optional_string(self.admin_api_key) is not None
        jira_configured = (
            _optional_string(self.jira_email) is not None
            and _optional_string(self.jira_api_token) is not None
        )
        return {
            "ok": True,
            "status": "success",
            "admin_api_key_configured": admin_configured,
            "jira_basic_configured": jira_configured,
            "default_org_configured": (
                _optional_string(self.default_org_id) is not None
            ),
            "admin_auth_type": "api_key" if admin_configured else None,
            "jira_auth_type": "basic_api_token" if jira_configured else None,
            "secret_values_exposed": False,
        }


class AtlassianAdminClient:
    """
    Fixed-origin trusted Atlassian Admin API transport.

    The model never controls the host, URL, Authorization header, or raw
    response handling. Redirects are disabled so credentials cannot be
    forwarded to another origin.
    """

    def __init__(
        self,
        *,
        credentials: AtlassianCredentialBroker,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError("Atlassian timeout must be greater than zero.")

        self.credentials = credentials
        self.client = httpx.Client(
            base_url=ATLASSIAN_API_ORIGIN,
            timeout=float(timeout_seconds),
            follow_redirects=False,
            transport=transport,
            headers={
                "Accept": "application/json",
                "User-Agent": "agentic-itsm/0.1",
            },
        )

    @staticmethod
    def _trusted_path(path: str) -> str:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("Atlassian path must be absolute-path form.")
        if "://" in path or "\\" in path or "\x00" in path:
            raise ValueError("Atlassian path is invalid.")
        if not path.startswith(_ALLOWED_PATH_PREFIXES):
            raise ValueError(
                "Atlassian path is outside the trusted Admin/User API surface."
            )
        return path

    def request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        trusted_path = self._trusted_path(path)
        headers = self.credentials.admin_headers()

        try:
            response = self.client.request(
                method=method,
                url=trusted_path,
                params=params,
                json=json_body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise AtlassianProviderError(
                code="transport_error",
                message=f"Atlassian transport failed during {operation}.",
            ) from exc

        if not 200 <= response.status_code < 300:
            raise error_from_status(
                operation=operation,
                status_code=response.status_code,
                retry_after=response.headers.get("Retry-After"),
            )

        if response.status_code == 204:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise AtlassianProviderError(
                code="invalid_response",
                message=f"Atlassian returned invalid JSON during {operation}.",
                status_code=response.status_code,
            ) from exc

    def get_json(
        self,
        path: str,
        *,
        operation: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return self.request_json(
            "GET",
            path,
            operation=operation,
            params=params,
        )

    def post_json(
        self,
        path: str,
        *,
        operation: str,
        json_body: Any,
    ) -> Any:
        return self.request_json(
            "POST",
            path,
            operation=operation,
            json_body=json_body,
        )


def _provider_status(exc: AtlassianProviderError) -> str:
    if exc.code == "not_found":
        return "not_found"
    if exc.code in {
        "credential_not_configured",
        "organization_not_configured",
        "authorization_failed",
        "bad_request",
    }:
        return "denied"
    if exc.code == "rate_limited":
        return "rate_limited"
    return "error"


class AtlassianAdminService:
    """J1 read-only organization/workspace/token-metadata provider."""

    def __init__(
        self,
        *,
        credentials: AtlassianCredentialBroker,
        client: AtlassianAdminClient,
    ) -> None:
        self.credentials = credentials
        self.client = client

    def credential_status(self) -> dict[str, Any]:
        return self.credentials.redacted_status()

    def list_organizations(self, *, limit: int = DEFAULT_PAGE_LIMIT) -> dict[str, Any]:
        try:
            limit = bounded_limit(limit)
            payload = self.client.get_json(
                "/admin/v1/orgs",
                operation="organization discovery",
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise AtlassianProviderError(
                    code="invalid_response",
                    message="Atlassian returned an invalid organization response.",
                )

            raw_items = payload["data"]
            organizations = []
            for item in raw_items[:limit]:
                if not isinstance(item, dict):
                    continue
                org_id = _optional_string(item.get("id"))
                if org_id is None:
                    continue
                attributes = item.get("attributes")
                if not isinstance(attributes, dict):
                    attributes = {}
                organizations.append(
                    {
                        "id": org_id,
                        "name": _optional_string(attributes.get("name")),
                        "type": _optional_string(item.get("type")),
                    }
                )

            next_cursor = response_next_cursor(payload)
            return {
                "ok": True,
                "status": "success",
                "organizations": organizations,
                "count": len(organizations),
                "truncated": len(raw_items) > limit or next_cursor is not None,
                "next_cursor_available": next_cursor is not None,
                "error": None,
            }
        except (AtlassianProviderError, ValueError) as exc:
            status = _provider_status(exc) if isinstance(exc, AtlassianProviderError) else "denied"
            return {
                "ok": False,
                "status": status,
                "organizations": [],
                "count": 0,
                "truncated": False,
                "next_cursor_available": False,
                "error": str(exc),
            }

    def get_organization(self, org_id: str | None = None) -> dict[str, Any]:
        try:
            resolved = self.credentials.resolve_org_id(org_id)
            payload = self.client.get_json(
                f"/admin/v1/orgs/{quote(resolved, safe='')}",
                operation="organization lookup",
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
                raise AtlassianProviderError(
                    code="invalid_response",
                    message="Atlassian returned an invalid organization response.",
                )
            item = payload["data"]
            returned_id = _optional_string(item.get("id"))
            if returned_id is None:
                raise AtlassianProviderError(
                    code="invalid_response",
                    message="Atlassian organization response did not contain an ID.",
                )
            attributes = item.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            return {
                "ok": True,
                "status": "success",
                "organization": {
                    "id": returned_id,
                    "name": _optional_string(attributes.get("name")),
                    "type": _optional_string(item.get("type")),
                },
                "error": None,
            }
        except AtlassianProviderError as exc:
            return {
                "ok": False,
                "status": _provider_status(exc),
                "organization": None,
                "error": str(exc),
            }

    def list_workspaces(
        self,
        *,
        org_id: str | None = None,
        name: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> dict[str, Any]:
        try:
            limit = bounded_limit(limit)
            resolved = self.credentials.resolve_org_id(org_id)
            body: dict[str, Any] = {"limit": limit}

            normalized_name = _optional_string(name)
            if name is not None and normalized_name is None:
                raise ValueError("workspace name must not be blank.")
            if normalized_name is not None:
                body["query"] = {
                    "field": {
                        "name": "attributes.name",
                        "values": [normalized_name],
                    }
                }

            payload = self.client.post_json(
                f"/admin/v2/orgs/{quote(resolved, safe='')}/workspaces",
                operation="workspace discovery",
                json_body=body,
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise AtlassianProviderError(
                    code="invalid_response",
                    message="Atlassian returned an invalid workspace response.",
                )

            raw_items = payload["data"]
            workspaces = []
            for item in raw_items[:limit]:
                if not isinstance(item, dict):
                    continue
                workspace_id = _optional_string(item.get("id"))
                if workspace_id is None:
                    continue
                attributes = item.get("attributes")
                if not isinstance(attributes, dict):
                    attributes = {}
                raw_regions = attributes.get("regions")
                regions = (
                    [value for value in raw_regions if _optional_string(value) is not None]
                    if isinstance(raw_regions, list)
                    else []
                )
                workspaces.append(
                    {
                        "id": workspace_id,
                        "name": _optional_string(attributes.get("name")),
                        "product_type": _optional_string(attributes.get("type")),
                        "type_key": _optional_string(attributes.get("typeKey")),
                        "status": _optional_string(attributes.get("status")),
                        "host_url": _optional_string(attributes.get("hostUrl")),
                        "realm": _optional_string(attributes.get("realm")),
                        "regions": regions,
                    }
                )

            next_cursor = response_next_cursor(payload)
            return {
                "ok": True,
                "status": "success",
                "organization_id": resolved,
                "workspaces": workspaces,
                "count": len(workspaces),
                "truncated": len(raw_items) > limit or next_cursor is not None,
                "next_cursor_available": next_cursor is not None,
                "error": None,
            }
        except (AtlassianProviderError, ValueError) as exc:
            status = _provider_status(exc) if isinstance(exc, AtlassianProviderError) else "denied"
            return {
                "ok": False,
                "status": status,
                "organization_id": None,
                "workspaces": [],
                "count": 0,
                "truncated": False,
                "next_cursor_available": False,
                "error": str(exc),
            }

    def list_api_token_metadata(
        self,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> dict[str, Any]:
        """
        List organization-wide user API-token metadata.

        Organization scope comes only from trusted
        ATLASSIAN_ORG_ID configuration.

        The caller cannot select an organization, account,
        URL, host, or raw provider path.

        Token secret values are never returned.
        """

        resolved_org: str | None = None

        try:
            limit = bounded_limit(
                limit
            )

            resolved_org = (
                self.credentials
                .resolve_org_id(
                    None
                )
            )

            payload = (
                self.client.get_json(
                    (
                        "/admin/api-access/v1/orgs/"
                        f"{quote(resolved_org, safe='')}"
                        "/api-tokens"
                    ),
                    operation=(
                        "organization API token "
                        "metadata lookup"
                    ),
                )
            )

            if (
                not isinstance(
                    payload,
                    dict,
                )
                or not isinstance(
                    payload.get(
                        "data"
                    ),
                    list,
                )
            ):
                raise AtlassianProviderError(
                    code="invalid_response",
                    message=(
                        "Atlassian returned an invalid "
                        "API token metadata response."
                    ),
                )

            raw_items = (
                payload[
                    "data"
                ]
            )

            tokens: list[
                dict[
                    str,
                    Any,
                ]
            ] = []

            for item in raw_items[
                :limit
            ]:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                token_id = (
                    _optional_string(
                        item.get(
                            "id"
                        )
                    )
                )

                if token_id is None:
                    continue

                user = (
                    item.get(
                        "user"
                    )
                )

                if not isinstance(
                    user,
                    dict,
                ):
                    user = {}

                raw_scopes = (
                    item.get(
                        "scopes"
                    )
                )

                scopes: list[str] = []

                if isinstance(
                    raw_scopes,
                    list,
                ):

                    for scope in raw_scopes:

                        normalized_scope = (
                            _optional_string(
                                scope
                            )
                        )

                        if (
                            normalized_scope
                            is not None
                        ):
                            scopes.append(
                                normalized_scope
                            )

                tokens.append(
                    {
                        "id":
                            token_id,

                        "label":
                            _optional_string(
                                item.get(
                                    "label"
                                )
                            ),

                        "status":
                            _optional_string(
                                item.get(
                                    "status"
                                )
                            ),

                        "created_at":
                            _optional_string(
                                item.get(
                                    "createdAt"
                                )
                            ),

                        "expires_at":
                            _optional_string(
                                item.get(
                                    "expiresAt"
                                )
                            ),

                        "last_active_at":
                            _optional_string(
                                item.get(
                                    "lastActiveAt"
                                )
                            ),

                        # Deliberately expose only the opaque
                        # account identifier from nested user
                        # metadata. Do not expose name/email here.
                        "user_id":
                            _optional_string(
                                user.get(
                                    "id"
                                )
                            ),

                        "scopes":
                            scopes,
                    }
                )

            next_cursor = (
                response_next_cursor(
                    payload
                )
            )

            return {
                "ok":
                    True,

                "status":
                    "success",

                "organization_id":
                    resolved_org,

                "tokens":
                    tokens,

                "count":
                    len(
                        tokens
                    ),

                "truncated":
                    (
                        len(
                            raw_items
                        )
                        > limit
                        or next_cursor
                        is not None
                    ),

                "next_cursor_available":
                    next_cursor
                    is not None,

                "secret_values_exposed":
                    False,

                "error":
                    None,
            }

        except (
            AtlassianProviderError,
            ValueError,
        ) as exc:

            status = (
                _provider_status(
                    exc
                )

                if isinstance(
                    exc,
                    AtlassianProviderError,
                )

                else "denied"
            )

            return {
                "ok":
                    False,

                "status":
                    status,

                "organization_id":
                    resolved_org,

                "tokens":
                    [],

                "count":
                    0,

                "truncated":
                    False,

                "next_cursor_available":
                    False,

                "secret_values_exposed":
                    False,

                "error":
                    str(
                        exc
                    ),
            }



def build_atlassian_admin_service(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> AtlassianAdminService:
    credentials = AtlassianCredentialBroker(
        admin_api_key=settings.atlassian_admin_api_key,
        default_org_id=settings.atlassian_org_id,
        jira_email=settings.jira_email,
        jira_api_token=settings.jira_api_token,
    )
    client = AtlassianAdminClient(
        credentials=credentials,
        timeout_seconds=settings.atlassian_admin_http_timeout_seconds,
        transport=transport,
    )
    return AtlassianAdminService(
        credentials=credentials,
        client=client,
    )
