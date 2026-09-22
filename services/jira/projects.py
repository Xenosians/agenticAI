from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Protocol,
)

from urllib.parse import (
    quote,
    urlsplit,
)

import httpx

from config import (
    Settings,
)


DEFAULT_PROJECT_LIMIT = 25
MAX_PROJECT_LIMIT = 50
MAX_PROJECT_QUERY_CHARS = 200
MAX_PROJECT_REFERENCE_CHARS = 100


@dataclass(
    frozen=True
)
class JiraProjectProviderError(
    RuntimeError
):
    code: str
    message: str
    status_code: int | None = None

    def __str__(
        self,
    ) -> str:
        return (
            self.message
        )


class JiraProjectReadProvider(
    Protocol
):
    def list_projects(
        self,
        *,
        query: str | None = None,
        limit: int = DEFAULT_PROJECT_LIMIT,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def get_project(
        self,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...


def _optional_string(
    value: Any,
) -> str | None:

    if not isinstance(
        value,
        str,
    ):
        return None

    normalized = (
        value.strip()
    )

    return (
        normalized
        if normalized
        else None
    )


def _provider_status(
    exc: JiraProjectProviderError,
) -> str:

    if exc.code in {
        "authorization_denied",
        "configuration_unavailable",
    }:
        return (
            "denied"
        )

    if exc.code == "not_found":
        return (
            "not_found"
        )

    return (
        "error"
    )


class JiraProjectReadService:
    """
    Trusted read-only Jira project provider.

    The model can choose only semantic project arguments.

    Trusted application code owns:
        - Jira origin
        - credentials
        - REST path
        - HTTP method
        - pagination bounds
        - provider response parsing

    No arbitrary Jira URL or HTTP method is model-facing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        timeout_seconds: float = 10.0,
        transport: (
            httpx.BaseTransport
            | None
        ) = None,
    ) -> None:

        self.base_url = (
            self._validate_base_url(
                base_url
            )
        )

        self.email = (
            self._require_non_empty(
                email,
                "Jira email",
            )
        )

        self.api_token = (
            self._require_non_empty(
                api_token,
                "Jira API token",
            )
        )

        if (
            not isinstance(
                timeout_seconds,
                (
                    int,
                    float,
                ),
            )
            or isinstance(
                timeout_seconds,
                bool,
            )
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "Jira timeout must be greater than zero."
            )

        self.client = (
            httpx.Client(
                base_url=(
                    self.base_url
                ),

                auth=(
                    self.email,
                    self.api_token,
                ),

                headers={
                    "Accept":
                        "application/json",
                },

                timeout=(
                    float(
                        timeout_seconds
                    )
                ),

                transport=(
                    transport
                ),
            )
        )

    @staticmethod
    def _require_non_empty(
        value: str,
        name: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                f"{name} must be a string."
            )

        normalized = (
            value.strip()
        )

        if not normalized:
            raise ValueError(
                f"{name} must not be empty."
            )

        return (
            normalized
        )

    @classmethod
    def _validate_base_url(
        cls,
        value: str,
    ) -> str:

        normalized = (
            cls._require_non_empty(
                value,
                "Jira base URL",
            )
            .rstrip("/")
        )

        parsed = (
            urlsplit(
                normalized
            )
        )

        if (
            parsed.scheme
            not in {
                "http",
                "https",
            }
        ):
            raise ValueError(
                "Jira base URL must use http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "Jira base URL must contain a hostname."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "Jira base URL must not contain credentials."
            )

        if (
            parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Jira base URL must not contain "
                "query or fragment components."
            )

        if parsed.path not in {
            "",
            "/",
        }:
            raise ValueError(
                "Jira base URL must be an origin "
                "without a path."
            )

        return (
            normalized
        )

    @staticmethod
    def _normalize_limit(
        limit: int,
    ) -> int:

        if (
            not isinstance(
                limit,
                int,
            )
            or isinstance(
                limit,
                bool,
            )
            or limit < 1
            or limit > MAX_PROJECT_LIMIT
        ):
            raise ValueError(
                "limit must be an integer between 1 and 50."
            )

        return (
            limit
        )

    @staticmethod
    def _normalize_query(
        query: str | None,
    ) -> str | None:

        if query is None:
            return None

        if not isinstance(
            query,
            str,
        ):
            raise ValueError(
                "query must be a string."
            )

        if (
            not query
            or query
            != query.strip()
        ):
            raise ValueError(
                "query must be a non-empty exact string "
                "without surrounding whitespace."
            )

        if (
            len(
                query
            )
            > MAX_PROJECT_QUERY_CHARS
        ):
            raise ValueError(
                "query is too long."
            )

        return (
            query
        )

    @staticmethod
    def _normalize_project_reference(
        project_id_or_key: str,
    ) -> str:

        if not isinstance(
            project_id_or_key,
            str,
        ):
            raise ValueError(
                "project_id_or_key must be a string."
            )

        if (
            not project_id_or_key
            or project_id_or_key
            != project_id_or_key.strip()
        ):
            raise ValueError(
                "project_id_or_key must be a non-empty exact "
                "string without surrounding whitespace."
            )

        if (
            len(
                project_id_or_key
            )
            > MAX_PROJECT_REFERENCE_CHARS
        ):
            raise ValueError(
                "project_id_or_key is too long."
            )

        return (
            project_id_or_key
        )

    def _get_json(
        self,
        path: str,
        *,
        params: (
            dict[
                str,
                Any,
            ]
            | None
        ) = None,
        operation: str,
    ) -> Any:

        try:
            response = (
                self.client.get(
                    path,
                    params=(
                        params
                    ),
                )
            )

        except httpx.TimeoutException as exc:
            raise JiraProjectProviderError(
                code="timeout",
                message=(
                    f"Jira timed out during {operation}."
                ),
            ) from exc

        except httpx.HTTPError as exc:
            raise JiraProjectProviderError(
                code="transport_error",
                message=(
                    f"Jira transport failed during {operation}."
                ),
            ) from exc

        if response.status_code == 401:
            raise JiraProjectProviderError(
                code="authentication_failed",
                message=(
                    "Jira authentication failed."
                ),
                status_code=401,
            )

        if response.status_code == 403:
            raise JiraProjectProviderError(
                code="authorization_denied",
                message=(
                    "Jira authorization denied this operation."
                ),
                status_code=403,
            )

        if response.status_code == 404:
            raise JiraProjectProviderError(
                code="not_found",
                message=(
                    "The requested Jira project was not found "
                    "or is not visible to this account."
                ),
                status_code=404,
            )

        if response.status_code == 429:
            raise JiraProjectProviderError(
                code="rate_limited",
                message=(
                    "Jira rate limited this operation."
                ),
                status_code=429,
            )

        if response.status_code >= 400:
            raise JiraProjectProviderError(
                code="provider_error",
                message=(
                    "Jira rejected the project operation."
                ),
                status_code=(
                    response.status_code
                ),
            )

        try:
            return (
                response.json()
            )

        except ValueError as exc:
            raise JiraProjectProviderError(
                code="invalid_response",
                message=(
                    "Jira returned an invalid JSON response."
                ),
                status_code=(
                    response.status_code
                ),
            ) from exc

    @classmethod
    def _decode_project(
        cls,
        payload: Any,
    ) -> dict[
        str,
        Any,
    ] | None:

        if not isinstance(
            payload,
            dict,
        ):
            return None

        project_id = (
            _optional_string(
                payload.get(
                    "id"
                )
            )
        )

        key = (
            _optional_string(
                payload.get(
                    "key"
                )
            )
        )

        name = (
            _optional_string(
                payload.get(
                    "name"
                )
            )
        )

        if (
            project_id is None
            or key is None
            or name is None
        ):
            return None

        category = (
            payload.get(
                "projectCategory"
            )
        )

        category_id = None
        category_name = None

        if isinstance(
            category,
            dict,
        ):
            category_id = (
                _optional_string(
                    category.get(
                        "id"
                    )
                )
            )

            category_name = (
                _optional_string(
                    category.get(
                        "name"
                    )
                )
            )

        simplified = (
            payload.get(
                "simplified"
            )
        )

        if not isinstance(
            simplified,
            bool,
        ):
            simplified = None

        archived = (
            payload.get(
                "archived"
            )
        )

        if not isinstance(
            archived,
            bool,
        ):
            archived = None

        return {
            "id":
                project_id,

            "key":
                key,

            "name":
                name,

            "project_type_key":
                _optional_string(
                    payload.get(
                        "projectTypeKey"
                    )
                ),

            "style":
                _optional_string(
                    payload.get(
                        "style"
                    )
                ),

            "simplified":
                simplified,

            "archived":
                archived,

            "category_id":
                category_id,

            "category_name":
                category_name,
        }

    def list_projects(
        self,
        *,
        query: str | None = None,
        limit: int = DEFAULT_PROJECT_LIMIT,
    ) -> dict[
        str,
        Any,
    ]:

        try:
            resolved_limit = (
                self._normalize_limit(
                    limit
                )
            )

            resolved_query = (
                self._normalize_query(
                    query
                )
            )

            params: dict[
                str,
                Any,
            ] = {
                "startAt":
                    0,

                "maxResults":
                    resolved_limit,
            }

            if (
                resolved_query
                is not None
            ):
                params[
                    "query"
                ] = (
                    resolved_query
                )

            payload = (
                self._get_json(
                    "/rest/api/3/project/search",
                    params=(
                        params
                    ),
                    operation=(
                        "project discovery"
                    ),
                )
            )

            if not isinstance(
                payload,
                dict,
            ):
                raise JiraProjectProviderError(
                    code="invalid_response",
                    message=(
                        "Jira returned an invalid "
                        "project-search response."
                    ),
                )

            raw_values = (
                payload.get(
                    "values"
                )
            )

            if not isinstance(
                raw_values,
                list,
            ):
                raise JiraProjectProviderError(
                    code="invalid_response",
                    message=(
                        "Jira project search did not return "
                        "a valid values collection."
                    ),
                )

            projects = []

            for item in (
                raw_values
            ):
                decoded = (
                    self._decode_project(
                        item
                    )
                )

                if (
                    decoded
                    is not None
                ):
                    projects.append(
                        decoded
                    )

            total = (
                payload.get(
                    "total"
                )
            )

            if (
                not isinstance(
                    total,
                    int,
                )
                or isinstance(
                    total,
                    bool,
                )
            ):
                total = None

            is_last = (
                payload.get(
                    "isLast"
                )
            )

            if not isinstance(
                is_last,
                bool,
            ):
                is_last = None

            truncated = (
                is_last is False
                or (
                    total is not None
                    and total
                    > len(
                        projects
                    )
                )
            )

            return {
                "ok":
                    True,

                "status":
                    "success",

                "query":
                    resolved_query,

                "projects":
                    projects,

                "count":
                    len(
                        projects
                    ),

                "total":
                    total,

                "truncated":
                    truncated,

                "error":
                    None,
            }

        except (
            JiraProjectProviderError,
            ValueError,
        ) as exc:

            status = (
                _provider_status(
                    exc
                )

                if isinstance(
                    exc,
                    JiraProjectProviderError,
                )

                else "denied"
            )

            return {
                "ok":
                    False,

                "status":
                    status,

                "query":
                    (
                        query
                        if isinstance(
                            query,
                            str,
                        )
                        else None
                    ),

                "projects":
                    [],

                "count":
                    0,

                "total":
                    None,

                "truncated":
                    False,

                "error":
                    str(
                        exc
                    ),
            }

    def get_project(
        self,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:

        resolved_reference = None

        try:
            resolved_reference = (
                self
                ._normalize_project_reference(
                    project_id_or_key
                )
            )

            payload = (
                self._get_json(
                    (
                        "/rest/api/3/project/"
                        + quote(
                            resolved_reference,
                            safe="",
                        )
                    ),
                    operation=(
                        "project lookup"
                    ),
                )
            )

            project = (
                self._decode_project(
                    payload
                )
            )

            if project is None:
                raise JiraProjectProviderError(
                    code="invalid_response",
                    message=(
                        "Jira returned invalid project metadata."
                    ),
                )

            return {
                "ok":
                    True,

                "status":
                    "success",

                "project_id_or_key":
                    resolved_reference,

                "project":
                    project,

                "error":
                    None,
            }

        except (
            JiraProjectProviderError,
            ValueError,
        ) as exc:

            status = (
                _provider_status(
                    exc
                )

                if isinstance(
                    exc,
                    JiraProjectProviderError,
                )

                else "denied"
            )

            return {
                "ok":
                    False,

                "status":
                    status,

                "project_id_or_key":
                    (
                        resolved_reference
                        if resolved_reference
                        is not None
                        else (
                            project_id_or_key
                            if isinstance(
                                project_id_or_key,
                                str,
                            )
                            else None
                        )
                    ),

                "project":
                    None,

                "error":
                    str(
                        exc
                    ),
            }


class UnavailableJiraProjectReadService:
    """
    Fail-closed provider used when Jira credentials are not configured.

    Keeping the semantic capability registered avoids registry/MCP
    mismatch while execution still fails safely.
    """

    ERROR = (
        "Jira project access is not configured."
    )

    def list_projects(
        self,
        *,
        query: str | None = None,
        limit: int = DEFAULT_PROJECT_LIMIT,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "query":
                query,

            "projects":
                [],

            "count":
                0,

            "total":
                None,

            "truncated":
                False,

            "error":
                self.ERROR,
        }

    def get_project(
        self,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "project_id_or_key":
                (
                    project_id_or_key
                    if isinstance(
                        project_id_or_key,
                        str,
                    )
                    else None
                ),

            "project":
                None,

            "error":
                self.ERROR,
        }


def build_jira_project_read_service(
    settings: Settings,
) -> JiraProjectReadProvider:

    if not (
        settings.jira_base_url
        and settings.jira_email
        and settings.jira_api_token
    ):
        return (
            UnavailableJiraProjectReadService()
        )

    return (
        JiraProjectReadService(
            base_url=(
                settings
                .require_jira_base_url()
            ),

            email=(
                settings
                .require_jira_email()
            ),

            api_token=(
                settings
                .require_jira_api_token()
            ),

            timeout_seconds=(
                settings
                .jira_http_timeout_seconds
            ),
        )
    )
