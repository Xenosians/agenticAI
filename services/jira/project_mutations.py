from __future__ import annotations

import json

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


# ============================================================
# TRUSTED TEMPLATE ALIASES
#
# Model-facing values are deliberately small semantic aliases.
#
# Atlassian-native project type/template keys remain trusted
# provider implementation details.
# ============================================================

PROJECT_TEMPLATE_PROFILES = {
    "software-kanban": {
        "project_type_key":
            "software",

        "project_template_key":
            (
                "com.pyxis.greenhopper.jira:"
                "gh-simplified-agility-kanban"
            ),
    },

    "software-scrum": {
        "project_type_key":
            "software",

        "project_template_key":
            (
                "com.pyxis.greenhopper.jira:"
                "gh-simplified-agility-scrum"
            ),
    },

    "business-task-tracking": {
        "project_type_key":
            "business",

        "project_template_key":
            (
                "com.atlassian.jira-core-project-templates:"
                "jira-core-simplified-task-tracking"
            ),
    },

    "service-it-management": {
        "project_type_key":
            "service_desk",

        "project_template_key":
            (
                "com.atlassian.servicedesk:"
                "simplified-it-service-management"
            ),
    },
}


MAX_PROJECT_KEY_CHARS = 64
MAX_PROJECT_NAME_CHARS = 255


class JiraProjectMutationProvider(
    Protocol
):
    def prepare_create_project(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def prepare_update_project(
        self,
        *,
        project_id_or_key: str,
        new_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def update_project(
        self,
        *,
        project_id_or_key: str,
        new_name: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def prepare_archive_project(
        self,
        *,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def archive_project(
        self,
        *,
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def prepare_delete_project(
        self,
        *,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def delete_project(
        self,
        *,
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...

    def create_project(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
        expected_project_type_key: str,
        expected_project_template_key: str,
        expected_lead_account_id: str,
    ) -> dict[
        str,
        Any,
    ]:
        ...


def jira_project_template_argument_values(
) -> dict[
    str,
    list[str],
]:

    return {
        "template":
            list(
                PROJECT_TEMPLATE_PROFILES
                .keys()
            ),
    }


def _exact_string(
    value: Any,
    *,
    field_name: str,
    max_length: int,
) -> str:

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"{field_name} must be a string."
        )

    if (
        not value
        or value
        != value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty exact "
            "string without surrounding whitespace."
        )

    if len(
        value
    ) > max_length:
        raise ValueError(
            f"{field_name} is too long."
        )

    if any(
        ord(character) < 32
        for character
        in value
    ):
        raise ValueError(
            f"{field_name} must not contain control characters."
        )

    return (
        value
    )


class JiraProjectMutationService:
    """
    Trusted governed Jira project mutation adapter.

    Model/user controlled:
        project_key
        project_name
        bounded template alias

    Trusted policy controlled:
        projectTypeKey
        projectTemplateKey
        leadAccountId
        Jira origin
        HTTP method/path
        provider validation
        read-back verification

    Creation is never executed without the ToolGateway approval
    boundary.
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
            _exact_string(
                email,
                field_name=(
                    "Jira email"
                ),
                max_length=320,
            )
        )

        self.api_token = (
            _exact_string(
                api_token,
                field_name=(
                    "Jira API token"
                ),
                max_length=4096,
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

                    "Content-Type":
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
    def _validate_base_url(
        value: str,
    ) -> str:

        normalized = (
            _exact_string(
                value.rstrip("/"),
                field_name=(
                    "Jira base URL"
                ),
                max_length=2048,
            )
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
    def _normalize_inputs(
        *,
        project_key: str,
        project_name: str,
        template: str,
    ) -> tuple[
        str,
        str,
        str,
        dict[
            str,
            str,
        ],
    ]:

        resolved_key = (
            _exact_string(
                project_key,
                field_name=(
                    "project_key"
                ),
                max_length=(
                    MAX_PROJECT_KEY_CHARS
                ),
            )
        )

        resolved_name = (
            _exact_string(
                project_name,
                field_name=(
                    "project_name"
                ),
                max_length=(
                    MAX_PROJECT_NAME_CHARS
                ),
            )
        )

        resolved_template = (
            _exact_string(
                template,
                field_name=(
                    "template"
                ),
                max_length=100,
            )
        )

        profile = (
            PROJECT_TEMPLATE_PROFILES
            .get(
                resolved_template
            )
        )

        if profile is None:
            raise ValueError(
                "template must be one of the trusted "
                "configured Jira project template aliases."
            )

        return (
            resolved_key,
            resolved_name,
            resolved_template,
            profile,
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
    ) -> Any:

        response = (
            self.client.get(
                path,
                params=(
                    params
                ),
            )
        )

        if response.status_code == 401:
            raise PermissionError(
                "Jira authentication failed."
            )

        if response.status_code == 403:
            raise PermissionError(
                "Jira authorization denied this operation."
            )

        if response.status_code == 404:
            raise LookupError(
                "The requested Jira resource was not found "
                "or is not accessible."
            )

        if response.status_code == 429:
            raise RuntimeError(
                "Jira rate limited this operation."
            )

        if response.status_code >= 400:
            raise RuntimeError(
                "Jira rejected the project operation."
            )

        try:
            return (
                response.json()
            )

        except ValueError as exc:
            raise RuntimeError(
                "Jira returned invalid JSON."
            ) from exc

    def _current_account_id(
        self,
    ) -> str:

        payload = (
            self._get_json(
                "/rest/api/3/myself"
            )
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                "Jira returned invalid current-user metadata."
            )

        account_id = (
            payload.get(
                "accountId"
            )
        )

        return (
            _exact_string(
                account_id,
                field_name=(
                    "trusted Jira account ID"
                ),
                max_length=256,
            )
        )

    def _validate_project_type_accessible(
        self,
        project_type_key: str,
    ) -> None:

        self._get_json(
            (
                "/rest/api/3/project/type/"
                + quote(
                    project_type_key,
                    safe="",
                )
                + "/accessible"
            )
        )

    def _get_validation_scalar(
        self,
        path: str,
        *,
        params: dict[
            str,
            Any,
        ],
        field_name: str,
        max_length: int,
    ) -> str:
        """
        Read one Jira project-validation scalar.

        Jira Cloud currently returns the project key/name
        validation values as bare response text even while
        declaring application/json.

        This compatibility path is intentionally scoped ONLY
        to those validation endpoints.

        General Jira provider responses remain strict JSON.
        """

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
            raise RuntimeError(
                "Jira timed out during project validation."
            ) from exc

        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Jira transport failed during project validation."
            ) from exc

        if response.status_code == 401:
            raise PermissionError(
                "Jira authentication failed."
            )

        if response.status_code == 403:
            raise PermissionError(
                "Jira authorization denied this operation."
            )

        if response.status_code == 404:
            raise LookupError(
                "The Jira project-validation endpoint "
                "was not found or is not accessible."
            )

        if response.status_code == 429:
            raise RuntimeError(
                "Jira rate limited project validation."
            )

        if response.status_code >= 400:
            raise RuntimeError(
                "Jira rejected project validation."
            )

        # ----------------------------------------------------
        # Accept either representation:
        #
        #   JSON string:
        #       "PROJECT"
        #
        #   Jira Cloud live behavior:
        #       PROJECT
        #
        # Do NOT permit arbitrary JSON objects/lists here.
        # ----------------------------------------------------

        try:
            parsed = (
                response.json()
            )

        except ValueError:
            parsed = None

        if isinstance(
            parsed,
            str,
        ):
            scalar = (
                parsed
            )

        elif parsed is None:
            scalar = (
                response.text
            )

        else:
            raise RuntimeError(
                "Jira returned invalid project-validation data."
            )

        return (
            _exact_string(
                scalar,
                field_name=(
                    field_name
                ),
                max_length=(
                    max_length
                ),
            )
        )

    def _validate_key_exact(
        self,
        project_key: str,
    ) -> None:

        payload = (
            self._get_validation_scalar(
                (
                    "/rest/api/3/projectvalidate/"
                    "validProjectKey"
                ),
                params={
                    "key":
                        project_key,
                },
                field_name=(
                    "Jira project-key validation response"
                ),
                max_length=(
                    MAX_PROJECT_KEY_CHARS
                ),
            )
        )

        if (
            payload
            != project_key
        ):
            raise ValueError(
                "The requested Jira project key is invalid "
                "or already in use. Jira suggested a different "
                "key, but trusted policy will not rewrite the "
                "user's requested key."
            )

    def _validate_name_exact(
        self,
        project_name: str,
    ) -> None:

        payload = (
            self._get_validation_scalar(
                (
                    "/rest/api/3/projectvalidate/"
                    "validProjectName"
                ),
                params={
                    "name":
                        project_name,
                },
                field_name=(
                    "Jira project-name validation response"
                ),
                max_length=(
                    MAX_PROJECT_NAME_CHARS
                ),
            )
        )

        if (
            payload
            != project_name
        ):
            raise ValueError(
                "The requested Jira project name is already "
                "in use or requires provider-side rewriting. "
                "Trusted policy will not silently rename it."
            )

    def _lookup_project(
        self,
        project_key: str,
    ) -> tuple[
        str,
        dict[
            str,
            Any,
        ]
        | None,
    ]:

        try:
            response = (
                self.client.get(
                    (
                        "/rest/api/3/project/"
                        + quote(
                            project_key,
                            safe="",
                        )
                    )
                )
            )

        except httpx.HTTPError:
            return (
                "unknown",
                None,
            )

        if response.status_code == 404:
            return (
                "missing",
                None,
            )

        if response.status_code != 200:
            return (
                "unknown",
                None,
            )

        try:
            payload = (
                response.json()
            )

        except ValueError:
            return (
                "unknown",
                None,
            )

        if not isinstance(
            payload,
            dict,
        ):
            return (
                "unknown",
                None,
            )

        return (
            "found",
            payload,
        )

    @staticmethod
    def _verified_project(
        payload: dict[
            str,
            Any,
        ],
        *,
        project_key: str,
        project_name: str,
        project_type_key: str,
    ) -> bool:

        return (
            payload.get(
                "key"
            )
            == project_key

            and payload.get(
                "name"
            )
            == project_name

            and payload.get(
                "projectTypeKey"
            )
            == project_type_key
        )

    def prepare_create_project(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
    ) -> dict[
        str,
        Any,
    ]:

        try:
            (
                resolved_key,
                resolved_name,
                resolved_template,
                profile,
            ) = (
                self._normalize_inputs(
                    project_key=(
                        project_key
                    ),

                    project_name=(
                        project_name
                    ),

                    template=(
                        template
                    ),
                )
            )

            project_type_key = (
                profile[
                    "project_type_key"
                ]
            )

            project_template_key = (
                profile[
                    "project_template_key"
                ]
            )

            # ----------------------------------------------------
            # Trusted provider preconditions.
            #
            # Do not ask the model to synthesize these reads.
            # ----------------------------------------------------

            self._validate_project_type_accessible(
                project_type_key
            )

            self._validate_key_exact(
                resolved_key
            )

            self._validate_name_exact(
                resolved_name
            )

            lead_account_id = (
                self._current_account_id()
            )

            return {
                "ok":
                    True,

                "status":
                    "ready",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "execution_arguments": {
                    "project_key":
                        resolved_key,

                    "project_name":
                        resolved_name,

                    "template":
                        resolved_template,

                    "expected_project_type_key":
                        project_type_key,

                    "expected_project_template_key":
                        project_template_key,

                    "expected_lead_account_id":
                        lead_account_id,
                },

                "error":
                    None,
            }

        except PermissionError as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            ValueError,
            LookupError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "error",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

    @staticmethod
    def _normalize_project_reference(
        project_id_or_key: str,
    ) -> str:

        return (
            _exact_string(
                project_id_or_key,
                field_name=(
                    "project_id_or_key"
                ),
                max_length=100,
            )
        )

    @staticmethod
    def _project_snapshot(
        payload: Any,
    ) -> dict[
        str,
        str,
    ]:

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                "Jira returned invalid project metadata."
            )

        raw_id = (
            payload.get(
                "id"
            )
        )

        if isinstance(
            raw_id,
            int,
        ) and not isinstance(
            raw_id,
            bool,
        ):
            project_id = (
                str(
                    raw_id
                )
            )

        elif isinstance(
            raw_id,
            str,
        ):
            project_id = (
                raw_id
            )

        else:
            raise RuntimeError(
                "Jira project metadata did not contain "
                "a valid project ID."
            )

        project_id = (
            _exact_string(
                project_id,
                field_name=(
                    "trusted Jira project ID"
                ),
                max_length=100,
            )
        )

        project_key = (
            _exact_string(
                payload.get(
                    "key"
                ),
                field_name=(
                    "trusted Jira project key"
                ),
                max_length=(
                    MAX_PROJECT_KEY_CHARS
                ),
            )
        )

        project_name = (
            _exact_string(
                payload.get(
                    "name"
                ),
                field_name=(
                    "trusted Jira project name"
                ),
                max_length=(
                    MAX_PROJECT_NAME_CHARS
                ),
            )
        )

        return {
            "id":
                project_id,

            "key":
                project_key,

            "name":
                project_name,
        }

    def _get_project_snapshot(
        self,
        project_id_or_key: str,
    ) -> dict[
        str,
        str,
    ]:

        reference = (
            self._normalize_project_reference(
                project_id_or_key
            )
        )

        payload = (
            self._get_json(
                (
                    "/rest/api/3/project/"
                    + quote(
                        reference,
                        safe="",
                    )
                )
            )
        )

        return (
            self._project_snapshot(
                payload
            )
        )

    @staticmethod
    def _snapshot_matches(
        snapshot: dict[
            str,
            str,
        ],
        *,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> bool:

        return (
            snapshot.get(
                "id"
            )
            == expected_project_id

            and snapshot.get(
                "key"
            )
            == expected_project_key

            and snapshot.get(
                "name"
            )
            == expected_project_name
        )

    def prepare_update_project(
        self,
        *,
        project_id_or_key: str,
        new_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Read-only policy preparation for a project rename.

        The user/model controls only:
            project_id_or_key
            new_name

        Trusted policy resolves and freezes:
            immutable project ID
            current project key
            current project name
        """

        try:
            resolved_reference = (
                self._normalize_project_reference(
                    project_id_or_key
                )
            )

            resolved_new_name = (
                _exact_string(
                    new_name,
                    field_name=(
                        "new_name"
                    ),
                    max_length=(
                        MAX_PROJECT_NAME_CHARS
                    ),
                )
            )

            current = (
                self._get_project_snapshot(
                    resolved_reference
                )
            )

            if (
                current[
                    "name"
                ]
                == resolved_new_name
            ):
                raise ValueError(
                    "The Jira project already has the "
                    "requested name."
                )

            # ------------------------------------------------
            # Exact provider validation.
            #
            # If Jira suggests another name, reject rather
            # than silently rewriting the user's request.
            # ------------------------------------------------

            self._validate_name_exact(
                resolved_new_name
            )

            return {
                "ok":
                    True,

                "status":
                    "ready",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "execution_arguments": {
                    "project_id_or_key":
                        resolved_reference,

                    "new_name":
                        resolved_new_name,

                    "expected_project_id":
                        current[
                            "id"
                        ],

                    "expected_project_key":
                        current[
                            "key"
                        ],

                    "expected_project_name":
                        current[
                            "name"
                        ],
                },

                "error":
                    None,
            }

        except PermissionError as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            ValueError,
            LookupError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "error",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

    @staticmethod
    def _update_result(
        *,
        ok: bool,
        status: str,
        project_id: str,
        project_key: str,
        previous_project_name: str,
        new_project_name: str,
        mutation_performed: (
            bool
            | None
        ),
        verification_ok: bool,
        reconciled: bool,
        error: (
            str
            | None
        ),
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                ok,

            "status":
                status,

            "operation":
                "update",

            "project_id":
                project_id,

            "project_key":
                project_key,

            "project_name":
                (
                    new_project_name
                    if ok
                    else previous_project_name
                ),

            "previous_project_name":
                previous_project_name,

            "new_project_name":
                new_project_name,

            "template":
                None,

            "project_type_key":
                None,

            "mutation_performed":
                mutation_performed,

            "verification_ok":
                verification_ok,

            "reconciled":
                reconciled,

            "error":
                error,
        }

    def _reconcile_uncertain_update(
        self,
        *,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
        new_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        try:
            current = (
                self._get_project_snapshot(
                    expected_project_id
                )
            )

        except Exception:

            return (
                self._update_result(
                    ok=False,
                    status=(
                        "outcome_unknown"
                    ),
                    project_id=(
                        expected_project_id
                    ),
                    project_key=(
                        expected_project_key
                    ),
                    previous_project_name=(
                        expected_project_name
                    ),
                    new_project_name=(
                        new_name
                    ),
                    mutation_performed=None,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira project update returned an "
                        "ambiguous provider outcome and "
                        "trusted read-back could not determine "
                        "the final project state."
                    ),
                )
            )

        # ----------------------------------------------------
        # Exact requested state exists.
        # ----------------------------------------------------

        if (
            current.get(
                "id"
            )
            == expected_project_id

            and current.get(
                "key"
            )
            == expected_project_key

            and current.get(
                "name"
            )
            == new_name
        ):

            return (
                self._update_result(
                    ok=True,
                    status="success",
                    project_id=(
                        expected_project_id
                    ),
                    project_key=(
                        expected_project_key
                    ),
                    previous_project_name=(
                        expected_project_name
                    ),
                    new_project_name=(
                        new_name
                    ),
                    mutation_performed=True,
                    verification_ok=True,
                    reconciled=True,
                    error=None,
                )
            )

        # ----------------------------------------------------
        # Exact old state still exists.
        #
        # We can prove the requested mutation is not visible.
        # Do not retry automatically.
        # ----------------------------------------------------

        if (
            current.get(
                "id"
            )
            == expected_project_id

            and current.get(
                "key"
            )
            == expected_project_key

            and current.get(
                "name"
            )
            == expected_project_name
        ):

            return (
                self._update_result(
                    ok=False,
                    status="error",
                    project_id=(
                        expected_project_id
                    ),
                    project_key=(
                        expected_project_key
                    ),
                    previous_project_name=(
                        expected_project_name
                    ),
                    new_project_name=(
                        new_name
                    ),
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=True,
                    error=(
                        "Jira project update was not verified "
                        "as applied. Trusted read-back still "
                        "shows the exact pre-update state."
                    ),
                )
            )

        # ----------------------------------------------------
        # Some third state exists.
        # ----------------------------------------------------

        return (
            self._update_result(
                ok=False,
                status=(
                    "outcome_unknown"
                ),
                project_id=(
                    expected_project_id
                ),
                project_key=(
                    expected_project_key
                ),
                previous_project_name=(
                    expected_project_name
                ),
                new_project_name=(
                    new_name
                ),
                mutation_performed=None,
                verification_ok=False,
                reconciled=False,
                error=(
                    "Jira project state changed, but it does "
                    "not match either the approved old state "
                    "or the requested new state."
                ),
            )
        )

    def update_project(
        self,
        *,
        project_id_or_key: str,
        new_name: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Execute one approved exact project rename.

        The provider-native path uses the trusted immutable
        project ID captured before approval rather than trusting
        the model-facing reference at execution time.
        """

        resolved_project_id = None
        resolved_project_key = None
        resolved_old_name = None
        resolved_new_name = None

        try:
            resolved_reference = (
                self._normalize_project_reference(
                    project_id_or_key
                )
            )

            resolved_new_name = (
                _exact_string(
                    new_name,
                    field_name=(
                        "new_name"
                    ),
                    max_length=(
                        MAX_PROJECT_NAME_CHARS
                    ),
                )
            )

            resolved_project_id = (
                _exact_string(
                    expected_project_id,
                    field_name=(
                        "expected_project_id"
                    ),
                    max_length=100,
                )
            )

            resolved_project_key = (
                _exact_string(
                    expected_project_key,
                    field_name=(
                        "expected_project_key"
                    ),
                    max_length=(
                        MAX_PROJECT_KEY_CHARS
                    ),
                )
            )

            resolved_old_name = (
                _exact_string(
                    expected_project_name,
                    field_name=(
                        "expected_project_name"
                    ),
                    max_length=(
                        MAX_PROJECT_NAME_CHARS
                    ),
                )
            )

            # Keep the original persisted semantic reference
            # syntactically valid, even though execution uses
            # the trusted project ID.
            if not resolved_reference:
                raise ValueError(
                    "project_id_or_key is invalid."
                )

            # =================================================
            # POST-APPROVAL SNAPSHOT REVALIDATION
            # =================================================

            current = (
                self._get_project_snapshot(
                    resolved_project_id
                )
            )

            if not (
                self._snapshot_matches(
                    current,
                    expected_project_id=(
                        resolved_project_id
                    ),
                    expected_project_key=(
                        resolved_project_key
                    ),
                    expected_project_name=(
                        resolved_old_name
                    ),
                )
            ):
                raise ValueError(
                    "Jira project state changed after approval. "
                    "The approved rename snapshot is stale."
                )

            # Name must still be exactly available immediately
            # before the mutation.
            self._validate_name_exact(
                resolved_new_name
            )

            body = {
                "name":
                    resolved_new_name,
            }

            try:
                response = (
                    self.client.put(
                        (
                            "/rest/api/3/project/"
                            + quote(
                                resolved_project_id,
                                safe="",
                            )
                        ),
                        content=(
                            json.dumps(
                                body
                            )
                        ),
                    )
                )

            except (
                httpx.TimeoutException,
                httpx.TransportError,
            ):

                return (
                    self._reconcile_uncertain_update(
                        expected_project_id=(
                            resolved_project_id
                        ),
                        expected_project_key=(
                            resolved_project_key
                        ),
                        expected_project_name=(
                            resolved_old_name
                        ),
                        new_name=(
                            resolved_new_name
                        ),
                    )
                )

            if response.status_code in {
                400,
                401,
                403,
                404,
                409,
            }:

                return (
                    self._update_result(
                        ok=False,
                        status="denied",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        previous_project_name=(
                            resolved_old_name
                        ),
                        new_project_name=(
                            resolved_new_name
                        ),
                        mutation_performed=False,
                        verification_ok=False,
                        reconciled=False,
                        error=(
                            "Jira rejected the approved project "
                            "rename. No successful provider "
                            "mutation was confirmed."
                        ),
                    )
                )

            if response.status_code >= 500:

                return (
                    self._reconcile_uncertain_update(
                        expected_project_id=(
                            resolved_project_id
                        ),
                        expected_project_key=(
                            resolved_project_key
                        ),
                        expected_project_name=(
                            resolved_old_name
                        ),
                        new_name=(
                            resolved_new_name
                        ),
                    )
                )

            if response.status_code != 200:

                return (
                    self._update_result(
                        ok=False,
                        status=(
                            "outcome_unknown"
                        ),
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        previous_project_name=(
                            resolved_old_name
                        ),
                        new_project_name=(
                            resolved_new_name
                        ),
                        mutation_performed=None,
                        verification_ok=False,
                        reconciled=False,
                        error=(
                            "Jira returned an unexpected "
                            "response to project update."
                        ),
                    )
                )

            # =================================================
            # PROVIDER READ-BACK IS AUTHORITATIVE
            # =================================================

            verified = (
                self._get_project_snapshot(
                    resolved_project_id
                )
            )

            if (
                verified.get(
                    "id"
                )
                == resolved_project_id

                and verified.get(
                    "key"
                )
                == resolved_project_key

                and verified.get(
                    "name"
                )
                == resolved_new_name
            ):

                return (
                    self._update_result(
                        ok=True,
                        status="success",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        previous_project_name=(
                            resolved_old_name
                        ),
                        new_project_name=(
                            resolved_new_name
                        ),
                        mutation_performed=True,
                        verification_ok=True,
                        reconciled=False,
                        error=None,
                    )
                )

            # 200 was received, but exact final state was not
            # proven. Never call this completed.
            return (
                self._update_result(
                    ok=False,
                    status=(
                        "outcome_unknown"
                    ),
                    project_id=(
                        resolved_project_id
                    ),
                    project_key=(
                        resolved_project_key
                    ),
                    previous_project_name=(
                        resolved_old_name
                    ),
                    new_project_name=(
                        resolved_new_name
                    ),
                    mutation_performed=None,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira accepted the project update, "
                        "but trusted read-back could not verify "
                        "the exact approved final state."
                    ),
                )
            )

        except PermissionError as exc:
            status = (
                "denied"
            )
            error = (
                str(
                    exc
                )
            )

        except (
            ValueError,
            LookupError,
        ) as exc:
            status = (
                "denied"
            )
            error = (
                str(
                    exc
                )
            )

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:
            status = (
                "error"
            )
            error = (
                str(
                    exc
                )
            )

        return (
            self._update_result(
                ok=False,
                status=(
                    status
                ),
                project_id=(
                    resolved_project_id
                    or (
                        expected_project_id
                        if isinstance(
                            expected_project_id,
                            str,
                        )
                        else ""
                    )
                ),
                project_key=(
                    resolved_project_key
                    or (
                        expected_project_key
                        if isinstance(
                            expected_project_key,
                            str,
                        )
                        else ""
                    )
                ),
                previous_project_name=(
                    resolved_old_name
                    or (
                        expected_project_name
                        if isinstance(
                            expected_project_name,
                            str,
                        )
                        else ""
                    )
                ),
                new_project_name=(
                    resolved_new_name
                    or (
                        new_name
                        if isinstance(
                            new_name,
                            str,
                        )
                        else ""
                    )
                ),
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=(
                    error
                ),
            )
        )

    def _project_in_status(
        self,
        *,
        project_id: str,
        project_key: str,
        project_name: str,
        status: str,
    ) -> bool:
        """
        Verify one exact project against Jira's project-search
        lifecycle status filter.

        The lifecycle status is trusted provider state.

        Supported values here are intentionally fixed rather
        than model-controlled.
        """

        if status not in {
            "live",
            "archived",
        }:
            raise ValueError(
                "Unsupported trusted Jira project status."
            )

        if (
            not isinstance(
                project_id,
                str,
            )
            or not project_id.isdigit()
        ):
            raise ValueError(
                "Trusted Jira project ID must be numeric."
            )

        payload = (
            self._get_json(
                "/rest/api/3/project/search",
                params={
                    "id": [
                        int(
                            project_id
                        ),
                    ],

                    "status": [
                        status,
                    ],

                    "maxResults":
                        2,
                },
            )
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                "Jira returned invalid project-search metadata."
            )

        values = (
            payload.get(
                "values"
            )
        )

        if not isinstance(
            values,
            list,
        ):
            raise RuntimeError(
                "Jira project search did not return "
                "a valid values collection."
            )

        matches = []

        for item in values:

            try:
                snapshot = (
                    self._project_snapshot(
                        item
                    )
                )

            except (
                RuntimeError,
                ValueError,
            ):
                continue

            if (
                snapshot[
                    "id"
                ]
                == project_id

                and snapshot[
                    "key"
                ]
                == project_key

                and snapshot[
                    "name"
                ]
                == project_name
            ):
                matches.append(
                    snapshot
                )

        if len(
            matches
        ) > 1:
            raise RuntimeError(
                "Jira returned duplicate exact project "
                "records for one trusted project ID."
            )

        return (
            len(
                matches
            )
            == 1
        )

    def _resolve_archive_state(
        self,
        *,
        project_id: str,
        project_key: str,
        project_name: str,
    ) -> str:
        """
        Return exactly one trusted lifecycle classification:

            archived
            live
            unknown

        Both/neither matching fails closed as unknown.
        """

        archived = (
            self._project_in_status(
                project_id=(
                    project_id
                ),

                project_key=(
                    project_key
                ),

                project_name=(
                    project_name
                ),

                status="archived",
            )
        )

        live = (
            self._project_in_status(
                project_id=(
                    project_id
                ),

                project_key=(
                    project_key
                ),

                project_name=(
                    project_name
                ),

                status="live",
            )
        )

        if (
            archived
            and not live
        ):
            return (
                "archived"
            )

        if (
            live
            and not archived
        ):
            return (
                "live"
            )

        return (
            "unknown"
        )

    def prepare_archive_project(
        self,
        *,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Read-only HIGH-risk archive preparation.

        The original model/user project reference is preserved.

        Trusted code binds:
            immutable project ID
            current project key
            current project name
            current live lifecycle state
        """

        try:
            resolved_reference = (
                self._normalize_project_reference(
                    project_id_or_key
                )
            )

            current = (
                self._get_project_snapshot(
                    resolved_reference
                )
            )

            state = (
                self._resolve_archive_state(
                    project_id=(
                        current[
                            "id"
                        ]
                    ),

                    project_key=(
                        current[
                            "key"
                        ]
                    ),

                    project_name=(
                        current[
                            "name"
                        ]
                    ),
                )
            )

            if state == "archived":
                raise ValueError(
                    "The Jira project is already archived."
                )

            if state != "live":
                raise RuntimeError(
                    "Trusted policy could not prove that "
                    "the Jira project is currently live."
                )

            return {
                "ok":
                    True,

                "status":
                    "ready",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "execution_arguments": {
                    "project_id_or_key":
                        resolved_reference,

                    "expected_project_id":
                        current[
                            "id"
                        ],

                    "expected_project_key":
                        current[
                            "key"
                        ],

                    "expected_project_name":
                        current[
                            "name"
                        ],
                },

                "error":
                    None,
            }

        except PermissionError as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            ValueError,
            LookupError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "error",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

    def _resolve_delete_state(
        self,
        *,
        project_id: str,
        project_key: str,
        project_name: str,
    ) -> str:
        """
        Classify one exact trusted Jira project for governed deletion.

        Returns:
            live
            archived
            deleted
            unknown

        "deleted" requires:
            - immutable project ID no longer resolves; and
            - the exact project is absent from both trusted
              live and archived project-search views.
        """

        (
            lookup_status,
            payload,
        ) = (
            self._lookup_project(
                project_id
            )
        )

        try:
            archived = (
                self._project_in_status(
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                    status="archived",
                )
            )

            live = (
                self._project_in_status(
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                    status="live",
                )
            )

        except (
            PermissionError,
            LookupError,
            RuntimeError,
            ValueError,
            httpx.HTTPError,
        ):
            return (
                "unknown"
            )

        if (
            lookup_status
            == "missing"
            and not live
            and not archived
        ):
            return (
                "deleted"
            )

        if (
            lookup_status
            == "found"
            and isinstance(
                payload,
                dict,
            )
        ):
            try:
                snapshot = (
                    self._project_snapshot(
                        payload
                    )
                )

            except (
                RuntimeError,
                ValueError,
            ):
                return (
                    "unknown"
                )

            exact = (
                snapshot.get(
                    "id"
                )
                == project_id

                and snapshot.get(
                    "key"
                )
                == project_key

                and snapshot.get(
                    "name"
                )
                == project_name
            )

            if not exact:
                return (
                    "unknown"
                )

            if (
                live
                and not archived
            ):
                return (
                    "live"
                )

            if (
                archived
                and not live
            ):
                return (
                    "archived"
                )

        return (
            "unknown"
        )

    def prepare_delete_project(
        self,
        *,
        project_id_or_key: str,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Read-only HIGH-risk Jira project deletion preparation.

        Only a provably LIVE project is eligible.

        Trusted policy freezes:
            immutable project ID
            exact project key
            exact project name
        """

        try:
            resolved_reference = (
                self._normalize_project_reference(
                    project_id_or_key
                )
            )

            current = (
                self._get_project_snapshot(
                    resolved_reference
                )
            )

            state = (
                self._resolve_delete_state(
                    project_id=(
                        current[
                            "id"
                        ]
                    ),
                    project_key=(
                        current[
                            "key"
                        ]
                    ),
                    project_name=(
                        current[
                            "name"
                        ]
                    ),
                )
            )

            if state == "archived":
                raise ValueError(
                    "The Jira project is archived. "
                    "Archived projects are not eligible for "
                    "direct governed deletion."
                )

            if state != "live":
                raise RuntimeError(
                    "Trusted policy could not prove that "
                    "the Jira project is currently live."
                )

            return {
                "ok":
                    True,

                "status":
                    "ready",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "execution_arguments": {
                    "project_id_or_key":
                        resolved_reference,

                    "expected_project_id":
                        current[
                            "id"
                        ],

                    "expected_project_key":
                        current[
                            "key"
                        ],

                    "expected_project_name":
                        current[
                            "name"
                        ],
                },

                "error":
                    None,
            }

        except PermissionError as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            ValueError,
            LookupError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "denied",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            return {
                "ok":
                    False,

                "status":
                    "error",

                "risk":
                    "high",

                "requires_approval":
                    True,

                "error":
                    str(
                        exc
                    ),
            }

    @staticmethod
    def _delete_result(
        *,
        ok: bool,
        status: str,
        project_id: str,
        project_key: str,
        project_name: str,
        deleted: (
            bool
            | None
        ),
        mutation_performed: (
            bool
            | None
        ),
        verification_ok: bool,
        reconciled: bool,
        error: (
            str
            | None
        ),
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                ok,

            "status":
                status,

            "operation":
                "delete",

            "project_id":
                project_id,

            "project_key":
                project_key,

            "project_name":
                project_name,

            "previous_project_name":
                None,

            "new_project_name":
                None,

            "template":
                None,

            "project_type_key":
                None,

            "archived":
                None,

            "deleted":
                deleted,

            "mutation_performed":
                mutation_performed,

            "verification_ok":
                verification_ok,

            "reconciled":
                reconciled,

            "error":
                error,
        }

    def _reconcile_uncertain_delete(
        self,
        *,
        project_id: str,
        project_key: str,
        project_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        try:
            state = (
                self._resolve_delete_state(
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                )
            )

        except Exception:
            state = (
                "unknown"
            )

        if state == "deleted":

            return (
                self._delete_result(
                    ok=True,
                    status="success",
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                    deleted=True,
                    mutation_performed=True,
                    verification_ok=True,
                    reconciled=True,
                    error=None,
                )
            )

        if state == "live":

            return (
                self._delete_result(
                    ok=False,
                    status="error",
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                    deleted=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=True,
                    error=(
                        "The Jira deletion request had an uncertain "
                        "transport outcome, but trusted read-back "
                        "still shows the exact project as live."
                    ),
                )
            )

        return (
            self._delete_result(
                ok=False,
                status="outcome_unknown",
                project_id=(
                    project_id
                ),
                project_key=(
                    project_key
                ),
                project_name=(
                    project_name
                ),
                deleted=None,
                mutation_performed=None,
                verification_ok=False,
                reconciled=False,
                error=(
                    "Jira project deletion had an ambiguous "
                    "provider outcome and trusted read-back "
                    "could not prove the requested final state."
                ),
            )
        )

    def delete_project(
        self,
        *,
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Execute one exact approved Jira project deletion.

        Trusted execution always sends:
            enableUndo=true

        The model cannot weaken that boundary.
        """

        resolved_project_id = None
        resolved_project_key = None
        resolved_project_name = None

        try:
            self._normalize_project_reference(
                project_id_or_key
            )

            resolved_project_id = (
                _exact_string(
                    expected_project_id,
                    field_name=(
                        "expected_project_id"
                    ),
                    max_length=100,
                )
            )

            resolved_project_key = (
                _exact_string(
                    expected_project_key,
                    field_name=(
                        "expected_project_key"
                    ),
                    max_length=(
                        MAX_PROJECT_KEY_CHARS
                    ),
                )
            )

            resolved_project_name = (
                _exact_string(
                    expected_project_name,
                    field_name=(
                        "expected_project_name"
                    ),
                    max_length=(
                        MAX_PROJECT_NAME_CHARS
                    ),
                )
            )

            # =================================================
            # POST-APPROVAL SNAPSHOT REVALIDATION
            # =================================================

            current = (
                self._get_project_snapshot(
                    resolved_project_id
                )
            )

            if not (
                self._snapshot_matches(
                    current,
                    expected_project_id=(
                        resolved_project_id
                    ),
                    expected_project_key=(
                        resolved_project_key
                    ),
                    expected_project_name=(
                        resolved_project_name
                    ),
                )
            ):
                raise ValueError(
                    "Jira project state changed after approval. "
                    "The approved deletion snapshot is stale."
                )

            state = (
                self._resolve_delete_state(
                    project_id=(
                        resolved_project_id
                    ),
                    project_key=(
                        resolved_project_key
                    ),
                    project_name=(
                        resolved_project_name
                    ),
                )
            )

            if state != "live":
                raise ValueError(
                    "The approved Jira project is no longer "
                    "provably live. Refusing stale deletion."
                )

            # =================================================
            # REMOTE MUTATION
            # =================================================

            try:
                response = (
                    self.client.delete(
                        (
                            "/rest/api/3/project/"
                            + quote(
                                resolved_project_id,
                                safe="",
                            )
                        ),
                        params={
                            "enableUndo":
                                "true",
                        },
                    )
                )

            except (
                httpx.TimeoutException,
                httpx.TransportError,
            ):

                return (
                    self._reconcile_uncertain_delete(
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                    )
                )

            if response.status_code in {
                400,
                401,
                403,
                404,
                409,
            }:

                return (
                    self._delete_result(
                        ok=False,
                        status="denied",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                        deleted=False,
                        mutation_performed=False,
                        verification_ok=False,
                        reconciled=False,
                        error=(
                            "Jira rejected the approved project "
                            "deletion. No successful provider "
                            "mutation was confirmed."
                        ),
                    )
                )

            if response.status_code >= 500:

                return (
                    self._reconcile_uncertain_delete(
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                    )
                )

            if response.status_code != 204:

                return (
                    self._delete_result(
                        ok=False,
                        status="outcome_unknown",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                        deleted=None,
                        mutation_performed=None,
                        verification_ok=False,
                        reconciled=False,
                        error=(
                            "Jira returned an unexpected "
                            "response to project deletion."
                        ),
                    )
                )

            # =================================================
            # AUTHORITATIVE READ-BACK
            # =================================================

            final_state = (
                self._resolve_delete_state(
                    project_id=(
                        resolved_project_id
                    ),
                    project_key=(
                        resolved_project_key
                    ),
                    project_name=(
                        resolved_project_name
                    ),
                )
            )

            if final_state == "deleted":

                return (
                    self._delete_result(
                        ok=True,
                        status="success",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                        deleted=True,
                        mutation_performed=True,
                        verification_ok=True,
                        reconciled=False,
                        error=None,
                    )
                )

            return (
                self._delete_result(
                    ok=False,
                    status="outcome_unknown",
                    project_id=(
                        resolved_project_id
                    ),
                    project_key=(
                        resolved_project_key
                    ),
                    project_name=(
                        resolved_project_name
                    ),
                    deleted=(
                        False
                        if final_state
                        in {
                            "live",
                            "archived",
                        }
                        else None
                    ),
                    mutation_performed=None,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira reported successful project deletion, "
                        "but trusted read-back did not verify the "
                        "requested final state."
                    ),
                )
            )

        except PermissionError as exc:
            status = (
                "denied"
            )

            error = (
                str(
                    exc
                )
            )

        except (
            ValueError,
            LookupError,
        ) as exc:
            status = (
                "denied"
            )

            error = (
                str(
                    exc
                )
            )

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:
            status = (
                "error"
            )

            error = (
                str(
                    exc
                )
            )

        return (
            self._delete_result(
                ok=False,
                status=(
                    status
                ),
                project_id=(
                    resolved_project_id
                    or (
                        expected_project_id
                        if isinstance(
                            expected_project_id,
                            str,
                        )
                        else ""
                    )
                ),
                project_key=(
                    resolved_project_key
                    or (
                        expected_project_key
                        if isinstance(
                            expected_project_key,
                            str,
                        )
                        else ""
                    )
                ),
                project_name=(
                    resolved_project_name
                    or (
                        expected_project_name
                        if isinstance(
                            expected_project_name,
                            str,
                        )
                        else ""
                    )
                ),
                deleted=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=(
                    error
                ),
            )
        )

    @staticmethod
    def _archive_result(
        *,
        ok: bool,
        status: str,
        project_id: str,
        project_key: str,
        project_name: str,
        archived: (
            bool
            | None
        ),
        mutation_performed: (
            bool
            | None
        ),
        verification_ok: bool,
        reconciled: bool,
        error: (
            str
            | None
        ),
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                ok,

            "status":
                status,

            "operation":
                "archive",

            "project_id":
                project_id,

            "project_key":
                project_key,

            "project_name":
                project_name,

            "previous_project_name":
                None,

            "new_project_name":
                None,

            "template":
                None,

            "project_type_key":
                None,

            "archived":
                archived,

            "mutation_performed":
                mutation_performed,

            "verification_ok":
                verification_ok,

            "reconciled":
                reconciled,

            "error":
                error,
        }

    def _reconcile_uncertain_archive(
        self,
        *,
        project_id: str,
        project_key: str,
        project_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        try:
            state = (
                self._resolve_archive_state(
                    project_id=(
                        project_id
                    ),

                    project_key=(
                        project_key
                    ),

                    project_name=(
                        project_name
                    ),
                )
            )

        except Exception:

            state = (
                "unknown"
            )

        if state == "archived":

            return (
                self._archive_result(
                    ok=True,
                    status="success",
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                    archived=True,
                    mutation_performed=True,
                    verification_ok=True,
                    reconciled=True,
                    error=None,
                )
            )

        if state == "live":

            return (
                self._archive_result(
                    ok=False,
                    status="error",
                    project_id=(
                        project_id
                    ),
                    project_key=(
                        project_key
                    ),
                    project_name=(
                        project_name
                    ),
                    archived=False,
                    mutation_performed=False,
                    verification_ok=False,
                    reconciled=True,
                    error=(
                        "The archive request had an uncertain "
                        "transport outcome, but trusted read-back "
                        "still shows the exact project as live."
                    ),
                )
            )

        return (
            self._archive_result(
                ok=False,
                status=(
                    "outcome_unknown"
                ),
                project_id=(
                    project_id
                ),
                project_key=(
                    project_key
                ),
                project_name=(
                    project_name
                ),
                archived=None,
                mutation_performed=None,
                verification_ok=False,
                reconciled=False,
                error=(
                    "Jira project archival had an ambiguous "
                    "provider outcome and trusted read-back "
                    "could not prove whether the project is "
                    "live or archived."
                ),
            )
        )

    def archive_project(
        self,
        *,
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Execute one exact approved Jira project archive.

        The POST path uses the trusted immutable project ID
        captured before approval.
        """

        resolved_project_id = None
        resolved_project_key = None
        resolved_project_name = None

        try:
            # Preserve and validate the original semantic
            # project reference even though provider execution
            # uses the trusted immutable ID.
            self._normalize_project_reference(
                project_id_or_key
            )

            resolved_project_id = (
                _exact_string(
                    expected_project_id,
                    field_name=(
                        "expected_project_id"
                    ),
                    max_length=100,
                )
            )

            resolved_project_key = (
                _exact_string(
                    expected_project_key,
                    field_name=(
                        "expected_project_key"
                    ),
                    max_length=(
                        MAX_PROJECT_KEY_CHARS
                    ),
                )
            )

            resolved_project_name = (
                _exact_string(
                    expected_project_name,
                    field_name=(
                        "expected_project_name"
                    ),
                    max_length=(
                        MAX_PROJECT_NAME_CHARS
                    ),
                )
            )

            # =================================================
            # POST-APPROVAL EXACT SNAPSHOT REVALIDATION
            # =================================================

            current = (
                self._get_project_snapshot(
                    resolved_project_id
                )
            )

            if not (
                self._snapshot_matches(
                    current,
                    expected_project_id=(
                        resolved_project_id
                    ),
                    expected_project_key=(
                        resolved_project_key
                    ),
                    expected_project_name=(
                        resolved_project_name
                    ),
                )
            ):
                raise ValueError(
                    "Jira project state changed after approval. "
                    "The approved archive snapshot is stale."
                )

            state = (
                self._resolve_archive_state(
                    project_id=(
                        resolved_project_id
                    ),

                    project_key=(
                        resolved_project_key
                    ),

                    project_name=(
                        resolved_project_name
                    ),
                )
            )

            if state != "live":
                raise ValueError(
                    "The approved Jira project is no longer "
                    "provably live. Refusing stale archive "
                    "execution."
                )

            # =================================================
            # REMOTE MUTATION
            # =================================================

            try:
                response = (
                    self.client.post(
                        (
                            "/rest/api/3/project/"
                            + quote(
                                resolved_project_id,
                                safe="",
                            )
                            + "/archive"
                        )
                    )
                )

            except (
                httpx.TimeoutException,
                httpx.TransportError,
            ):

                return (
                    self._reconcile_uncertain_archive(
                        project_id=(
                            resolved_project_id
                        ),

                        project_key=(
                            resolved_project_key
                        ),

                        project_name=(
                            resolved_project_name
                        ),
                    )
                )

            if response.status_code in {
                400,
                401,
                403,
                404,
                409,
            }:

                return (
                    self._archive_result(
                        ok=False,
                        status="denied",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                        archived=False,
                        mutation_performed=False,
                        verification_ok=False,
                        reconciled=False,
                        error=(
                            "Jira rejected the approved project "
                            "archive. No successful provider "
                            "mutation was confirmed."
                        ),
                    )
                )

            if response.status_code >= 500:

                return (
                    self._reconcile_uncertain_archive(
                        project_id=(
                            resolved_project_id
                        ),

                        project_key=(
                            resolved_project_key
                        ),

                        project_name=(
                            resolved_project_name
                        ),
                    )
                )

            if response.status_code != 204:

                return (
                    self._archive_result(
                        ok=False,
                        status=(
                            "outcome_unknown"
                        ),
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                        archived=None,
                        mutation_performed=None,
                        verification_ok=False,
                        reconciled=False,
                        error=(
                            "Jira returned an unexpected "
                            "response to project archival."
                        ),
                    )
                )

            # =================================================
            # AUTHORITATIVE LIFECYCLE READ-BACK
            # =================================================

            final_state = (
                self._resolve_archive_state(
                    project_id=(
                        resolved_project_id
                    ),

                    project_key=(
                        resolved_project_key
                    ),

                    project_name=(
                        resolved_project_name
                    ),
                )
            )

            if final_state == "archived":

                return (
                    self._archive_result(
                        ok=True,
                        status="success",
                        project_id=(
                            resolved_project_id
                        ),
                        project_key=(
                            resolved_project_key
                        ),
                        project_name=(
                            resolved_project_name
                        ),
                        archived=True,
                        mutation_performed=True,
                        verification_ok=True,
                        reconciled=False,
                        error=None,
                    )
                )

            # Jira returned 204 but trusted lifecycle state
            # contradicts or cannot prove the requested result.
            return (
                self._archive_result(
                    ok=False,
                    status=(
                        "outcome_unknown"
                    ),
                    project_id=(
                        resolved_project_id
                    ),
                    project_key=(
                        resolved_project_key
                    ),
                    project_name=(
                        resolved_project_name
                    ),
                    archived=(
                        False
                        if final_state
                        == "live"
                        else None
                    ),
                    mutation_performed=None,
                    verification_ok=False,
                    reconciled=False,
                    error=(
                        "Jira reported successful archival, "
                        "but trusted lifecycle read-back did "
                        "not verify the exact project as archived."
                    ),
                )
            )

        except PermissionError as exc:

            status = (
                "denied"
            )

            error = (
                str(
                    exc
                )
            )

        except (
            ValueError,
            LookupError,
        ) as exc:

            status = (
                "denied"
            )

            error = (
                str(
                    exc
                )
            )

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            status = (
                "error"
            )

            error = (
                str(
                    exc
                )
            )

        return (
            self._archive_result(
                ok=False,
                status=(
                    status
                ),
                project_id=(
                    resolved_project_id
                    or (
                        expected_project_id
                        if isinstance(
                            expected_project_id,
                            str,
                        )
                        else ""
                    )
                ),
                project_key=(
                    resolved_project_key
                    or (
                        expected_project_key
                        if isinstance(
                            expected_project_key,
                            str,
                        )
                        else ""
                    )
                ),
                project_name=(
                    resolved_project_name
                    or (
                        expected_project_name
                        if isinstance(
                            expected_project_name,
                            str,
                        )
                        else ""
                    )
                ),
                archived=False,
                mutation_performed=False,
                verification_ok=False,
                reconciled=False,
                error=(
                    error
                ),
            )
        )

    def _success_result(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
        project_type_key: str,
        project_id: (
            str
            | None
        ),
        reconciled: bool,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                True,

            "status":
                "success",

            "operation":
                "create",

            "project_id":
                project_id,

            "project_key":
                project_key,

            "project_name":
                project_name,

            "template":
                template,

            "project_type_key":
                project_type_key,

            "mutation_performed":
                True,

            "verification_ok":
                True,

            "reconciled":
                reconciled,

            "error":
                None,
        }

    def _ambiguous_result(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
        project_type_key: str,
        mutation_performed: (
            bool
            | None
        ),
        error: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "outcome_unknown",

            "operation":
                "create",

            "project_id":
                None,

            "project_key":
                project_key,

            "project_name":
                project_name,

            "template":
                template,

            "project_type_key":
                project_type_key,

            "mutation_performed":
                mutation_performed,

            "verification_ok":
                False,

            "reconciled":
                False,

            "error":
                error,
        }

    def _reconcile_uncertain_create(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
        project_type_key: str,
    ) -> dict[
        str,
        Any,
    ]:

        (
            lookup_status,
            payload,
        ) = (
            self._lookup_project(
                project_key
            )
        )

        if (
            lookup_status
            == "found"
            and isinstance(
                payload,
                dict,
            )
            and self._verified_project(
                payload,
                project_key=(
                    project_key
                ),
                project_name=(
                    project_name
                ),
                project_type_key=(
                    project_type_key
                ),
            )
        ):

            raw_id = (
                payload.get(
                    "id"
                )
            )

            project_id = (
                str(
                    raw_id
                )
                if raw_id
                is not None
                else None
            )

            return (
                self._success_result(
                    project_key=(
                        project_key
                    ),

                    project_name=(
                        project_name
                    ),

                    template=(
                        template
                    ),

                    project_type_key=(
                        project_type_key
                    ),

                    project_id=(
                        project_id
                    ),

                    reconciled=True,
                )
            )

        return (
            self._ambiguous_result(
                project_key=(
                    project_key
                ),

                project_name=(
                    project_name
                ),

                template=(
                    template
                ),

                project_type_key=(
                    project_type_key
                ),

                mutation_performed=(
                    None
                ),

                error=(
                    "Jira project creation returned an "
                    "ambiguous provider outcome and trusted "
                    "read-back could not prove the final state."
                ),
            )
        )

    def create_project(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
        expected_project_type_key: str,
        expected_project_template_key: str,
        expected_lead_account_id: str,
    ) -> dict[
        str,
        Any,
    ]:

        try:
            (
                resolved_key,
                resolved_name,
                resolved_template,
                profile,
            ) = (
                self._normalize_inputs(
                    project_key=(
                        project_key
                    ),

                    project_name=(
                        project_name
                    ),

                    template=(
                        template
                    ),
                )
            )

            expected_type = (
                _exact_string(
                    expected_project_type_key,
                    field_name=(
                        "expected_project_type_key"
                    ),
                    max_length=100,
                )
            )

            expected_template = (
                _exact_string(
                    expected_project_template_key,
                    field_name=(
                        "expected_project_template_key"
                    ),
                    max_length=512,
                )
            )

            expected_lead = (
                _exact_string(
                    expected_lead_account_id,
                    field_name=(
                        "expected_lead_account_id"
                    ),
                    max_length=256,
                )
            )

            # ----------------------------------------------------
            # Approval snapshot revalidation.
            # ----------------------------------------------------

            if (
                profile[
                    "project_type_key"
                ]
                != expected_type
            ):
                raise ValueError(
                    "Approved Jira project type no longer "
                    "matches trusted template configuration."
                )

            if (
                profile[
                    "project_template_key"
                ]
                != expected_template
            ):
                raise ValueError(
                    "Approved Jira project template no longer "
                    "matches trusted template configuration."
                )

            current_lead = (
                self._current_account_id()
            )

            if (
                current_lead
                != expected_lead
            ):
                raise ValueError(
                    "The authenticated Jira identity changed "
                    "after approval."
                )

            self._validate_project_type_accessible(
                expected_type
            )

            # Re-check exact key/name immediately before mutation.
            self._validate_key_exact(
                resolved_key
            )

            self._validate_name_exact(
                resolved_name
            )

            body = {
                "key":
                    resolved_key,

                "name":
                    resolved_name,

                "projectTypeKey":
                    expected_type,

                "projectTemplateKey":
                    expected_template,

                "leadAccountId":
                    expected_lead,

                "assigneeType":
                    "PROJECT_LEAD",
            }

            try:
                response = (
                    self.client.post(
                        "/rest/api/3/project",
                        content=(
                            json.dumps(
                                body
                            )
                        ),
                    )
                )

            except (
                httpx.TimeoutException,
                httpx.TransportError,
            ):

                return (
                    self
                    ._reconcile_uncertain_create(
                        project_key=(
                            resolved_key
                        ),

                        project_name=(
                            resolved_name
                        ),

                        template=(
                            resolved_template
                        ),

                        project_type_key=(
                            expected_type
                        ),
                    )
                )

            # ----------------------------------------------------
            # 4xx means Jira rejected the create request.
            # Do not claim mutation success.
            # ----------------------------------------------------

            if response.status_code in {
                400,
                401,
                403,
                404,
                409,
            }:

                return {
                    "ok":
                        False,

                    "status":
                        "denied",

                    "operation":
                        "create",

                    "project_id":
                        None,

                    "project_key":
                        resolved_key,

                    "project_name":
                        resolved_name,

                    "template":
                        resolved_template,

                    "project_type_key":
                        expected_type,

                    "mutation_performed":
                        False,

                    "verification_ok":
                        False,

                    "reconciled":
                        False,

                    "error":
                        (
                            "Jira rejected project creation. "
                            "The authenticated account may lack "
                            "permission, the template may be "
                            "unavailable, or provider validation "
                            "may have changed."
                        ),
                }

            # A server-side response may be ambiguous.
            if response.status_code >= 500:

                return (
                    self
                    ._reconcile_uncertain_create(
                        project_key=(
                            resolved_key
                        ),

                        project_name=(
                            resolved_name
                        ),

                        template=(
                            resolved_template
                        ),

                        project_type_key=(
                            expected_type
                        ),
                    )
                )

            if response.status_code != 201:

                return (
                    self._ambiguous_result(
                        project_key=(
                            resolved_key
                        ),

                        project_name=(
                            resolved_name
                        ),

                        template=(
                            resolved_template
                        ),

                        project_type_key=(
                            expected_type
                        ),

                        mutation_performed=(
                            None
                        ),

                        error=(
                            "Jira returned an unexpected response "
                            "to project creation."
                        ),
                    )
                )

            try:
                creation_payload = (
                    response.json()
                )

            except ValueError:
                creation_payload = {}

            raw_project_id = (
                creation_payload.get(
                    "id"
                )
                if isinstance(
                    creation_payload,
                    dict,
                )
                else None
            )

            project_id = (
                str(
                    raw_project_id
                )
                if raw_project_id
                is not None
                else None
            )

            # ----------------------------------------------------
            # Read-back is authoritative for completion.
            # ----------------------------------------------------

            (
                lookup_status,
                project_payload,
            ) = (
                self._lookup_project(
                    resolved_key
                )
            )

            if (
                lookup_status
                == "found"
                and isinstance(
                    project_payload,
                    dict,
                )
                and self._verified_project(
                    project_payload,
                    project_key=(
                        resolved_key
                    ),

                    project_name=(
                        resolved_name
                    ),

                    project_type_key=(
                        expected_type
                    ),
                )
            ):

                if project_id is None:
                    raw_verified_id = (
                        project_payload.get(
                            "id"
                        )
                    )

                    project_id = (
                        str(
                            raw_verified_id
                        )
                        if raw_verified_id
                        is not None
                        else None
                    )

                return (
                    self._success_result(
                        project_key=(
                            resolved_key
                        ),

                        project_name=(
                            resolved_name
                        ),

                        template=(
                            resolved_template
                        ),

                        project_type_key=(
                            expected_type
                        ),

                        project_id=(
                            project_id
                        ),

                        reconciled=False,
                    )
                )

            # 201 means Jira reported creation, but our trusted
            # read-back could not prove the expected final state.
            return (
                self._ambiguous_result(
                    project_key=(
                        resolved_key
                    ),

                    project_name=(
                        resolved_name
                    ),

                    template=(
                        resolved_template
                    ),

                    project_type_key=(
                        expected_type
                    ),

                    mutation_performed=(
                        True
                    ),

                    error=(
                        "Jira reported project creation, but "
                        "trusted read-back could not verify the "
                        "expected final project state."
                    ),
                )
            )

        except PermissionError as exc:

            status = (
                "denied"
            )

            error = (
                str(
                    exc
                )
            )

        except (
            ValueError,
            LookupError,
        ) as exc:

            status = (
                "denied"
            )

            error = (
                str(
                    exc
                )
            )

        except (
            RuntimeError,
            httpx.HTTPError,
        ) as exc:

            status = (
                "error"
            )

            error = (
                str(
                    exc
                )
            )

        return {
            "ok":
                False,

            "status":
                status,

            "operation":
                "create",

            "project_id":
                None,

            "project_key":
                (
                    project_key
                    if isinstance(
                        project_key,
                        str,
                    )
                    else None
                ),

            "project_name":
                (
                    project_name
                    if isinstance(
                        project_name,
                        str,
                    )
                    else None
                ),

            "template":
                (
                    template
                    if isinstance(
                        template,
                        str,
                    )
                    else None
                ),

            "project_type_key":
                None,

            "mutation_performed":
                False,

            "verification_ok":
                False,

            "reconciled":
                False,

            "error":
                error,
        }


class UnavailableJiraProjectMutationService:

    ERROR = (
        "Jira project mutation access is not configured."
    )

    def prepare_create_project(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "risk":
                "medium",

            "requires_approval":
                True,

            "error":
                self.ERROR,
        }

    def prepare_update_project(
        self,
        *,
        project_id_or_key: str,
        new_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "risk":
                "medium",

            "requires_approval":
                True,

            "error":
                self.ERROR,
        }

    def update_project(
        self,
        *,
        project_id_or_key: str,
        new_name: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "operation":
                "update",

            "project_id":
                expected_project_id,

            "project_key":
                expected_project_key,

            "project_name":
                expected_project_name,

            "previous_project_name":
                expected_project_name,

            "new_project_name":
                new_name,

            "template":
                None,

            "project_type_key":
                None,

            "mutation_performed":
                False,

            "verification_ok":
                False,

            "reconciled":
                False,

            "error":
                self.ERROR,
        }

    def prepare_archive_project(
        self,
        *,
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

            "risk":
                "high",

            "requires_approval":
                True,

            "error":
                self.ERROR,
        }

    def archive_project(
        self,
        *,
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "operation":
                "archive",

            "project_id":
                expected_project_id,

            "project_key":
                expected_project_key,

            "project_name":
                expected_project_name,

            "previous_project_name":
                None,

            "new_project_name":
                None,

            "template":
                None,

            "project_type_key":
                None,

            "archived":
                False,

            "mutation_performed":
                False,

            "verification_ok":
                False,

            "reconciled":
                False,

            "error":
                self.ERROR,
        }

    def prepare_delete_project(
        self,
        *,
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

            "risk":
                "high",

            "requires_approval":
                True,

            "error":
                self.ERROR,
        }

    def delete_project(
        self,
        *,
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "operation":
                "delete",

            "project_id":
                expected_project_id,

            "project_key":
                expected_project_key,

            "project_name":
                expected_project_name,

            "previous_project_name":
                None,

            "new_project_name":
                None,

            "template":
                None,

            "project_type_key":
                None,

            "archived":
                None,

            "deleted":
                False,

            "mutation_performed":
                False,

            "verification_ok":
                False,

            "reconciled":
                False,

            "error":
                self.ERROR,
        }

    def create_project(
        self,
        *,
        project_key: str,
        project_name: str,
        template: str,
        expected_project_type_key: str,
        expected_project_template_key: str,
        expected_lead_account_id: str,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "ok":
                False,

            "status":
                "denied",

            "operation":
                "create",

            "project_id":
                None,

            "project_key":
                project_key,

            "project_name":
                project_name,

            "template":
                template,

            "project_type_key":
                None,

            "mutation_performed":
                False,

            "verification_ok":
                False,

            "reconciled":
                False,

            "error":
                self.ERROR,
        }


def build_jira_project_mutation_service(
    settings: Settings,
    *,
    transport: (
        httpx.BaseTransport
        | None
    ) = None,
) -> JiraProjectMutationProvider:

    if not (
        settings.jira_base_url
        and settings.jira_email
        and settings.jira_api_token
    ):
        return (
            UnavailableJiraProjectMutationService()
        )

    return (
        JiraProjectMutationService(
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

            transport=(
                transport
            ),
        )
    )
