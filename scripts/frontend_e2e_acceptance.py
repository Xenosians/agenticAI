#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from playwright.sync_api import (
        sync_playwright,
        TimeoutError as PlaywrightTimeoutError,
    )
except ImportError:
    print(
        "\nPlaywright is not installed.\n\n"
        "Run:\n"
        "  python -m pip install playwright\n"
        "  python -m playwright install chromium\n",
        file=sys.stderr,
    )
    raise SystemExit(2)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    ROOT
    / "config"
    / "cross_repo_acceptance_cases.json"
)

DEFAULT_REPORT_DIR = (
    ROOT
    / ".runtime"
    / "acceptance"
    / "frontend-e2e"
)

TERMINAL = {
    "completed",
    "failed",
    "denied",
    "outcome_unknown",
    "reconciliation_required",
}

# Cases where we explicitly prove reload persistence.
RELOAD_CASES = {
    "workspace-project-info-read",
    "workspace-mkdir-approval-gate",
    "workspace-mkdir-rejection",
}

# This one deliberately reloads immediately after submission so the browser
# must recover the already-created durable job rather than resubmit it.
RECOVERY_CASE = "workspace-process-snapshot-read"


def utc_token() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")


def sha256_json(
    value: Any,
) -> str | None:
    if value is None:
        return None

    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def extract_proposal(
    data: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
    str | None,
]:
    """
    Return:
      tool,
      approval_id,
      hash(arguments)

    We intentionally do not persist raw arguments in the report.
    """

    proposal = data.get(
        "proposed_tool"
    )

    if isinstance(
        proposal,
        dict,
    ):
        tool = proposal.get(
            "tool"
        )

        approval_id = proposal.get(
            "approval_id"
        )

        args = proposal.get(
            "arguments"
        )

        return (
            tool
            if isinstance(
                tool,
                str,
            )
            else None,
            approval_id
            if isinstance(
                approval_id,
                str,
            )
            else None,
            sha256_json(
                args
            ),
        )

    if isinstance(
        proposal,
        str,
    ) and proposal:
        return (
            proposal,
            None,
            None,
        )

    # Read-only results commonly expose the tool inside the specialist result.
    result = data.get(
        "result"
    )

    if isinstance(
        result,
        dict,
    ):
        results = result.get(
            "results"
        )

        if isinstance(
            results,
            list,
        ):
            for item in results:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                tool = (
                    item.get(
                        "proposed_tool"
                    )
                    or item.get(
                        "tool"
                    )
                )

                if not isinstance(
                    tool,
                    str,
                ) or not tool:
                    continue

                args = item.get(
                    "proposed_arguments"
                )

                if args is None:
                    args = item.get(
                        "arguments"
                    )

                approval_id = item.get(
                    "approval_id"
                )

                return (
                    tool,
                    approval_id
                    if isinstance(
                        approval_id,
                        str,
                    )
                    else None,
                    sha256_json(
                        args
                    ),
                )

    return (
        None,
        None,
        None,
    )


def summarize_job(
    data: dict[str, Any],
) -> dict[str, Any]:
    tool, approval_id, args_hash = (
        extract_proposal(
            data
        )
    )

    return {
        "job_id":
            data.get(
                "job_id"
            ),
        "status":
            data.get(
                "status"
            ),
        "selected_agent":
            data.get(
                "selected_agent"
            ),
        "tool":
            tool,
        "approval_id":
            approval_id,
        "arguments_sha256":
            args_hash,
        "result_present":
            data.get(
                "result"
            )
            is not None,
        "error":
            data.get(
                "error"
            )
            if isinstance(
                data.get(
                    "error"
                ),
                str,
            )
            else None,
    }


def expand_prompt(
    value: str,
    run_token: str,
) -> tuple[
    str,
    list[str],
]:
    missing: list[str] = []

    def replace_env(
        match: re.Match[str],
    ) -> str:
        name = match.group(
            1
        )

        current = os.environ.get(
            name
        )

        if not current:
            missing.append(
                name
            )

            return match.group(
                0
            )

        return current

    value = re.sub(
        r"\$\{([A-Z0-9_]+)\}",
        replace_env,
        value,
    )

    value = value.replace(
        "{run_token}",
        run_token,
    )

    return (
        value,
        sorted(
            set(
                missing
            )
        ),
    )


def load_cases(
    path: Path,
    selected: list[str],
) -> list[
    dict[str, Any]
]:
    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    cases = [
        item
        for item
        in raw[
            "cases"
        ]
        if item.get(
            "enabled",
            True,
        )
    ]

    if not selected:
        return cases

    wanted = set(
        selected
    )

    known = {
        item["id"]
        for item
        in cases
    }

    unknown = sorted(
        wanted
        - known
    )

    if unknown:
        raise SystemExit(
            "Unknown case(s): "
            + ", ".join(
                unknown
            )
        )

    return [
        item
        for item
        in cases
        if item["id"]
        in wanted
    ]


class BrowserEvidence:
    """
    Observe only browser-generated traffic.

    This class never creates a backend job itself.
    All actions originate from the real frontend.
    """

    def __init__(
        self,
    ) -> None:
        self.current_case: (
            str
            | None
        ) = None

        self.job_snapshots: dict[
            str,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ] = {}

        self.socket_tokens: list[
            dict[
                str,
                Any,
            ]
        ] = []

        self.websockets: list[
            dict[
                str,
                Any,
            ]
        ] = []

    def attach(
        self,
        page,
    ) -> None:
        def on_response(
            response,
        ) -> None:
            case_id = (
                self.current_case
            )

            if not case_id:
                return

            url = response.url

            if (
                "/api/v2/jobs/socket-token"
                in url
            ):
                self.socket_tokens.append(
                    {
                        "case_id":
                            case_id,
                        "status":
                            response.status,
                    }
                )

            if (
                "/api/v2/jobs/"
                in url
                and response.request.method
                == "GET"
            ):
                try:
                    payload = (
                        response.json()
                    )
                except Exception:
                    return

                if not isinstance(
                    payload,
                    dict,
                ):
                    return

                job_id = (
                    payload.get(
                        "job_id"
                    )
                )

                if not isinstance(
                    job_id,
                    str,
                ):
                    return

                item = summarize_job(
                    payload
                )

                item[
                    "case_id"
                ] = case_id

                self.job_snapshots.setdefault(
                    job_id,
                    [],
                ).append(
                    item
                )

        def on_websocket(
            ws,
        ) -> None:
            case_id = (
                self.current_case
            )

            if not case_id:
                return

            item = {
                "case_id":
                    case_id,
                "url":
                    ws.url,
                "frames_received":
                    0,
                "frames_sent":
                    0,
            }

            self.websockets.append(
                item
            )

            def received(
                _payload,
            ) -> None:
                item[
                    "frames_received"
                ] += 1

            def sent(
                _payload,
            ) -> None:
                item[
                    "frames_sent"
                ] += 1

            ws.on(
                "framereceived",
                received,
            )

            ws.on(
                "framesent",
                sent,
            )

        page.on(
            "response",
            on_response,
        )

        page.on(
            "websocket",
            on_websocket,
        )

    def latest(
        self,
        job_id: str,
    ) -> (
        dict[
            str,
            Any,
        ]
        | None
    ):
        values = (
            self.job_snapshots.get(
                job_id,
                [],
            )
        )

        return (
            values[-1]
            if values
            else None
        )

    def wait_status(
        self,
        page,
        job_id: str,
        statuses: set[str],
        timeout_ms: int,
    ) -> dict[
        str,
        Any,
    ]:
        deadline = (
            time.monotonic()
            + timeout_ms
            / 1000
        )

        while (
            time.monotonic()
            < deadline
        ):
            current = (
                self.latest(
                    job_id
                )
            )

            if (
                current
                and current.get(
                    "status"
                )
                in statuses
            ):
                return current

            page.wait_for_timeout(
                200
            )

        raise TimeoutError(
            "job "
            + job_id
            + " did not reach "
            + repr(
                sorted(
                    statuses
                )
            )
            + "; latest="
            + repr(
                self.latest(
                    job_id
                )
            )
        )


def login(
    page,
    frontend_url: str,
    timeout_ms: int,
    force_login: bool = False,
) -> dict[str, Any]:
    if force_login:
        page.context.clear_cookies()

    page.goto(
        frontend_url,
        wait_until="domcontentloaded",
    )

    # Reuse a valid browser session only when acceptance does not
    # explicitly require a fresh frontend login.
    try:
        if force_login:
            raise PlaywrightTimeoutError("fresh login required")

        page.locator(
            "#messageInput"
        ).wait_for(
            state="visible",
            timeout=4000,
        )

        login_mode = (
            "existing_session"
        )

    except PlaywrightTimeoutError:
        email = (
            os.environ.get(
                "ACCEPTANCE_EMAIL"
            )
            or input(
                "Acceptance app email: "
            ).strip()
        )

        password = (
            os.environ.get(
                "ACCEPTANCE_PASSWORD"
            )
            or getpass.getpass(
                "Acceptance app password: "
            )
        )

        if (
            not email
            or not password
        ):
            raise RuntimeError(
                "acceptance login credentials are required"
            )

        email_input = (
            page.locator(
                'input[type="email"]'
            ).first
        )

        password_input = (
            page.locator(
                'input[type="password"]'
            ).first
        )

        email_input.wait_for(
            state="visible",
            timeout=timeout_ms,
        )

        email_input.fill(
            email
        )

        # The current Karax auth field commits on change/blur.
        email_input.press(
            "Tab"
        )

        password_input.fill(
            password
        )

        password_input.press(
            "Tab"
        )

        page.get_by_role(
            "button",
            name="Sign in",
            exact=True,
        ).click()

        try:
            page.locator(
                "#messageInput"
            ).wait_for(
                state="visible",
                timeout=timeout_ms,
            )

        except PlaywrightTimeoutError:
            auth_error = (
                page.locator(
                    ".auth-error"
                )
            )

            detail = (
                auth_error.inner_text()
                if auth_error.count()
                else
                "workspace did not appear"
            )

            raise RuntimeError(
                "frontend login failed: "
                + detail
            )

        login_mode = (
            "fresh_login"
        )

    # Prove normal session restoration.
    page.reload(
        wait_until="domcontentloaded"
    )

    page.locator(
        "#messageInput"
    ).wait_for(
        state="visible",
        timeout=timeout_ms,
    )

    return {
        "login_mode":
            login_mode,
        "session_reload":
            "PASS",
    }


def new_chat(
    page,
    timeout_ms: int,
) -> None:
    page.get_by_title(
        "More"
    ).click()

    page.get_by_role(
        "button",
        name="New chat",
        exact=True,
    ).click()

    page.locator(
        "#messageInput"
    ).wait_for(
        state="visible",
        timeout=timeout_ms,
    )

    # Allow the async durable chat creation response to settle.
    page.wait_for_timeout(
        300
    )


def send_prompt(
    page,
    prompt: str,
    timeout_ms: int,
) -> tuple[
    str,
    dict[str, Any],
]:
    textarea = (
        page.locator(
            "#messageInput"
        )
    )

    textarea.fill(
        prompt
    )

    send = (
        page.get_by_role(
            "button",
            name="Send message",
        )
    )

    deadline = (
        time.monotonic()
        + timeout_ms
        / 1000
    )

    while (
        time.monotonic()
        < deadline
    ):
        if send.is_enabled():
            break

        page.wait_for_timeout(
            100
        )
    else:
        raise TimeoutError(
            "frontend Send button never became enabled"
        )

    with page.expect_response(
        lambda response:
            response.url.rstrip(
                "/"
            ).endswith(
                "/api/v1/jobs"
            )
            and response.request.method
            == "POST",
        timeout=timeout_ms,
    ) as response_info:
        send.click()

    response = (
        response_info.value
    )

    payload = (
        response.json()
    )

    if (
        response.status
        >= 400
    ):
        raise RuntimeError(
            "frontend job submission returned HTTP "
            + str(
                response.status
            )
        )

    job_id = payload.get(
        "job_id"
    )

    if not isinstance(
        job_id,
        str,
    ) or not job_id:
        raise RuntimeError(
            "frontend submission returned no job_id"
        )

    return (
        job_id,
        payload,
    )


def require_tool(
    snapshot: dict[str, Any],
    expected: str | None,
) -> None:
    if not expected:
        return

    actual = (
        snapshot.get(
            "tool"
        )
    )

    if (
        actual
        != expected
    ):
        raise AssertionError(
            f"expected tool {expected!r}, observed {actual!r}"
        )


def reload_check(
    page,
    evidence: BrowserEvidence,
    job_id: str,
    before: dict[str, Any],
    timeout_ms: int,
) -> dict[str, Any]:
    before_count = len(
        evidence.job_snapshots.get(
            job_id,
            [],
        )
    )

    page.reload(
        wait_until="domcontentloaded"
    )

    page.locator(
        "#messageInput"
    ).wait_for(
        state="visible",
        timeout=timeout_ms,
    )

    deadline = (
        time.monotonic()
        + timeout_ms
        / 1000
    )

    while (
        time.monotonic()
        < deadline
    ):
        values = (
            evidence.job_snapshots.get(
                job_id,
                [],
            )
        )

        if (
            len(
                values
            )
            > before_count
        ):
            after = (
                values[-1]
            )

            if (
                after.get(
                    "status"
                )
                != before.get(
                    "status"
                )
            ):
                page.wait_for_timeout(
                    150
                )
                continue

            if (
                before.get(
                    "approval_id"
                )
                and after.get(
                    "approval_id"
                )
                != before.get(
                    "approval_id"
                )
            ):
                raise AssertionError(
                    "reload changed the saved approval identity"
                )

            if (
                before.get(
                    "arguments_sha256"
                )
                and after.get(
                    "arguments_sha256"
                )
                != before.get(
                    "arguments_sha256"
                )
            ):
                raise AssertionError(
                    "reload changed the exact saved arguments"
                )

            return after

        page.wait_for_timeout(
            200
        )

    raise TimeoutError(
        "reload did not reread the same durable job"
    )


def run_case(
    page,
    evidence: BrowserEvidence,
    case: dict[str, Any],
    *,
    run_token: str,
    timeout_ms: int,
    execute_disposable: bool,
    workspace_root: Path,
    integrations: set[str],
) -> dict[str, Any]:
    case_id = (
        case["id"]
    )

    result: dict[
        str,
        Any,
    ] = {
        "id":
            case_id,
        "kind":
            case.get(
                "kind"
            ),
        "expected_tool":
            case.get(
                "expected_tool"
            ),
        "status":
            "FAIL",
    }

    required_env = (
        case.get(
            "requires_env",
            [],
        )
    )

    missing = [
        name
        for name
        in required_env
        if not os.environ.get(
            name
        )
    ]

    if missing:
        result.update(
            status="SKIP",
            reason="missing_env",
            missing_env=missing,
        )

        return result

    integration = (
        case.get(
            "requires_integration"
        )
    )

    if (
        integration
        and integration
        not in integrations
    ):
        result.update(
            status="SKIP",
            reason="integration_not_selected",
            integration=integration,
        )

        return result

    prompt, prompt_missing = (
        expand_prompt(
            case[
                "prompt"
            ],
            run_token,
        )
    )

    if prompt_missing:
        result.update(
            status="SKIP",
            reason="prompt_env_missing",
            missing_env=prompt_missing,
        )

        return result

    evidence.current_case = (
        case_id
    )

    new_chat(
        page,
        timeout_ms,
    )

    job_id, created = (
        send_prompt(
            page,
            prompt,
            timeout_ms,
        )
    )

    result[
        "job_id"
    ] = job_id

    result[
        "submission_status"
    ] = created.get(
        "status"
    )

    # Simulate a browser disappearing immediately after submission.
    # We do NOT submit the prompt again.
    if (
        case_id
        == RECOVERY_CASE
    ):
        page.reload(
            wait_until="domcontentloaded"
        )

        page.locator(
            "#messageInput"
        ).wait_for(
            state="visible",
            timeout=timeout_ms,
        )

        result[
            "midflight_reload"
        ] = "EXERCISED"

    kind = case.get(
        "kind",
        "read",
    )

    if kind in {
        "approval_gate",
        "rejection_gate",
    }:
        waiting = (
            evidence.wait_status(
                page,
                job_id,
                {
                    "waiting_approval"
                },
                timeout_ms,
            )
        )

        require_tool(
            waiting,
            case.get(
                "expected_tool"
            ),
        )

        result[
            "waiting_approval"
        ] = waiting

        if (
            kind
            == "rejection_gate"
            or not execute_disposable
        ):
            page.get_by_role(
                "button",
                name="Reject action",
                exact=True,
            ).click()

            final = (
                evidence.wait_status(
                    page,
                    job_id,
                    {
                        "denied",
                        "failed",
                    },
                    timeout_ms,
                )
            )

            expected = set(
                case.get(
                    "expected_post_rejection_status",
                    [
                        "denied"
                    ],
                )
            )

            if (
                final[
                    "status"
                ]
                not in expected
            ):
                raise AssertionError(
                    "rejection expected "
                    + repr(
                        sorted(
                            expected
                        )
                    )
                    + ", got "
                    + repr(
                        final[
                            "status"
                        ]
                    )
                )

            result[
                "decision"
            ] = "rejected"

            result[
                "final"
            ] = final

        else:
            # Only the explicitly disposable workspace mkdir gate gets here.
            page.get_by_role(
                "button",
                name="Approve action",
                exact=True,
            ).click()

            final = (
                evidence.wait_status(
                    page,
                    job_id,
                    TERMINAL,
                    timeout_ms,
                )
            )

            if (
                final[
                    "status"
                ]
                != "completed"
            ):
                raise AssertionError(
                    "approved disposable mutation ended as "
                    + repr(
                        final[
                            "status"
                        ]
                    )
                )

            result[
                "decision"
            ] = "approved"

            result[
                "final"
            ] = final

            match = re.search(
                r"(acceptance_probe_[A-Za-z0-9T_-]+)",
                prompt,
            )

            if match:
                target = (
                    workspace_root
                    / match.group(
                        1
                    )
                )

                if not target.is_dir():
                    raise AssertionError(
                        "independent mkdir readback missing: "
                        + str(
                            target
                        )
                    )

                result[
                    "independent_readback"
                ] = str(
                    target
                )

                # The workspace mkdir fixture should be an empty directory.
                target.rmdir()

                result[
                    "cleanup"
                ] = "PASS"

    elif kind in {
        "fail_closed",
        "security_fail_closed",
    }:
        expected = set(
            case.get(
                "expected_terminal_status",
                [
                    "failed",
                    "denied",
                    "completed",
                ],
            )
        )

        final = (
            evidence.wait_status(
                page,
                job_id,
                expected
                | TERMINAL,
                timeout_ms,
            )
        )

        if (
            final[
                "status"
            ]
            not in expected
        ):
            raise AssertionError(
                "expected "
                + repr(
                    sorted(
                        expected
                    )
                )
                + ", got "
                + repr(
                    final[
                        "status"
                    ]
                )
            )

        if case.get(
            "forbid_waiting_approval"
        ):
            if any(
                item.get(
                    "status"
                )
                == "waiting_approval"
                for item
                in evidence.job_snapshots.get(
                    job_id,
                    [],
                )
            ):
                raise AssertionError(
                    "fail-closed request reached waiting_approval"
                )

        forbidden = (
            case.get(
                "forbid_approval_tool"
            )
        )

        if (
            forbidden
            and final.get(
                "tool"
            )
            == forbidden
        ):
            raise AssertionError(
                "forbidden tool was proposed"
            )

        if case.get(
            "expect_no_authority_delegation"
        ):
            if final.get(
                "selected_agent"
            ):
                raise AssertionError(
                    "ambiguous mutation delegated specialist authority"
                )

            if final.get(
                "tool"
            ):
                raise AssertionError(
                    "ambiguous mutation created a tool proposal"
                )

        required_error = (
            case.get(
                "expected_error_contains"
            )
        )

        if (
            required_error
            and required_error
            not in (
                final.get(
                    "error"
                )
                or ""
            )
        ):
            raise AssertionError(
                "durable error does not contain expected fail-closed text"
            )

        result[
            "final"
        ] = final

    else:
        expected = set(
            case.get(
                "expected_terminal_status",
                [
                    "completed"
                ],
            )
        )

        final = (
            evidence.wait_status(
                page,
                job_id,
                expected
                | TERMINAL,
                timeout_ms,
            )
        )

        if (
            final[
                "status"
            ]
            not in expected
        ):
            raise AssertionError(
                "expected "
                + repr(
                    sorted(
                        expected
                    )
                )
                + ", got "
                + repr(
                    final[
                        "status"
                    ]
                )
            )

        require_tool(
            final,
            case.get(
                "expected_tool"
            ),
        )

        result[
            "final"
        ] = final

    if (
        case_id
        in RELOAD_CASES
    ):
        after = (
            reload_check(
                page,
                evidence,
                job_id,
                result[
                    "final"
                ],
                timeout_ms,
            )
        )

        result[
            "reload_snapshot"
        ] = after

        result[
            "reload_persistence"
        ] = "PASS"

        if (
            result[
                "final"
            ][
                "status"
            ]
            in TERMINAL
            and page.get_by_role(
                "button",
                name="Approve action",
                exact=True,
            ).count()
        ):
            raise AssertionError(
                "terminal job restored an approval button"
            )

    # Product rendering must still exist after the durable result.
    if not page.locator(
        ".conversation"
    ).is_visible():
        raise AssertionError(
            "frontend conversation disappeared"
        )

    result[
        "frontend_rendered"
    ] = True

    socket_items = [
        item
        for item
        in evidence.websockets
        if item[
            "case_id"
        ]
        == case_id
    ]

    result[
        "websocket_observed"
    ] = any(
        "/socket/websocket"
        in item[
            "url"
        ]
        for item
        in socket_items
    )

    result[
        "socket_token_200"
    ] = any(
        item[
            "case_id"
        ]
        == case_id
        and item[
            "status"
        ]
        == 200
        for item
        in evidence.socket_tokens
    )

    for item in socket_items:
        if re.search(
            r"(token|csrf|password|session)=",
            item[
                "url"
            ],
            re.I,
        ):
            raise AssertionError(
                "credential-like query parameter appeared in WebSocket URL"
            )

    result[
        "websocket_url_secret_check"
    ] = "PASS"

    result[
        "status"
    ] = "PASS"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive MVP acceptance through the real Agentic frontend."
        )
    )

    parser.add_argument(
        "--frontend-url",
        default=os.environ.get(
            "FRONTEND_URL",
            "http://127.0.0.1:8080",
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--case",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--list",
        action="store_true",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Chromium so you can watch the frontend run itself.",
    )

    parser.add_argument(
        "--force-login",
        action="store_true",
        help="Clear browser cookies and prove a fresh frontend login.",
    )

    parser.add_argument(
        "--include-integration",
        action="append",
        default=[],
        choices=[
            "openwrt",
        ],
    )

    parser.add_argument(
        "--execute-disposable",
        action="store_true",
        help=(
            "Approve only the disposable workspace mkdir gate. "
            "All other approval cases remain reject-only."
        ),
    )

    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ACCEPTANCE_WORKSPACE_ROOT",
                "/mnt/c/project",
            )
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    args = parser.parse_args()

    cases = load_cases(
        args.manifest,
        args.case,
    )

    if args.list:
        print(
            "Frontend E2E cases"
        )

        print(
            "=================="
        )

        for item in cases:
            requirements = []

            if item.get(
                "requires_env"
            ):
                requirements.append(
                    "env="
                    + ",".join(
                        item[
                            "requires_env"
                        ]
                    )
                )

            if item.get(
                "requires_integration"
            ):
                requirements.append(
                    "integration="
                    + item[
                        "requires_integration"
                    ]
                )

            suffix = (
                " ["
                + "; ".join(
                    requirements
                )
                + "]"
                if requirements
                else ""
            )

            print(
                f"{item['id']:<36} "
                f"{item.get('kind', 'read'):<22}"
                f"{suffix}"
            )

        return 0

    run_token = (
        utc_token()
    )

    timeout_ms = int(
        args.timeout
        * 1000
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        args.report_dir
        / (
            run_token
            + ".json"
        )
    )

    screenshot_dir = (
        args.report_dir
        / run_token
    )

    screenshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: dict[
        str,
        Any,
    ] = {
        "schema":
            "frontend-e2e-acceptance.v1",
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "frontend_url":
            args.frontend_url,
        "manifest":
            str(
                args.manifest
            ),
        "run_token":
            run_token,
        "execute_disposable":
            args.execute_disposable,
        "included_integrations":
            sorted(
                set(
                    args.include_integration
                )
            ),
        "cases":
            [],
        "summary":
            {},
    }

    with sync_playwright() as playwright:
        browser = (
            playwright.chromium.launch(
                headless=not args.headed
            )
        )

        context = (
            browser.new_context()
        )

        page = (
            context.new_page()
        )

        evidence = (
            BrowserEvidence()
        )

        evidence.attach(
            page
        )

        try:
            report[
                "auth"
            ] = login(
                page,
                args.frontend_url,
                timeout_ms,
                force_login=args.force_login,
            )

            for case in cases:
                print(
                    f"[RUN] {case['id']}"
                )

                try:
                    item = run_case(
                        page,
                        evidence,
                        case,
                        run_token=run_token,
                        timeout_ms=timeout_ms,
                        execute_disposable=args.execute_disposable,
                        workspace_root=args.workspace_root,
                        integrations=set(
                            args.include_integration
                        ),
                    )

                except Exception as exc:
                    safe_id = re.sub(
                        r"[^A-Za-z0-9_.-]+",
                        "_",
                        case[
                            "id"
                        ],
                    )

                    screenshot = (
                        screenshot_dir
                        / (
                            safe_id
                            + ".png"
                        )
                    )

                    try:
                        page.screenshot(
                            path=str(
                                screenshot
                            ),
                            full_page=True,
                        )
                    except Exception:
                        pass

                    item = {
                        "id":
                            case[
                                "id"
                            ],
                        "kind":
                            case.get(
                                "kind"
                            ),
                        "status":
                            "FAIL",
                        "error_type":
                            type(
                                exc
                            ).__name__,
                        "error":
                            str(
                                exc
                            ),
                        "screenshot":
                            str(
                                screenshot
                            ),
                    }

                finally:
                    evidence.current_case = (
                        None
                    )

                report[
                    "cases"
                ].append(
                    item
                )

                detail = (
                    item.get(
                        "reason"
                    )
                    or item.get(
                        "error"
                    )
                    or ""
                )

                print(
                    f"[{item['status']}] "
                    f"{case['id']} "
                    f"{detail}"
                )

        finally:
            context.close()
            browser.close()

    passed = sum(
        1
        for item
        in report[
            "cases"
        ]
        if item[
            "status"
        ]
        == "PASS"
    )

    failed = sum(
        1
        for item
        in report[
            "cases"
        ]
        if item[
            "status"
        ]
        == "FAIL"
    )

    skipped = sum(
        1
        for item
        in report[
            "cases"
        ]
        if item[
            "status"
        ]
        == "SKIP"
    )

    socket_seen = any(
        item.get(
            "websocket_observed"
        )
        for item
        in report[
            "cases"
        ]
        if item[
            "status"
        ]
        == "PASS"
    )

    socket_token_seen = any(
        item.get(
            "socket_token_200"
        )
        for item
        in report[
            "cases"
        ]
        if item[
            "status"
        ]
        == "PASS"
    )

    report[
        "summary"
    ] = {
        "passed":
            passed,
        "failed":
            failed,
        "skipped":
            skipped,
        "websocket_observed":
            socket_seen,
        "socket_token_200_observed":
            socket_token_seen,
        "overall":
            "PASS"
            if failed
            == 0
            else "FAIL",
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Frontend E2E acceptance"
    )

    print(
        "======================="
    )

    print(
        f"PASS: {passed}"
    )

    print(
        f"FAIL: {failed}"
    )

    print(
        f"SKIP: {skipped}"
    )

    print(
        "WebSocket observed: "
        + (
            "yes"
            if socket_seen
            else "no"
        )
    )

    print(
        "socket-token 200 observed: "
        + (
            "yes"
            if socket_token_seen
            else "no"
        )
    )

    print(
        "report: "
        + str(
            report_path
        )
    )

    return (
        1
        if failed
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
