#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


AI_REPO = Path("/mnt/c/project/agenticaiPersonal/itsm-agent")
BACKEND_REPO = Path("/mnt/c/project/agenticBackend/itsm_backend")
FRONTEND_REPO = Path("/mnt/c/project/agenticFrontend")

DEFAULT_CASE_FILE = AI_REPO / "config/v18_03_acceptance_cases.json"
DEFAULT_BACKEND_URL = "http://127.0.0.1:4000"
TERMINAL = {
    "completed",
    "failed",
    "waiting_approval",
    "reconciliation_required",
}


class AcceptanceFailure(RuntimeError):
    pass


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(
    command: list[str],
    *,
    cwd: Path,
) -> None:
    # subprocess cwd does not trigger interactive direnv hooks. Keep the
    # backend configuration scoped to its process rather than the AI runtime.
    if cwd == BACKEND_REPO:
        command = ["direnv", "exec", str(BACKEND_REPO), *command]
    print("+", " ".join(command))
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def run_repo_gate() -> None:
    print("V18-03 repository gate")
    print("======================")

    _run(
        [
            "python",
            "-m",
            "compileall",
            "-q",
            "tools",
            "scripts",
            "tests/unit/tools",
        ],
        cwd=AI_REPO,
    )
    _run(
        [
            "pytest",
            "-q",
            "tests/unit/tools/test_capability_contract.py",
            "tests/unit/tools/test_explicit_semantics.py",
            "tests/unit/tools/test_full_registry_contract.py",
        ],
        cwd=AI_REPO,
    )
    _run(
        [
            "python",
            "scripts/audit_capability_contracts.py",
        ],
        cwd=AI_REPO,
    )
    _run(
        ["git", "diff", "--check"],
        cwd=AI_REPO,
    )

    _run(
        ["mix", "compile"],
        cwd=BACKEND_REPO,
    )
    _run(
        [
            "mix",
            "test",
            "test/itsm_backend_web/api_contract_test.exs",
        ],
        cwd=BACKEND_REPO,
    )
    _run(
        ["git", "diff", "--check"],
        cwd=BACKEND_REPO,
    )

    output_dir = (
        FRONTEND_REPO
        / ".runtime"
        / "v18_03"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    _run(
        [
            "nim",
            "js",
            "-d:release",
            f"-o:{output_dir / 'app.js'}",
            "src/agenticFrontend.nim",
        ],
        cwd=FRONTEND_REPO,
    )
    _run(
        ["git", "diff", "--check"],
        cwd=FRONTEND_REPO,
    )

    print()
    print("PASS: repository gate")


def _expand_prompt(
    template: str,
    *,
    run_token: str,
) -> str:
    value = template.replace(
        "{run_token}",
        run_token,
    )

    pattern = re.compile(
        r"\$\{([A-Z0-9_]+)\}"
    )

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        env_value = os.getenv(name)
        if env_value is None:
            raise AcceptanceFailure(
                f"Required environment value {name} is missing."
            )
        return env_value

    return pattern.sub(
        replace,
        value,
    )


def _job_payload(
    payload: Any,
) -> dict[str, Any]:
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("job"), dict)
    ):
        return payload["job"]

    if isinstance(payload, dict):
        return payload

    raise AcceptanceFailure(
        "Backend returned a non-object job response."
    )


def _tool_name(
    job: dict[str, Any],
) -> str | None:
    proposal = job.get(
        "proposed_tool"
    )

    if isinstance(proposal, str):
        return proposal

    if not isinstance(proposal, dict):
        return None

    for key in (
        "tool",
        "name",
        "tool_name",
    ):
        value = proposal.get(key)
        if isinstance(value, str) and value:
            return value

    return None


def _tool_arguments(
    job: dict[str, Any],
) -> dict[str, Any]:
    proposal = job.get(
        "proposed_tool"
    )

    if not isinstance(
        proposal,
        dict,
    ):
        return {}

    for key in (
        "arguments",
        "args",
    ):
        value = proposal.get(key)
        if isinstance(value, dict):
            return value

    return {}


class BrowserPathClient:
    def __init__(
        self,
        backend_url: str,
        *,
        timeout_seconds: float,
    ) -> None:
        self.client = httpx.Client(
            base_url=backend_url.rstrip("/"),
            follow_redirects=True,
            timeout=20.0,
        )
        self.timeout_seconds = timeout_seconds
        self.csrf_token: str | None = None

    def close(self) -> None:
        self.client.close()

    def preflight(self) -> None:
        response = self.client.get(
            "/api/health"
        )
        response.raise_for_status()

    def login(
        self,
        email: str,
        password: str,
    ) -> None:
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": email,
                "password": password,
            },
        )
        response.raise_for_status()
        payload = response.json()

        token = payload.get(
            "csrf_token"
        )
        if not isinstance(token, str) or not token:
            raise AcceptanceFailure(
                "Login response did not contain csrf_token."
            )

        self.csrf_token = token
        print("authenticated: yes")

    def _mutation_headers(self) -> dict[str, str]:
        if not self.csrf_token:
            raise AcceptanceFailure(
                "CSRF token is not available."
            )

        return {
            "X-CSRF-Token":
                self.csrf_token,
        }

    def create_chat(
        self,
        title: str,
    ) -> str:
        response = self.client.post(
            "/api/v1/chats",
            headers=self._mutation_headers(),
            json={"title": title},
        )
        response.raise_for_status()
        payload = response.json()
        chat = payload.get(
            "chat"
        )

        if not isinstance(chat, dict):
            raise AcceptanceFailure(
                "Chat creation response did not contain a chat."
            )

        chat_id = (
            chat.get("id")
            or chat.get("chat_id")
        )
        if not isinstance(chat_id, str) or not chat_id:
            raise AcceptanceFailure(
                "Created chat has no usable identifier."
            )

        return chat_id

    def integrations(
        self,
    ) -> dict[str, dict[str, Any]]:
        response = self.client.get(
            "/api/v1/integrations"
        )
        response.raise_for_status()
        payload = response.json()

        result: dict[
            str,
            dict[str, Any],
        ] = {}

        values = payload.get(
            "integrations",
            [],
        )
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, dict):
                    continue
                integration_id = item.get(
                    "id"
                )
                if (
                    isinstance(integration_id, str)
                    and integration_id
                ):
                    result[
                        integration_id
                    ] = item

        return result

    def create_job(
        self,
        *,
        chat_id: str,
        message: str,
    ) -> dict[str, Any]:
        response = self.client.post(
            "/api/v1/jobs",
            headers=self._mutation_headers(),
            json={
                "chat_id": chat_id,
                "message": message,
            },
        )
        if response.status_code != 202:
            raise AcceptanceFailure(
                "Job creation returned "
                f"HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        return _job_payload(
            response.json()
        )

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        response = self.client.get(
            f"/api/v1/jobs/{job_id}"
        )
        response.raise_for_status()
        return _job_payload(
            response.json()
        )

    def approve_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v1/jobs/{job_id}/approve",
            headers=self._mutation_headers(),
            json={},
        )
        response.raise_for_status()
        return _job_payload(
            response.json()
        )

    def delete_chat(
        self,
        chat_id: str,
    ) -> None:
        response = self.client.delete(
            f"/api/v1/chats/{chat_id}",
            headers=self._mutation_headers(),
        )
        if response.status_code not in {
            200,
            202,
            204,
        }:
            raise AcceptanceFailure(
                "Chat cleanup returned "
                f"HTTP {response.status_code}."
            )

    def wait_for_terminal(
        self,
        job_id: str,
        *,
        poll_seconds: float,
    ) -> dict[str, Any]:
        deadline = (
            time.monotonic()
            + self.timeout_seconds
        )

        last_status: str | None = None

        while (
            time.monotonic()
            < deadline
        ):
            job = self.get_job(
                job_id
            )
            status = job.get(
                "status"
            )

            if (
                isinstance(status, str)
                and status != last_status
            ):
                print(
                    "  status:",
                    status,
                )
                last_status = status

            if status in TERMINAL:
                return job

            time.sleep(
                poll_seconds
            )

        raise AcceptanceFailure(
            f"Timed out waiting for job {job_id}."
        )


def _skip_reason(
    case: dict[str, Any],
    integrations: dict[
        str,
        dict[str, Any],
    ],
) -> str | None:
    for name in case.get(
        "requires_env",
        [],
    ):
        if not os.getenv(name):
            return (
                f"missing environment value {name}"
            )

    integration_id = case.get(
        "requires_integration"
    )

    if isinstance(
        integration_id,
        str,
    ):
        info = integrations.get(
            integration_id
        )
        if (
            not isinstance(info, dict)
            or info.get("configured")
            is not True
        ):
            return (
                f"integration {integration_id} is not configured"
            )

    return None


def _assert_case(
    case: dict[str, Any],
    job: dict[str, Any],
) -> None:
    status = job.get(
        "status"
    )
    tool = _tool_name(
        job
    )

    expected_statuses = case.get(
        "expected_terminal_status"
    )
    if (
        isinstance(expected_statuses, list)
        and status not in expected_statuses
    ):
        raise AcceptanceFailure(
            f"Expected terminal status {expected_statuses}, got {status!r}."
        )

    expected_tool = case.get(
        "expected_tool"
    )
    if (
        isinstance(expected_tool, str)
        and tool != expected_tool
    ):
        raise AcceptanceFailure(
            f"Expected tool {expected_tool!r}, got {tool!r}."
        )

    if (
        case.get(
            "forbid_waiting_approval"
        )
        is True
        and status == "waiting_approval"
    ):
        raise AcceptanceFailure(
            "Fail-closed case reached waiting_approval."
        )

    forbidden_approval_tool = case.get(
        "forbid_approval_tool"
    )
    if (
        isinstance(
            forbidden_approval_tool,
            str,
        )
        and status == "waiting_approval"
        and tool == forbidden_approval_tool
    ):
        raise AcceptanceFailure(
            "Ambiguous request became an approvable mutation."
        )

    invented_rule = case.get(
        "forbid_invented_argument"
    )
    if isinstance(
        invented_rule,
        dict,
    ):
        target_tool = invented_rule.get(
            "tool"
        )
        argument = invented_rule.get(
            "argument"
        )

        if (
            tool == target_tool
            and isinstance(
                argument,
                str,
            )
        ):
            value = _tool_arguments(
                job
            ).get(
                argument
            )

            if (
                value is not None
                and str(value).strip()
            ):
                raise AcceptanceFailure(
                    "Fail-closed request acquired a value for "
                    f"missing grounded argument {argument!r}."
                )


def run_live(
    args: argparse.Namespace,
) -> None:
    email = (
        args.email
        or os.getenv(
            "V18_03_EMAIL"
        )
    )

    if not email:
        raise AcceptanceFailure(
            "Set V18_03_EMAIL or pass --email."
        )

    password = os.getenv(
        "V18_03_PASSWORD"
    )
    if password is None:
        password = getpass.getpass(
            "V18-03 login password: "
        )

    case_file = Path(
        args.case_file
    )
    config = json.loads(
        case_file.read_text(
            encoding="utf-8"
        )
    )

    cases = config.get(
        "cases"
    )
    if not isinstance(
        cases,
        list,
    ):
        raise AcceptanceFailure(
            "Case manifest does not contain cases."
        )

    selected = set(
        args.case
        or []
    )

    run_token = _now_token()
    report_dir = (
        AI_REPO
        / ".runtime"
        / "acceptance"
        / "v18-03"
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir
        / f"{run_token}.json"
    )

    results: list[
        dict[str, Any]
    ] = []

    client = BrowserPathClient(
        args.backend_url,
        timeout_seconds=args.timeout,
    )

    chat_id: str | None = None

    try:
        print("V18-03 live acceptance")
        print("======================")
        client.preflight()
        print("backend_health: reachable")

        client.login(
            email,
            password,
        )

        integrations = client.integrations()

        chat_id = client.create_chat(
            f"V18-03 acceptance {run_token}"
        )
        print(
            "acceptance_chat:",
            chat_id,
        )

        for case in cases:
            case_id = case.get(
                "id"
            )

            if (
                not isinstance(
                    case_id,
                    str,
                )
                or not case_id
            ):
                raise AcceptanceFailure(
                    "Case without valid id."
                )

            if (
                selected
                and case_id
                not in selected
            ):
                continue

            if (
                case.get(
                    "enabled",
                    True,
                )
                is not True
            ):
                continue

            reason = _skip_reason(
                case,
                integrations,
            )

            if reason:
                print(
                    f"[SKIP] {case_id}: {reason}"
                )
                results.append(
                    {
                        "id": case_id,
                        "outcome": "skipped",
                        "reason": reason,
                    }
                )
                continue

            print()
            print(
                f"[RUN] {case_id}"
            )

            prompt_template = case.get(
                "prompt"
            )
            if not isinstance(
                prompt_template,
                str,
            ):
                raise AcceptanceFailure(
                    f"{case_id}: missing prompt."
                )

            prompt = _expand_prompt(
                prompt_template,
                run_token=run_token,
            )

            created = client.create_job(
                chat_id=chat_id,
                message=prompt,
            )

            job_id = created.get(
                "job_id"
            )
            if not isinstance(
                job_id,
                str,
            ):
                raise AcceptanceFailure(
                    f"{case_id}: create response has no job_id."
                )

            print(
                "  job_id:",
                job_id,
            )

            terminal = (
                client.wait_for_terminal(
                    job_id,
                    poll_seconds=args.poll_interval,
                )
            )

            _assert_case(
                case,
                terminal,
            )

            if (
                terminal.get(
                    "status"
                )
                == "waiting_approval"
                and args.execute_approvals
            ):
                if (
                    case.get(
                        "allow_execute"
                    )
                    is not True
                ):
                    raise AcceptanceFailure(
                        f"{case_id}: --execute-approvals was supplied, "
                        "but this case is not explicitly allow_execute=true."
                    )

                print(
                    "  approving explicitly allowed case"
                )
                client.approve_job(
                    job_id
                )
                terminal = (
                    client.wait_for_terminal(
                        job_id,
                        poll_seconds=args.poll_interval,
                    )
                )

                post_approval_status = terminal.get(
                    "status"
                )

                expected_post_approval = case.get(
                    "expected_post_approval_status"
                )

                if isinstance(
                    expected_post_approval,
                    list,
                ):
                    if (
                        post_approval_status
                        not in expected_post_approval
                    ):
                        raise AcceptanceFailure(
                            f"{case_id}: expected post-approval "
                            f"status {expected_post_approval}, "
                            f"got {post_approval_status!r}."
                        )
                elif post_approval_status not in {
                    "completed",
                    "failed",
                    "reconciliation_required",
                }:
                    raise AcceptanceFailure(
                        f"{case_id}: approved job reached "
                        f"unexpected state "
                        f"{post_approval_status!r}."
                    )

            print(
                "  tool:",
                _tool_name(
                    terminal
                ),
            )
            print(
                f"[PASS] {case_id}"
            )

            results.append(
                {
                    "id": case_id,
                    "outcome": "passed",
                    "job_id": job_id,
                    "status": terminal.get(
                        "status"
                    ),
                    "selected_agent": terminal.get(
                        "selected_agent"
                    ),
                    "tool": _tool_name(
                        terminal
                    ),
                }
            )

    except Exception as exc:
        results.append(
            {
                "id": "__runner__",
                "outcome": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
        )
        raise

    finally:
        report = {
            "version": "V18-03",
            "run_token": run_token,
            "backend_url": args.backend_url,
            "results": results,
        }

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            "report:",
            report_path,
        )

        if (
            args.cleanup_chat
            and chat_id
            and client.csrf_token
        ):
            try:
                client.delete_chat(
                    chat_id
                )
                print(
                    "acceptance_chat_cleanup: requested"
                )
            except Exception as cleanup_exc:
                print(
                    "WARNING: chat cleanup failed:",
                    type(cleanup_exc).__name__,
                    file=sys.stderr,
                )

        client.close()

    failed = [
        item
        for item in results
        if item.get(
            "outcome"
        )
        == "failed"
    ]

    if failed:
        raise AcceptanceFailure(
            "One or more acceptance cases failed."
        )

    print()
    print("PASS: V18-03 live acceptance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SRS 1.8 V18-03 cross-repository and live "
            "browser-path acceptance harness."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "repo",
            "live",
            "all",
        ),
        default="repo",
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
    )
    parser.add_argument(
        "--email",
        default=None,
    )
    parser.add_argument(
        "--case-file",
        default=str(
            DEFAULT_CASE_FILE
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "Run only this case id. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--execute-approvals",
        action="store_true",
        help=(
            "Approve only cases explicitly marked allow_execute=true. "
            "The shipped manifest has no executable mutation case."
        ),
    )
    parser.add_argument(
        "--cleanup-chat",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {
        "repo",
        "all",
    }:
        run_repo_gate()

    if args.mode in {
        "live",
        "all",
    }:
        run_live(
            args
        )


if __name__ == "__main__":
    try:
        main()
    except (
        AcceptanceFailure,
        httpx.HTTPError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            "FAIL:",
            str(exc),
            file=sys.stderr,
        )
        raise SystemExit(1)
