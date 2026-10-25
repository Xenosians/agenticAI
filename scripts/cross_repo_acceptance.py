#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator, FormatChecker


AI_REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_REPO.parents[1]


def _acceptance_env(name: str, default: str | None = None) -> str | None:
    """Read one acceptance setting from the environment."""
    value = os.getenv(name)
    return default if value is None else value


BACKEND_REPO = Path(_acceptance_env(
    "ACCEPTANCE_BACKEND_REPO", str(PROJECT_ROOT / "agenticBackend/itsm_backend")
))
FRONTEND_REPO = Path(_acceptance_env(
    "ACCEPTANCE_FRONTEND_REPO", str(PROJECT_ROOT / "agenticFrontend")
))

DEFAULT_CASE_FILE = AI_REPO / "config/cross_repo_acceptance_cases.json"
DEFAULT_BACKEND_URL = "http://127.0.0.1:4000"
DEFAULT_JOB_API_VERSION = 2


class AcceptanceFailure(RuntimeError):
    pass


class PublicJobContracts:
    """Use the same canonical schemas as Phoenix, without copying them here."""

    def __init__(self, version: int, contracts_dir: Path) -> None:
        if version not in {1, 2}:
            raise ValueError("Job API version must be 1 or 2.")
        self.version = version
        self.create = self._load(contracts_dir / "v1/public/job-create-response.schema.json")
        self.job = self._load(contracts_dir / f"v{version}/public/job-response.schema.json")
        self.proposal = self._load(contracts_dir / "v1/common/tool-proposal.schema.json")
        statuses = set(self.job.schema["properties"]["status"]["enum"])
        self.terminal_statuses = statuses - {"pending", "processing"}

    @staticmethod
    def _load(path: Path) -> Draft202012Validator:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcceptanceFailure(f"Cannot read canonical public contract: {path}") from exc
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema, format_checker=FormatChecker())

    @staticmethod
    def validate(validator: Draft202012Validator, payload: Any, label: str) -> dict[str, Any]:
        error = next(validator.iter_errors(payload), None)
        if error is not None:
            # Do not include schema exception text: it embeds response content.
            location = ".".join(str(part) for part in error.absolute_path) or "$"
            raise AcceptanceFailure(
                f"Invalid {label} contract at {location} ({error.validator})."
            )
        return payload

    def validate_job(self, payload: Any, *, expected_id: str) -> dict[str, Any]:
        job = self.validate(self.job, _job_payload(payload), f"v{self.version} job")
        if job["job_id"] != expected_id:
            raise AcceptanceFailure("Backend returned a different job identifier.")
        if job["proposed_tool"] is not None:
            self.validate(self.proposal, job["proposed_tool"], "tool proposal")
        if job["status"] == "waiting_approval":
            proposal = job["proposed_tool"]
            if not isinstance(proposal, dict) or not proposal.get("approval_id"):
                raise AcceptanceFailure("Approval state has no persisted proposal identifier.")
        return job


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
    print("Cross-repository gate")
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
        / "cross-repo-acceptance"
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
        env_value = _acceptance_env(name)
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
        job_api_version: int = DEFAULT_JOB_API_VERSION,
        contracts_dir: Path = BACKEND_REPO / "contracts",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive.")
        self.contracts = PublicJobContracts(job_api_version, contracts_dir)
        self.job_api_version = job_api_version
        # One slow backend read must not end a multi-minute job wait; the
        # job deadline (timeout_seconds) bounds the overall run instead.
        self.client = httpx.Client(
            base_url=backend_url.rstrip("/"),
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=10.0),
            transport=transport,
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
            chat.get("chat_id")
            or chat.get("id")
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
                f"HTTP {response.status_code}."
            )

        return self.contracts.validate(
            self.contracts.create,
            _job_payload(response.json()),
            "job creation",
        )

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        response = self.client.get(
            f"/api/v{self.job_api_version}/jobs/{job_id}"
        )
        response.raise_for_status()
        return self.contracts.validate_job(
            response.json(), expected_id=job_id,
        )

    def approve_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v{self.job_api_version}/jobs/{job_id}/approve",
            headers=self._mutation_headers(),
            json={},
        )
        response.raise_for_status()
        return self.contracts.validate_job(
            response.json(), expected_id=job_id,
        )

    def reject_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        if self.job_api_version != 2:
            raise AcceptanceFailure(
                "Human rejection acceptance requires the v2 job contract."
            )
        response = self.client.post(
            f"/api/v2/jobs/{job_id}/reject",
            headers=self._mutation_headers(),
            json={},
        )
        response.raise_for_status()
        return self.contracts.validate_job(
            response.json(),
            expected_id=job_id,
        )

    def assert_approval_replay_rejected(self, job_id: str, *, status: str) -> None:
        response = self.client.post(
            f"/api/v{self.job_api_version}/jobs/{job_id}/approve",
            headers=self._mutation_headers(),
            json={},
        )
        if response.status_code != 409:
            raise AcceptanceFailure("Terminal approval replay was not rejected with HTTP 409.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise AcceptanceFailure("Approval replay returned a non-object rejection.") from exc
        if not isinstance(payload, dict) or payload.get("error") != "job_not_waiting_approval":
            raise AcceptanceFailure("Approval replay lacked a terminal-state rejection.")
        # Legacy job GET projects this storage state, while conflict responses
        # continue to expose the underlying terminal status.
        matching_statuses = {status}
        if self.job_api_version == 1 and status == "reconciliation_required":
            matching_statuses.add("outcome_unknown")
        if payload.get("status") not in matching_statuses:
            raise AcceptanceFailure("Approval replay reported a different terminal state.")

    def assert_rejection_replay_rejected(
        self,
        job_id: str,
        *,
        status: str,
    ) -> None:
        response = self.client.post(
            f"/api/v2/jobs/{job_id}/reject",
            headers=self._mutation_headers(),
            json={},
        )
        if response.status_code != 409:
            raise AcceptanceFailure(
                "Terminal rejection replay was not rejected with HTTP 409."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise AcceptanceFailure(
                "Rejection replay returned a non-object rejection."
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("error") != "job_not_waiting_approval"
        ):
            raise AcceptanceFailure(
                "Rejection replay lacked a terminal-state rejection."
            )
        if payload.get("status") != status:
            raise AcceptanceFailure(
                "Rejection replay reported a different terminal state."
            )

    def get_history(self, chat_id: str) -> dict[str, Any]:
        response = self.client.get(
            f"/api/v{self.job_api_version}/chats/{chat_id}/history",
            params={"limit": 200},
        )
        response.raise_for_status()
        payload = response.json()
        if (not isinstance(payload, dict) or payload.get("chat_id") != chat_id
                or not isinstance(payload.get("messages"), list)):
            raise AcceptanceFailure("Chat history did not preserve the canonical chat identifier.")
        return payload

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
        if poll_seconds <= 0:
            raise ValueError("Poll interval must be positive.")
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

            if status in self.contracts.terminal_statuses:
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
        if not _acceptance_env(name):
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
    *,
    job_api_version: int = DEFAULT_JOB_API_VERSION,
) -> None:
    status = job.get(
        "status"
    )
    tool = _tool_name(
        job
    )

    expected_statuses = case.get(
        f"expected_terminal_status_v{job_api_version}",
        case.get("expected_terminal_status"),
    )
    if (
        isinstance(expected_statuses, list)
        and status not in expected_statuses
    ):
        raise AcceptanceFailure(
            f"Expected terminal status {expected_statuses}, got {status!r}."
        )

    result = job.get("result")
    workers = result.get("results", []) if isinstance(result, dict) else []
    if not isinstance(workers, list) or any(not isinstance(worker, dict) for worker in workers):
        raise AcceptanceFailure("Job contains invalid worker evidence.")

    security_fail_closed = (
        case.get("expect_no_authority_delegation")
        is True
    )

    if security_fail_closed:
        if status != "failed":
            raise AcceptanceFailure(
                "Security fail-closed case did not terminate as failed."
            )

        if job.get("selected_agent") is not None:
            raise AcceptanceFailure(
                "Ambiguous mutation delegated authority to a specialist."
            )

        if job.get("proposed_tool") is not None or tool is not None:
            raise AcceptanceFailure(
                "Ambiguous mutation produced a tool proposal."
            )

        if result is not None:
            raise AcceptanceFailure(
                "Ambiguous mutation produced application execution results."
            )

        durable_error = job.get("error")
        if not isinstance(durable_error, str) or not durable_error.strip():
            raise AcceptanceFailure(
                "Security fail-closed case lacks a durable refusal reason."
            )

        expected_error = case.get("expected_error_contains")
        if (
            isinstance(expected_error, str)
            and expected_error
            and expected_error.lower() not in durable_error.lower()
        ):
            raise AcceptanceFailure(
                "Security fail-closed case returned an unexpected refusal reason."
            )

    if case.get("require_tool_result") is True:
        expected = case.get("expected_tool")
        evidence = [worker for worker in workers
                    if worker.get("proposed_tool") == expected
                    and worker.get("status") == "success"
                    and isinstance(worker.get("tool_result"), dict)]
        if not evidence:
            raise AcceptanceFailure("Completed read lacks successful structured tool evidence.")
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise AcceptanceFailure("Completed read lacks an application answer.")

    if case.get("forbid_tool_execution") is True:
        if any(worker.get("tool_result") is not None for worker in workers):
            raise AcceptanceFailure("Fail-closed case contains executed tool evidence.")

        if not security_fail_closed:
            if not isinstance(result, dict) or not isinstance(result.get("results"), list):
                raise AcceptanceFailure("Fail-closed case lacks structured resolution evidence.")
            answer = result.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise AcceptanceFailure("Fail-closed case lacks an application answer.")
            safe_workers = all(worker.get("status") in {"clarification_required", "denied"}
                               for worker in workers)
            no_route = not workers and result.get("routes") == []
            if not ((workers and safe_workers) or no_route):
                raise AcceptanceFailure("Fail-closed case lacks a clarification, denial, or empty plan.")

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


def _select_cases(config: Any, selected: set[str]) -> list[dict[str, Any]]:
    cases = config.get("cases") if isinstance(config, dict) else None
    if not isinstance(cases, list):
        raise AcceptanceFailure("Case manifest does not contain cases.")
    known: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise AcceptanceFailure("Case without a valid identifier.")
        identity = case["id"]
        if identity in known:
            raise AcceptanceFailure(f"Duplicate acceptance case identifier: {identity}.")
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise AcceptanceFailure(f"{identity}: missing prompt.")
        expected = case.get("expected_terminal_status")
        if not isinstance(expected, list) or not expected or not all(
            isinstance(value, str) for value in expected
        ):
            raise AcceptanceFailure(f"{identity}: explicit expected terminal states are required.")
        known[identity] = case
    unknown = selected - known.keys()
    if unknown:
        raise AcceptanceFailure("Unknown acceptance case identifiers: " + ", ".join(sorted(unknown)))
    chosen = [case for identity, case in known.items()
              if (not selected or identity in selected) and case.get("enabled", True) is True]
    if not chosen:
        raise AcceptanceFailure("No enabled acceptance cases were selected.")
    return chosen


def _assert_job_context(job: dict[str, Any], *, chat_id: str, message: str) -> None:
    if job.get("chat_id") != chat_id or job.get("conversation_id") != chat_id:
        raise AcceptanceFailure("Durable job is not bound to the canonical acceptance chat.")
    if job.get("message") != message:
        raise AcceptanceFailure("Durable job did not preserve the submitted request.")


def _assert_history_roundtrip(
    history: dict[str, Any], job: dict[str, Any], *, chat_id: str,
) -> None:
    _assert_job_context(job, chat_id=chat_id, message=job["message"])
    if history.get("chat_id") != chat_id or not isinstance(history.get("messages"), list):
        raise AcceptanceFailure("Chat history did not preserve the canonical chat identifier.")
    turns = [turn for turn in history["messages"]
             if isinstance(turn, dict) and turn.get("job_id") == job["job_id"]]
    users = [turn for turn in turns if turn.get("role") == "user"]
    assistants = [turn for turn in turns if turn.get("role") == "assistant"]
    if len(users) != 1 or users[0].get("content") != job["message"]:
        raise AcceptanceFailure("Canonical chat history lacks the exact submitted job turn.")
    result = job.get("result")
    answer = result.get("answer") if isinstance(result, dict) else None
    if job["status"] in {"failed", "denied", "outcome_unknown", "reconciliation_required"}:
        answer = job.get("error") or answer
    if (not isinstance(answer, str) or not answer.strip() or len(assistants) != 1
            or assistants[0].get("content") != answer):
        raise AcceptanceFailure("Canonical chat history lacks the visible durable job answer.")
    if any(turn.get("status") != job["status"] for turn in turns):
        raise AcceptanceFailure("Canonical chat history projects a different job state.")
    if assistants[0].get("proposed_tool") != job.get("proposed_tool"):
        raise AcceptanceFailure("Canonical chat history changed the persisted tool proposal.")
    if job.get("workflow_contract") is not None or job.get("workflow_contract_hash") is not None:
        from agent.requested_workflow import RequestedWorkflowContract, verify_workflow_binding
        try:
            contract = RequestedWorkflowContract.model_validate(job.get("workflow_contract"))
            verify_workflow_binding(contract, job.get("workflow_contract_hash"))
        except ValueError as exc:
            raise AcceptanceFailure("Durable job has an invalid reviewed workflow binding.") from exc
        if (users[0].get("workflow_contract") != job["workflow_contract"]
                or users[0].get("workflow_contract_hash") != job["workflow_contract_hash"]):
            raise AcceptanceFailure("Canonical chat history changed the reviewed workflow binding.")


def _assert_proposal_binding(original: dict[str, Any], resolved: dict[str, Any]) -> None:
    for key in ("job_id", "user_id", "chat_id", "conversation_id", "message", "attempts",
                "selected_agent", "proposed_tool"):
        if resolved.get(key) != original.get(key):
            raise AcceptanceFailure("Approved job changed its persisted request or proposal binding.")


def _assert_approval_execution(
    original: dict[str, Any], resolved: dict[str, Any], case: dict[str, Any],
) -> None:
    _assert_proposal_binding(original, resolved)
    expected = case.get("expected_post_approval_status", ["completed"])
    if (not isinstance(expected, list) or not expected or resolved.get("status") not in expected
            or resolved.get("status") == "waiting_approval"):
        raise AcceptanceFailure("Approved job did not reach its explicit terminal outcome.")
    proposal = original.get("proposed_tool")
    approval_id = proposal.get("approval_id") if isinstance(proposal, dict) else None
    result = resolved.get("result")
    outcome = result.get("approval_execution") if isinstance(result, dict) else None
    if not isinstance(approval_id, str) or not approval_id or not isinstance(outcome, dict):
        raise AcceptanceFailure("Approved job lacks persisted approval execution evidence.")
    if outcome.get("approval_id") != approval_id:
        raise AcceptanceFailure("Approval execution identifies a different persisted proposal.")
    states = {"succeeded": ("approved", "completed"), "failed": ("failed", "failed"),
              "unresolved": ("executing", "outcome_unknown")}
    pair = states.get(outcome.get("execution_status"))
    status = resolved.get("status")
    if status == "reconciliation_required":
        status = "outcome_unknown"
    if (outcome.get("protocol_status") != "processed" or pair is None
            or pair != (outcome.get("approval_status"), status)
            or not isinstance(outcome.get("replayed"), bool)):
        raise AcceptanceFailure("Approval execution has an inconsistent authority outcome.")
    evidence = outcome.get("result")
    error = outcome.get("error")
    if ((evidence is not None and not isinstance(evidence, dict))
            or (error is not None and not isinstance(error, str))):
        raise AcceptanceFailure("Approval execution has malformed structured evidence.")
    if status == "completed":
        if (not isinstance(evidence, dict) or evidence.get("ok") is not True
                or evidence.get("status") not in {"success", "executed"} or error):
            raise AcceptanceFailure("Completed approval lacks a successful structured tool result.")
        if not isinstance(result.get("answer"), str) or not result["answer"].strip():
            raise AcceptanceFailure("Completed approval lacks a visible durable answer.")


def _assert_rejection_execution(
    original: dict[str, Any],
    resolved: dict[str, Any],
    case: dict[str, Any],
) -> None:
    _assert_proposal_binding(original, resolved)
    expected = case.get("expected_post_rejection_status", ["denied"])
    if (
        not isinstance(expected, list)
        or not expected
        or resolved.get("status") not in expected
    ):
        raise AcceptanceFailure(
            "Rejected job did not reach its explicit terminal outcome."
        )

    proposal = original.get("proposed_tool")
    approval_id = proposal.get("approval_id") if isinstance(proposal, dict) else None
    result = resolved.get("result")
    outcome = result.get("approval_execution") if isinstance(result, dict) else None

    if not isinstance(approval_id, str) or not approval_id or not isinstance(outcome, dict):
        raise AcceptanceFailure(
            "Rejected job lacks persisted approval decision evidence."
        )
    if (
        outcome.get("approval_id") != approval_id
        or outcome.get("protocol_status") != "processed"
        or outcome.get("execution_status") != "rejected"
        or outcome.get("approval_status") != "rejected"
        or not isinstance(outcome.get("replayed"), bool)
    ):
        raise AcceptanceFailure(
            "Rejected job has an inconsistent authority outcome."
        )
    if outcome.get("result") is not None:
        raise AcceptanceFailure(
            "Human rejection unexpectedly contains tool execution evidence."
        )
    if not isinstance(outcome.get("error"), str) or not outcome["error"].strip():
        raise AcceptanceFailure(
            "Human rejection lacks a durable explanation."
        )

    workers = result.get("results", []) if isinstance(result, dict) else []
    if (
        not isinstance(workers, list)
        or any(
            not isinstance(worker, dict) or worker.get("tool_result") is not None
            for worker in workers
        )
    ):
        raise AcceptanceFailure(
            "Human rejection contains executed tool evidence."
        )
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise AcceptanceFailure(
            "Human rejection lacks a visible durable answer."
        )
    if not isinstance(resolved.get("error"), str) or not resolved["error"].strip():
        raise AcceptanceFailure(
            "Denied job lacks its durable rejection reason."
        )


def _execute_rejection_case(
    client: BrowserPathClient,
    case: dict[str, Any],
    original: dict[str, Any],
    *,
    poll_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case.get("allow_reject") is not True:
        raise AcceptanceFailure(
            "Automatic rejection requires explicit allow_reject=true."
        )
    if client.job_api_version != 2:
        raise AcceptanceFailure(
            "Human rejection acceptance requires --job-api-version 2."
        )

    original = deepcopy(original)
    job_id = original["job_id"]

    response = client.reject_job(job_id)
    _assert_proposal_binding(original, response)

    resolved = client.wait_for_terminal(job_id, poll_seconds=poll_seconds)
    _assert_rejection_execution(original, resolved, case)

    reopened = client.get_job(job_id)
    _assert_rejection_execution(original, reopened, case)
    if reopened != resolved:
        raise AcceptanceFailure(
            "Rejected terminal job changed when reopened."
        )

    client.assert_approval_replay_rejected(
        job_id,
        status=reopened["status"],
    )
    client.assert_rejection_replay_rejected(
        job_id,
        status=reopened["status"],
    )

    after_replay = client.get_job(job_id)
    if after_replay != reopened:
        raise AcceptanceFailure(
            "Rejected decision replay changed the durable terminal job."
        )

    proposal = original["proposed_tool"]
    return after_replay, {
        "decision": "reject",
        "original_job_id": job_id,
        "approval_id": proposal["approval_id"],
        "proposal_sha256": hashlib.sha256(
            json.dumps(
                proposal,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "proposal_binding_preserved": True,
        "no_tool_execution_verified": True,
        "terminal_reopen_unchanged": True,
        "approve_replay_http_status": 409,
        "reject_replay_http_status": 409,
        "terminal_after_replay_unchanged": True,
    }


def _execute_approval_case(
    client: BrowserPathClient, case: dict[str, Any], original: dict[str, Any], *, poll_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case.get("allow_execute") is not True:
        raise AcceptanceFailure("Approval execution requires an explicit allow_execute=true case.")
    original = deepcopy(original)
    job_id = original["job_id"]
    response = client.approve_job(job_id)
    _assert_proposal_binding(original, response)
    resolved = client.wait_for_terminal(job_id, poll_seconds=poll_seconds)
    _assert_approval_execution(original, resolved, case)
    reopened = client.get_job(job_id)
    _assert_approval_execution(original, reopened, case)
    if reopened != resolved:
        raise AcceptanceFailure("Approved terminal job changed when reopened.")
    client.assert_approval_replay_rejected(job_id, status=reopened["status"])
    after_replay = client.get_job(job_id)
    if after_replay != reopened:
        raise AcceptanceFailure("Rejected approval replay changed the durable terminal job.")
    proposal = original["proposed_tool"]
    return after_replay, {
        "original_job_id": job_id,
        "approval_id": proposal["approval_id"],
        "proposal_sha256": hashlib.sha256(json.dumps(
            proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
        "execution_status": reopened["result"]["approval_execution"]["execution_status"],
        "structured_tool_success_verified": reopened["status"] == "completed",
        "proposal_binding_preserved": True,
        "terminal_reopen_unchanged": True,
        "replay_http_status": 409,
        "terminal_after_replay_unchanged": True,
        "external_execution_count_verified": False,
        "model_not_rerun_verified": False,
    }


def run_live(
    args: argparse.Namespace,
) -> None:
    email = (
        args.email
        or _acceptance_env(
            "ACCEPTANCE_EMAIL"
        )
    )

    if not email:
        raise AcceptanceFailure(
            "Set ACCEPTANCE_EMAIL or pass --email."
        )

    password = _acceptance_env(
        "ACCEPTANCE_PASSWORD"
    )
    if password is None:
        if args.non_interactive:
            raise AcceptanceFailure("Set ACCEPTANCE_PASSWORD for a non-interactive live run.")
        password = getpass.getpass(
            "Acceptance login password: "
        )

    case_file = Path(
        args.case_file
    )
    config = json.loads(
        case_file.read_text(
            encoding="utf-8"
        )
    )

    selected = set(
        args.case
        or []
    )
    cases = _select_cases(config, selected)

    run_token = _now_token()
    report_dir = (
        AI_REPO
        / ".runtime"
        / "acceptance"
        / "cross-repo"
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
        job_api_version=args.job_api_version,
        contracts_dir=Path(args.contracts_dir),
    )

    chat_id: str | None = None

    try:
        print("Cross-repository live acceptance")
        print("======================")
        client.preflight()
        print("backend_health: reachable")

        client.login(
            email,
            password,
        )

        integrations = client.integrations()

        chat_id = client.create_chat(
            f"ITSM acceptance {run_token}"
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

            _assert_job_context(terminal, chat_id=chat_id, message=prompt)

            try:
                _assert_case(
                    case,
                    terminal,
                    job_api_version=args.job_api_version,
                )
            except Exception as case_exc:
                diagnostic = {
                    "id": case_id,
                    "outcome": "failed",
                    "job_id": job_id,
                    "status": terminal.get("status"),
                    "selected_agent": terminal.get("selected_agent"),
                    "tool": _tool_name(terminal),
                    "durable_error": terminal.get("error"),
                    "error_type": type(case_exc).__name__,
                    "error": str(case_exc)[:500],
                }
                results.append(diagnostic)

                print(f"[FAIL] {case_id}")
                print("  job_id:", job_id)
                print("  status:", terminal.get("status"))
                print("  selected_agent:", terminal.get("selected_agent"))
                print("  tool:", _tool_name(terminal))
                if terminal.get("error"):
                    print("  durable_error:", terminal.get("error"))

                raise

            execution_checks = None
            if (
                terminal.get("status") == "waiting_approval"
                and case.get("kind") == "rejection_gate"
            ):
                print("  rejecting explicitly safe acceptance proposal")
                terminal, execution_checks = _execute_rejection_case(
                    client,
                    case,
                    terminal,
                    poll_seconds=args.poll_interval,
                )
            elif (
                terminal.get("status") == "waiting_approval"
                and args.execute_approvals
            ):
                if case.get("allow_execute") is not True:
                    raise AcceptanceFailure(
                        f"{case_id}: --execute-approvals was supplied, "
                        "but this case is not explicitly allow_execute=true."
                    )
                print("  approving explicitly allowed case")
                terminal, execution_checks = _execute_approval_case(
                    client, case, terminal, poll_seconds=args.poll_interval,
                )

            _assert_history_roundtrip(client.get_history(chat_id), terminal, chat_id=chat_id)

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
                    "canonical_history_roundtrip": True,
                    "approval_execution_checks": execution_checks,
                }
            )

    except Exception as exc:
        # Case-level assertion failures already carry their bounded durable
        # diagnostic. Reserve __runner__ for failures outside a concrete case.
        if not results or results[-1].get("outcome") != "failed":
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
            "suite": "cross-repository-acceptance",
            "schema": "durable-acceptance.v2",
            "scope": "Authenticated Phoenix HTTP API; excludes browser rendering and exhaustive tool coverage",
            "job_api_version": args.job_api_version,
            "run_token": run_token,
            "backend_url": args.backend_url,
            "selected_case_count": len(cases),
            "complete": len(results) == len(cases) and all(
                item.get("outcome") == "passed" for item in results
            ),
            "coverage_limits": config.get("coverage_limits", []),
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

    if any(item.get("outcome") == "skipped" for item in results):
        raise AcceptanceFailure("Live acceptance is incomplete: selected cases were skipped; see the report.")

    print()
    print("PASS: cross-repository live acceptance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-repository and live "
            "authenticated durable HTTP acceptance harness."
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
        "--job-api-version", type=int, choices=(1, 2), default=DEFAULT_JOB_API_VERSION,
        help="Job reads/approval contract version (default: 2). Creation remains POST /api/v1/jobs.",
    )
    parser.add_argument(
        "--contracts-dir", type=Path, default=BACKEND_REPO / "contracts",
        help="Canonical contracts directory from the current backend checkout.",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Fail if the configured test account password is unavailable; never prompt.",
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
