from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx

from config import Settings


TERMINAL = {"completed", "failed"}


def _post(client: httpx.Client, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.post(url, json=payload or {})
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Phoenix returned a non-object response.")
    return body


def _get(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Phoenix returned a non-object response.")
    return body


def create_job(client: httpx.Client, base_url: str, user_id: str, message: str) -> str:
    body = _post(
        client,
        f"{base_url}/api/v1/jobs",
        {"user_id": user_id, "message": message},
    )
    job_id = body.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RuntimeError("Phoenix did not return a durable job id.")
    return job_id


def wait_for_state(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    *,
    target: set[str],
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        body = _get(client, f"{base_url}/api/v1/jobs/{job_id}")
        status = body.get("status")
        if isinstance(status, str) and status in target:
            return body
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for durable job {job_id}.")
        time.sleep(poll_seconds)


def approve_job(client: httpx.Client, base_url: str, job_id: str) -> dict[str, Any]:
    return _post(client, f"{base_url}/api/v1/jobs/{job_id}/approve")


def _find_string(value: Any, key: str) -> str | None:
    if isinstance(value, dict):
        direct = value.get(key)
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        for child in value.values():
            found = _find_string(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_string(child, key)
            if found is not None:
                return found
    return None


def confirm(prompt: str) -> bool:
    reply = input(f"{prompt} [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive MVP proof: governed account creation -> encrypted "
            "SurrealDB persistence -> separately governed Jira ticket creation."
        )
    )
    parser.add_argument("--requester", required=True)
    parser.add_argument("--given-name", required=True)
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--department")
    parser.add_argument("--role")
    parser.add_argument("--jira-project", required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    settings = Settings()
    base_url = settings.require_phoenix_base_url().rstrip("/")

    identity = f"{args.given_name} {args.family_name}"
    details = []
    if args.department:
        details.append(f"department {args.department}")
    if args.role:
        details.append(f"role {args.role}")
    qualifier = " with " + " and ".join(details) if details else ""
    account_message = f"Create a corporate account for {identity}{qualifier}."

    timeout = settings.phoenix_http_timeout_seconds
    with httpx.Client(timeout=timeout) as client:
        account_job_id = create_job(client, base_url, args.requester, account_message)
        print(f"account job: {account_job_id}")

        account_job = wait_for_state(
            client,
            base_url,
            account_job_id,
            target={"waiting_approval", *TERMINAL},
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )

        if account_job.get("status") != "waiting_approval":
            print(f"account job stopped in status: {account_job.get('status')}")
            sys.exit(1)

        proposed = account_job.get("proposed_tool")
        print(f"account proposal: {proposed}")
        if not confirm("Approve this exact account creation?"):
            print("account creation not approved; Jira job was not created.")
            return

        approved = approve_job(client, base_url, account_job_id)
        if approved.get("status") != "completed":
            approved = wait_for_state(
                client,
                base_url,
                account_job_id,
                target=TERMINAL,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )

        if approved.get("status") != "completed":
            print(f"account creation ended in status: {approved.get('status')}")
            sys.exit(1)

        email = _find_string(approved.get("result"), "email")
        if email is None:
            print("account completed but no verified email was present in the safe result.")
            sys.exit(1)

        print(f"verified account email: {email}")
        print("temporary credential plaintext is not returned by this workflow.")

        jira_summary = f"Provision corporate access for {email}"
        jira_message = (
            f"Create Jira ticket in {args.jira_project} with summary {jira_summary}"
        )
        jira_job_id = create_job(client, base_url, args.requester, jira_message)
        print(f"jira job: {jira_job_id}")

        jira_job = wait_for_state(
            client,
            base_url,
            jira_job_id,
            target={"waiting_approval", *TERMINAL},
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )

        if jira_job.get("status") != "waiting_approval":
            print(f"jira job stopped in status: {jira_job.get('status')}")
            sys.exit(1)

        print(f"jira proposal: {jira_job.get('proposed_tool')}")
        if not confirm("Approve this exact Jira ticket creation?"):
            print("Jira ticket creation not approved.")
            return

        jira_approved = approve_job(client, base_url, jira_job_id)
        if jira_approved.get("status") != "completed":
            jira_approved = wait_for_state(
                client,
                base_url,
                jira_job_id,
                target=TERMINAL,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )

        print(f"jira job final status: {jira_approved.get('status')}")
        if jira_approved.get("status") != "completed":
            sys.exit(1)

        print("ACCOUNT -> SURREALDB -> JIRA DEMO: PASS")


if __name__ == "__main__":
    main()
