from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class DomainAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_domain: str
    domains: list[str] = Field(default_factory=list)
    cross_domain: bool = False


def classify_trajectory_domains(trajectory) -> DomainAssessment:
    values: list[str] = []
    values.extend(str(item) for item in getattr(trajectory, "routes", []) or [])

    for step in getattr(trajectory, "steps", []) or []:
        for value in (
            getattr(step, "agent", None),
            getattr(step, "proposed_tool", None),
        ):
            if value:
                values.append(str(value))

    text = " ".join(values).casefold()
    domains: list[str] = []

    rules = [
        ("ticketing", ("ticket", "jira")),
        ("git", ("git", "repository", "branch", "commit")),
        ("developer", ("developer", "process_exec", "workspace", "shell", "build", "test")),
        ("account", ("account", "unlock_user", "reset_password", "enable_user", "disable_user")),
        ("access", ("access", "grant_access", "revoke_access", "check_access")),
        ("assets", ("asset",)),
        ("knowledge", ("knowledge", "runbook")),
    ]

    for domain, markers in rules:
        if any(marker in text for marker in markers):
            domains.append(domain)

    if not domains:
        domains = ["other"]

    # Preserve deterministic rule order while removing duplicates.
    unique = list(dict.fromkeys(domains))
    return DomainAssessment(
        primary_domain="cross_domain" if len(unique) > 1 else unique[0],
        domains=unique,
        cross_domain=len(unique) > 1,
    )


def count_primary_domains(assessments: list[DomainAssessment]) -> dict[str, int]:
    return dict(Counter(item.primary_domain for item in assessments))
