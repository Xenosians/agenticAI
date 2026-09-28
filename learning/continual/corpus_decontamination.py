from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from learning.continual.storage import (
    canonical_json,
)

from learning.paths import (
    REPOSITORY_ROOT,
)


DEFAULT_DECONTAMINATION_PATH = (
    REPOSITORY_ROOT
    / "config"
    / "continual_corpus_decontamination.json"
)


class CorpusDecontaminationPolicy(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    source_id: str

    # Deterministic source-local holdout.
    #
    # Example:
    #   modulus=10, buckets=[0]
    #
    # reserves about 10% of stable identities for held-out evaluation.
    holdout_modulus: int = Field(
        default=10,
        ge=2,
    )

    holdout_buckets: list[int] = Field(
        default_factory=lambda: [
            0
        ]
    )

    blocked_instance_ids: list[str] = Field(
        default_factory=list
    )

    blocked_repositories: list[str] = Field(
        default_factory=list
    )

    blocked_repository_commits: list[str] = Field(
        default_factory=list
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    @model_validator(
        mode="after"
    )
    def validate_buckets(
        self,
    ) -> "CorpusDecontaminationPolicy":

        if len(
            self.holdout_buckets
        ) != len(
            set(
                self.holdout_buckets
            )
        ):
            raise ValueError(
                "holdout_buckets must be unique."
            )

        for bucket in self.holdout_buckets:

            if not (
                0
                <= bucket
                < self.holdout_modulus
            ):
                raise ValueError(
                    "holdout bucket must be within modulus."
                )

        return self


class CorpusDecontaminationManifest(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "continual-corpus-decontamination.v1"
        ),
        alias="schema",
    )

    policies: list[
        CorpusDecontaminationPolicy
    ] = Field(
        default_factory=list
    )

    @model_validator(
        mode="after"
    )
    def validate_unique_sources(
        self,
    ) -> "CorpusDecontaminationManifest":

        ids = [
            policy.source_id
            for policy
            in self.policies
        ]

        if len(
            ids
        ) != len(
            set(
                ids
            )
        ):
            raise ValueError(
                "Decontamination source_id values must be unique."
            )

        return self

    def policy_for(
        self,
        source_id: str,
    ) -> CorpusDecontaminationPolicy | None:

        for policy in self.policies:

            if (
                policy.source_id
                == source_id
            ):
                return policy

        return None


def load_decontamination_manifest(
    path: Path = (
        DEFAULT_DECONTAMINATION_PATH
    ),
) -> CorpusDecontaminationManifest:

    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        return (
            CorpusDecontaminationManifest()
        )

    return (
        CorpusDecontaminationManifest
        .model_validate_json(
            path.read_text(
                encoding="utf-8"
            )
        )
    )


def policy_sha256(
    policy: CorpusDecontaminationPolicy,
) -> str:

    return hashlib.sha256(
        canonical_json(
            policy.model_dump(
                mode="json"
            )
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def stable_identity(
    *,
    source_record_id: str | None,
    repository: str | None,
    base_commit: str | None,
    content_sha256: str,
) -> str:

    if (
        source_record_id
        and source_record_id.strip()
    ):
        return (
            "instance:"
            + source_record_id.strip()
        )

    if (
        repository
        and repository.strip()
        and base_commit
        and base_commit.strip()
    ):
        return (
            "repo-commit:"
            + repository.strip().lower()
            + "@"
            + base_commit.strip().lower()
        )

    return (
        "content:"
        + content_sha256
    )


def decontamination_reason(
    *,
    policy: CorpusDecontaminationPolicy,
    source_record_id: str | None,
    repository: str | None,
    base_commit: str | None,
    content_sha256: str,
) -> str | None:

    instance_id = (
        source_record_id.strip()
        if (
            source_record_id
            and source_record_id.strip()
        )
        else None
    )

    repo = (
        repository.strip().lower()
        if (
            repository
            and repository.strip()
        )
        else None
    )

    commit = (
        base_commit.strip().lower()
        if (
            base_commit
            and base_commit.strip()
        )
        else None
    )

    blocked_instances = {
        value.strip()
        for value
        in policy.blocked_instance_ids
        if value.strip()
    }

    if (
        instance_id
        and instance_id
        in blocked_instances
    ):
        return (
            "benchmark_instance_blocked"
        )

    blocked_repositories = {
        value.strip().lower()
        for value
        in policy.blocked_repositories
        if value.strip()
    }

    if (
        repo
        and repo
        in blocked_repositories
    ):
        return (
            "benchmark_repository_blocked"
        )

    blocked_commits = {
        value.strip().lower()
        for value
        in policy.blocked_repository_commits
        if value.strip()
    }

    if (
        repo
        and commit
        and (
            repo
            + "@"
            + commit
        )
        in blocked_commits
    ):
        return (
            "benchmark_repository_commit_blocked"
        )

    identity = stable_identity(
        source_record_id=(
            source_record_id
        ),
        repository=repository,
        base_commit=base_commit,
        content_sha256=(
            content_sha256
        ),
    )

    digest = hashlib.sha256(
        identity.encode(
            "utf-8"
        )
    ).digest()

    bucket = (
        int.from_bytes(
            digest[
                :8
            ],
            "big",
        )
        % policy.holdout_modulus
    )

    if (
        bucket
        in policy.holdout_buckets
    ):
        return (
            "reserved_developer_holdout"
        )

    return None
