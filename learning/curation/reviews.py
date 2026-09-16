from __future__ import annotations

import json
import os
import threading
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.evidence.sanitizer import (
    sanitize_value,
)


VALID_REVIEW_SUBJECT_TYPES = {
    "trajectory",
    "correction",
}


VALID_REVIEW_DECISIONS = {
    "approve",
    "reject",
}


VALID_REVIEW_SOURCES = {
    "trusted_review",
    "evaluation",
}


class ReviewDecision(
    BaseModel
):
    """
    Immutable trusted decision about one raw learning artifact.

    Raw trajectories and corrections remain untouched.

    The latest append-only decision for a subject determines its
    current review state.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "review-decision.v1"
        ),
        alias="schema",
    )

    review_id: str

    observed_at: str

    subject_type: str

    subject_id: str

    decision: str

    source: str

    reason: str


class ReviewDecisionRecorder:
    """
    Append-only trusted review ledger.

    Runtime evidence never promotes itself. Eligibility is granted
    by a separate trusted review decision.
    """

    def __init__(
        self,
        *,
        path: Path,
        enabled: bool = True,
    ) -> None:

        self.path = (
            path
            .expanduser()
            .resolve()
        )

        self.enabled = (
            enabled
        )

        self._lock = (
            threading.Lock()
        )

    def build(
        self,
        *,
        subject_type: str,
        subject_id: str,
        decision: str,
        source: str,
        reason: str,
    ) -> ReviewDecision:

        normalized_subject_type = (
            subject_type
            .strip()
            .lower()
        )

        if (
            normalized_subject_type
            not in VALID_REVIEW_SUBJECT_TYPES
        ):

            raise ValueError(
                "Unsupported review subject_type: "
                f"{subject_type}"
            )

        normalized_subject_id = (
            subject_id
            .strip()
        )

        if not normalized_subject_id:

            raise ValueError(
                "subject_id must not be empty."
            )

        normalized_decision = (
            decision
            .strip()
            .lower()
        )

        if (
            normalized_decision
            not in VALID_REVIEW_DECISIONS
        ):

            raise ValueError(
                "Unsupported review decision: "
                f"{decision}"
            )

        normalized_source = (
            source
            .strip()
            .lower()
        )

        if (
            normalized_source
            not in VALID_REVIEW_SOURCES
        ):

            raise ValueError(
                "Unsupported review source: "
                f"{source}"
            )

        normalized_reason = (
            reason
            .strip()
        )

        if not normalized_reason:

            raise ValueError(
                "reason must not be empty."
            )

        return (
            ReviewDecision(
                review_id=(
                    uuid
                    .uuid4()
                    .hex
                ),

                observed_at=(
                    datetime
                    .now(
                        timezone.utc
                    )
                    .isoformat()
                ),

                subject_type=(
                    normalized_subject_type
                ),

                subject_id=(
                    normalized_subject_id
                ),

                decision=(
                    normalized_decision
                ),

                source=(
                    normalized_source
                ),

                reason=(
                    normalized_reason
                ),
            )
        )

    def record(
        self,
        *,
        subject_type: str,
        subject_id: str,
        decision: str,
        source: str,
        reason: str,
    ) -> dict | None:

        if not self.enabled:

            return None

        review = (
            self.build(
                subject_type=(
                    subject_type
                ),

                subject_id=(
                    subject_id
                ),

                decision=(
                    decision
                ),

                source=(
                    source
                ),

                reason=(
                    reason
                ),
            )
        )

        payload = (
            sanitize_value(
                review.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        with self._lock:

            with self.path.open(
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(
                    serialized
                )

                handle.write(
                    "\n"
                )

                handle.flush()

                os.fsync(
                    handle.fileno()
                )

        return (
            payload
        )


def load_reviews(
    path: Path,
) -> list[
    ReviewDecision
]:

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.exists():

        return []

    if not resolved.is_file():

        raise ValueError(
            "Review ledger path is not a file: "
            f"{resolved}"
        )

    reviews: list[
        ReviewDecision
    ] = []

    seen_review_ids: set[
        str
    ] = set()

    for (
        line_number,
        line,
    ) in enumerate(
        resolved
        .read_text(
            encoding="utf-8"
        )
        .splitlines(),
        start=1,
    ):

        if not line.strip():

            continue

        try:

            raw = (
                json.loads(
                    line
                )
            )

        except Exception as exc:

            raise ValueError(
                "Invalid review JSON at "
                f"{resolved}:{line_number}: "
                f"{exc}"
            ) from exc

        try:

            review = (
                ReviewDecision
                .model_validate(
                    raw
                )
            )

        except Exception as exc:

            raise ValueError(
                "Invalid review decision at "
                f"{resolved}:{line_number}: "
                f"{exc}"
            ) from exc

        if (
            review.review_id
            in seen_review_ids
        ):

            raise ValueError(
                "Duplicate review_id in review ledger: "
                f"{review.review_id}"
            )

        seen_review_ids.add(
            review.review_id
        )

        if (
            review.subject_type
            not in VALID_REVIEW_SUBJECT_TYPES
        ):

            raise ValueError(
                "Unsupported review subject_type "
                f"at {resolved}:{line_number}: "
                f"{review.subject_type}"
            )

        if (
            review.decision
            not in VALID_REVIEW_DECISIONS
        ):

            raise ValueError(
                "Unsupported review decision "
                f"at {resolved}:{line_number}: "
                f"{review.decision}"
            )

        if (
            review.source
            not in VALID_REVIEW_SOURCES
        ):

            raise ValueError(
                "Untrusted review source "
                f"at {resolved}:{line_number}: "
                f"{review.source}"
            )

        reviews.append(
            review
        )

    return (
        reviews
    )


def latest_review_index(
    reviews: list[
        ReviewDecision
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    ReviewDecision,
]:
    """
    Last append-only decision wins for a subject.

    File order is authoritative because the ledger itself is
    append-only.
    """

    index: dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ] = {}

    for review in reviews:

        index[
            (
                review.subject_type,
                review.subject_id,
            )
        ] = (
            review
        )

    return (
        index
    )


def subject_is_approved(
    *,
    review_index: dict[
        tuple[
            str,
            str,
        ],
        ReviewDecision,
    ],
    subject_type: str,
    subject_id: str,
) -> bool:

    review = (
        review_index.get(
            (
                subject_type,
                subject_id,
            )
        )
    )

    return (
        review is not None
        and review.decision
        == "approve"
    )