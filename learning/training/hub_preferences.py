from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from learning.continual.storage import (
    canonical_json,
    immutable_write_json,
    immutable_write_jsonl,
    load_jsonl_models,
    sha256_file,
)
from learning.curation.corpus_analysis import (
    load_eval_request_index,
    load_trajectories,
    normalize_request,
)
from learning.curation.reviews import (
    latest_review_index,
    load_reviews,
    subject_is_approved,
)
from learning.evidence.hub_routing import (
    DEFAULT_HUB_ROUTING_LEDGER,
    HubRoutingLedgerRecord,
)
from learning.evidence.sanitizer import sanitize_value
from learning.evidence.types import LearningTrajectory
from learning.paths import (
    EVALUATION_SUITE_ROOT,
    REVIEWS_PATH,
    RUNTIME_LEARNING_ROOT,
    TRAJECTORIES_PATH,
)


HUB_CORRECTIONS_PATH = (
    RUNTIME_LEARNING_ROOT
    / "hub-corrections.jsonl"
)

HUB_CORRECTION_REVIEWS_PATH = (
    RUNTIME_LEARNING_ROOT
    / "hub-correction-reviews.jsonl"
)

HUB_PREFERENCE_DATASET_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "datasets"
    / "hub-preference"
)

VALID_REVIEW_DECISIONS = {
    "approve",
    "reject",
}

VALID_REVIEW_SOURCES = {
    "trusted_review",
    "evaluation",
}


class HubCorrectionEvent(BaseModel):
    """
    Human-authored correction for one exact historical Hub routing attempt.

    The source attempt remains immutable in the Hub routing ledger. This event
    stores the reviewed replacement response separately and never rewrites the
    runtime evidence.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="hub-correction-event.v1",
        alias="schema",
    )

    correction_id: str
    observed_at: str

    trajectory_id: str
    attempt_id: str

    source: str = "explicit_user"
    note: str | None = None

    rejected_response_sha256: str
    rejected_output: dict[str, Any]

    chosen_response: str
    chosen_response_sha256: str
    chosen_response_exact: bool
    chosen_output: dict[str, Any]

    dataset_eligible: bool = False


class HubCorrectionReview(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="hub-correction-review.v1",
        alias="schema",
    )

    review_id: str
    observed_at: str

    correction_id: str
    decision: Literal["approve", "reject"]
    source: Literal["trusted_review", "evaluation"]
    reason: str


class HubPreferenceRecord(BaseModel):
    """
    Provenance-locked Hub DPO preference record.

    It is dataset-eligible only after both the source trajectory and the Hub
    correction have crossed explicit trusted review boundaries.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="hub-preference-record.v1",
        alias="schema",
    )

    record_id: str
    created_at: str

    trajectory_id: str
    correction_id: str
    source_attempt_id: str

    target_component: str = "hub"
    target_model_key: str

    user_request: str
    prompt_messages: list[dict[str, str]]

    rejected: str
    chosen: str

    failure_codes: list[str] = Field(
        default_factory=list
    )

    source_attempt_validation_status: str

    source_model_artifact_sha256: str
    source_model_weights_sha256: str
    source_tokenizer_artifact_sha256: str
    source_model_profile_sha256: str
    source_system_prompt_sha256: str
    source_capability_catalog_sha256: str
    source_messages_sha256: str

    source_attempt_sha256: str
    source_trajectory_sha256: str
    source_correction_sha256: str
    trajectory_review_id: str
    correction_review_id: str

    dataset_eligible: bool = True
    training_authorized: bool = False


class HubPreferenceManifest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default="hub-preference-manifest.v1",
        alias="schema",
    )

    dataset_id: str = "hub-preference"
    version: str
    created_at: str

    record_count: int
    content_sha256: str

    promoted_by: str
    promotion_reason: str

    source_trajectory_ids: list[str] = Field(
        default_factory=list
    )
    source_correction_ids: list[str] = Field(
        default_factory=list
    )
    source_attempt_ids: list[str] = Field(
        default_factory=list
    )

    exclusion_reason_counts: dict[str, int] = Field(
        default_factory=dict
    )

    training_authorized: bool = False
    promotion_authorized: bool = False


class HubPreferenceBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: HubPreferenceManifest
    output_directory: str


class HubCorrectionRecorder:
    def __init__(
        self,
        *,
        path: Path = HUB_CORRECTIONS_PATH,
        enabled: bool = True,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.enabled = enabled
        self._lock = threading.Lock()

    def build(
        self,
        *,
        trajectory_id: str,
        attempt: HubRoutingLedgerRecord,
        chosen_response: str,
        source: str = "explicit_user",
        note: str | None = None,
    ) -> HubCorrectionEvent:
        normalized_trajectory_id = trajectory_id.strip()

        if not normalized_trajectory_id:
            raise ValueError(
                "trajectory_id must not be empty."
            )

        if attempt.trajectory_id != normalized_trajectory_id:
            raise ValueError(
                "Hub routing attempt does not belong to the trajectory."
            )

        evidence = attempt.attempt

        if evidence.parsed_output is None:
            raise ValueError(
                "Hub routing attempt has no parsed output."
            )

        if not evidence.raw_response_exact:
            raise ValueError(
                "Hub routing response was changed by sanitization; "
                "it cannot seed an exact preference correction."
            )

        if not evidence.model_identity.complete:
            raise ValueError(
                "Hub routing attempt lacks complete model identity."
            )

        normalized_chosen = chosen_response.strip()

        if not normalized_chosen:
            raise ValueError(
                "chosen_response must not be empty."
            )

        try:
            chosen_output = json.loads(
                normalized_chosen
            )
        except Exception as exc:
            raise ValueError(
                f"chosen_response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(
            chosen_output,
            dict,
        ):
            raise ValueError(
                "chosen_response must decode to a JSON object."
            )

        if not isinstance(
            chosen_output.get(
                "delegations"
            ),
            list,
        ):
            raise ValueError(
                "chosen_response must contain a delegations list."
            )

        sanitized_chosen = sanitize_value(
            normalized_chosen
        )

        if not isinstance(
            sanitized_chosen,
            str,
        ):
            raise ValueError(
                "Sanitized chosen Hub response must remain text."
            )

        chosen_exact = (
            sanitized_chosen
            == normalized_chosen
        )

        if not chosen_exact:
            raise ValueError(
                "chosen_response contains content changed by the "
                "learning sanitizer; review a sanitized replacement instead."
            )

        sanitized_output = sanitize_value(
            chosen_output
        )

        if not isinstance(
            sanitized_output,
            dict,
        ):
            raise ValueError(
                "Sanitized chosen Hub output must remain an object."
            )

        if (
            sanitized_output
            == evidence.parsed_output
        ):
            raise ValueError(
                "Hub correction does not change the parsed routing output."
            )

        return HubCorrectionEvent(
            correction_id=(
                "hub-correction-"
                + uuid.uuid4().hex
            ),
            observed_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            trajectory_id=(
                normalized_trajectory_id
            ),
            attempt_id=(
                evidence.attempt_id
            ),
            source=(
                source.strip()
                or "explicit_user"
            ),
            note=note,
            rejected_response_sha256=(
                evidence.raw_response_sha256
            ),
            rejected_output=(
                evidence.parsed_output
            ),
            chosen_response=(
                sanitized_chosen
            ),
            chosen_response_sha256=(
                _sha256_text(
                    normalized_chosen
                )
            ),
            chosen_response_exact=True,
            chosen_output=(
                sanitized_output
            ),
            dataset_eligible=False,
        )

    def record(
        self,
        *,
        trajectory_id: str,
        attempt: HubRoutingLedgerRecord,
        chosen_response: str,
        source: str = "explicit_user",
        note: str | None = None,
    ) -> HubCorrectionEvent | None:
        if not self.enabled:
            return None

        correction = self.build(
            trajectory_id=trajectory_id,
            attempt=attempt,
            chosen_response=chosen_response,
            source=source,
            note=note,
        )

        _append_jsonl_model(
            path=self.path,
            value=correction,
            lock=self._lock,
        )

        return correction


class HubCorrectionReviewRecorder:
    def __init__(
        self,
        *,
        path: Path = HUB_CORRECTION_REVIEWS_PATH,
        enabled: bool = True,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.enabled = enabled
        self._lock = threading.Lock()

    def record(
        self,
        *,
        correction_id: str,
        decision: str,
        source: str,
        reason: str,
    ) -> HubCorrectionReview | None:
        if not self.enabled:
            return None

        normalized_correction_id = correction_id.strip()
        normalized_decision = decision.strip().lower()
        normalized_source = source.strip().lower()
        normalized_reason = reason.strip()

        if not normalized_correction_id:
            raise ValueError(
                "correction_id must not be empty."
            )

        if normalized_decision not in VALID_REVIEW_DECISIONS:
            raise ValueError(
                "decision must be approve or reject."
            )

        if normalized_source not in VALID_REVIEW_SOURCES:
            raise ValueError(
                "source must be trusted_review or evaluation."
            )

        if not normalized_reason:
            raise ValueError(
                "reason must not be empty."
            )

        review = HubCorrectionReview(
            review_id=(
                "hub-review-"
                + uuid.uuid4().hex
            ),
            observed_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            correction_id=(
                normalized_correction_id
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

        _append_jsonl_model(
            path=self.path,
            value=review,
            lock=self._lock,
        )

        return review


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _model_sha256(
    value: BaseModel,
) -> str:
    payload = value.model_dump(
        mode="json",
        by_alias=True,
    )

    return _sha256_text(
        canonical_json(
            payload
        )
    )


def _append_jsonl_model(
    *,
    path: Path,
    value: BaseModel,
    lock: threading.Lock,
) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = sanitize_value(
        value.model_dump(
            mode="json",
            by_alias=True,
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Serialized Hub learning artifact must remain an object."
        )

    line = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    with lock:
        with resolved.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                line
            )
            handle.write(
                "\n"
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )


def load_hub_corrections(
    path: Path = HUB_CORRECTIONS_PATH,
) -> list[HubCorrectionEvent]:
    return load_jsonl_models(
        path,
        HubCorrectionEvent,
    )


def load_hub_correction_reviews(
    path: Path = HUB_CORRECTION_REVIEWS_PATH,
) -> list[HubCorrectionReview]:
    return load_jsonl_models(
        path,
        HubCorrectionReview,
    )


def latest_hub_correction_review_index(
    reviews: list[HubCorrectionReview],
) -> dict[str, HubCorrectionReview]:
    index: dict[
        str,
        HubCorrectionReview,
    ] = {}

    for review in reviews:
        index[
            review.correction_id
        ] = review

    return index


def find_hub_routing_attempt(
    *,
    trajectory_id: str,
    attempt_id: str,
    path: Path = DEFAULT_HUB_ROUTING_LEDGER,
) -> HubRoutingLedgerRecord:
    matches = [
        record
        for record in load_jsonl_models(
            path,
            HubRoutingLedgerRecord,
        )
        if (
            record.trajectory_id
            == trajectory_id
            and record.attempt.attempt_id
            == attempt_id
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Hub routing attempt for "
            f"trajectory={trajectory_id!r} attempt={attempt_id!r}; "
            f"found {len(matches)}."
        )

    return matches[0]


def _failure_codes(
    trajectory: LearningTrajectory,
) -> list[str]:
    values: set[str] = set()

    for step in trajectory.steps:
        if (
            isinstance(
                step.outcome_code,
                str,
            )
            and step.outcome_code.strip()
        ):
            values.add(
                step.outcome_code.strip()
            )

        decision = step.semantic_guard_decision

        if isinstance(
            decision,
            dict,
        ):
            for key in (
                "decision_code",
                "code",
            ):
                candidate = decision.get(
                    key
                )

                if (
                    isinstance(
                        candidate,
                        str,
                    )
                    and candidate.strip()
                ):
                    values.add(
                        candidate.strip()
                    )

    return sorted(
        values
    )


def _next_version(
    root: Path,
) -> str:
    if not root.exists():
        return "v000001"

    highest = 0

    for child in root.iterdir():
        if not child.is_dir():
            continue

        name = child.name

        if (
            len(
                name
            ) == 7
            and name.startswith(
                "v"
            )
            and name[1:].isdigit()
        ):
            highest = max(
                highest,
                int(
                    name[1:]
                ),
            )

    return f"v{highest + 1:06d}"


def _validate_prompt_archive(
    record: HubRoutingLedgerRecord,
) -> list[dict[str, str]]:
    attempt = record.attempt

    messages = getattr(
        attempt,
        "messages",
        None,
    )

    messages_exact = getattr(
        attempt,
        "messages_exact",
        False,
    )

    if not messages_exact:
        raise ValueError(
            "Hub routing attempt has no exact sanitized prompt archive."
        )

    if not isinstance(
        messages,
        list,
    ) or not messages:
        raise ValueError(
            "Hub routing attempt has no prompt messages."
        )

    normalized: list[
        dict[str, str]
    ] = []

    for message in messages:
        if not isinstance(
            message,
            dict,
        ):
            raise ValueError(
                "Hub routing prompt message is malformed."
            )

        role = message.get(
            "role"
        )
        content = message.get(
            "content"
        )

        if not (
            isinstance(
                role,
                str,
            )
            and isinstance(
                content,
                str,
            )
        ):
            raise ValueError(
                "Hub routing prompt message must contain string role/content."
            )

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    archived_sha = _sha256_text(
        canonical_json(
            normalized
        )
    )

    if archived_sha != attempt.messages_sha256:
        raise ValueError(
            "Archived Hub routing messages do not match provenance hash."
        )

    return normalized


class HubPreferenceDatasetBuilder:
    def __init__(
        self,
        *,
        trajectory_path: Path = TRAJECTORIES_PATH,
        routing_path: Path = DEFAULT_HUB_ROUTING_LEDGER,
        correction_path: Path = HUB_CORRECTIONS_PATH,
        correction_review_path: Path = HUB_CORRECTION_REVIEWS_PATH,
        trajectory_review_path: Path = REVIEWS_PATH,
        dataset_root: Path = HUB_PREFERENCE_DATASET_ROOT,
    ) -> None:
        self.trajectory_path = trajectory_path.expanduser().resolve()
        self.routing_path = routing_path.expanduser().resolve()
        self.correction_path = correction_path.expanduser().resolve()
        self.correction_review_path = (
            correction_review_path.expanduser().resolve()
        )
        self.trajectory_review_path = (
            trajectory_review_path.expanduser().resolve()
        )
        self.dataset_root = dataset_root.expanduser().resolve()

    def build(
        self,
        *,
        eval_paths: list[Path] | None = None,
        promoted_by: str = "trusted_review",
        promotion_reason: str,
    ) -> HubPreferenceBuildResult:
        normalized_promoted_by = promoted_by.strip().lower()
        normalized_reason = promotion_reason.strip()

        if normalized_promoted_by not in VALID_REVIEW_SOURCES:
            raise ValueError(
                "promoted_by must be trusted_review or evaluation."
            )

        if not normalized_reason:
            raise ValueError(
                "promotion_reason must not be empty."
            )

        trajectories = load_trajectories(
            self.trajectory_path
        )
        trajectory_by_id = {
            item.trajectory_id: item
            for item in trajectories
        }

        attempts = load_jsonl_models(
            self.routing_path,
            HubRoutingLedgerRecord,
        )
        attempt_by_id = {
            item.attempt.attempt_id: item
            for item in attempts
        }

        corrections = load_hub_corrections(
            self.correction_path
        )

        correction_reviews = (
            latest_hub_correction_review_index(
                load_hub_correction_reviews(
                    self.correction_review_path
                )
            )
        )

        trajectory_reviews = load_reviews(
            self.trajectory_review_path
        )
        trajectory_review_index = latest_review_index(
            trajectory_reviews
        )

        eval_paths = (
            sorted(
                EVALUATION_SUITE_ROOT.glob(
                    "*.jsonl"
                )
            )
            if eval_paths is None
            else [
                Path(
                    path
                ).expanduser().resolve()
                for path in eval_paths
            ]
        )

        eval_index = (
            load_eval_request_index(
                eval_paths
            )
            if eval_paths
            else {}
        )

        records: list[
            HubPreferenceRecord
        ] = []

        exclusions: Counter[str] = Counter()

        for correction in corrections:
            correction_review = correction_reviews.get(
                correction.correction_id
            )

            if (
                correction_review is None
                or correction_review.decision
                != "approve"
            ):
                exclusions[
                    "hub_correction_review_not_approved"
                ] += 1
                continue

            trajectory = trajectory_by_id.get(
                correction.trajectory_id
            )

            if trajectory is None:
                exclusions[
                    "trajectory_missing"
                ] += 1
                continue

            if not subject_is_approved(
                review_index=trajectory_review_index,
                subject_type="trajectory",
                subject_id=trajectory.trajectory_id,
            ):
                exclusions[
                    "trajectory_review_not_approved"
                ] += 1
                continue

            trajectory_review = trajectory_review_index[
                (
                    "trajectory",
                    trajectory.trajectory_id,
                )
            ]

            if normalize_request(
                trajectory.user_request
            ) in eval_index:
                exclusions[
                    "held_out_contamination"
                ] += 1
                continue

            routing_record = attempt_by_id.get(
                correction.attempt_id
            )

            if routing_record is None:
                exclusions[
                    "routing_attempt_missing"
                ] += 1
                continue

            if (
                routing_record.trajectory_id
                != trajectory.trajectory_id
            ):
                exclusions[
                    "routing_trajectory_mismatch"
                ] += 1
                continue

            attempt = routing_record.attempt

            if not attempt.model_identity.complete:
                exclusions[
                    "hub_model_identity_incomplete"
                ] += 1
                continue

            if not attempt.raw_response_exact:
                exclusions[
                    "hub_raw_response_not_exact"
                ] += 1
                continue

            if not correction.chosen_response_exact:
                exclusions[
                    "hub_chosen_response_not_exact"
                ] += 1
                continue

            try:
                prompt_messages = _validate_prompt_archive(
                    routing_record
                )
            except ValueError as exc:
                exclusions[
                    str(
                        exc
                    )
                ] += 1
                continue

            if (
                _sha256_text(
                    trajectory.user_request
                )
                != attempt.user_request_sha256
            ):
                exclusions[
                    "hub_user_request_hash_mismatch"
                ] += 1
                continue

            if (
                _sha256_text(
                    attempt.raw_response
                )
                != attempt.raw_response_sha256
            ):
                exclusions[
                    "hub_rejected_response_hash_mismatch"
                ] += 1
                continue

            if (
                _sha256_text(
                    correction.chosen_response
                )
                != correction.chosen_response_sha256
            ):
                exclusions[
                    "hub_chosen_response_hash_mismatch"
                ] += 1
                continue

            if (
                correction.rejected_response_sha256
                != attempt.raw_response_sha256
            ):
                exclusions[
                    "hub_correction_rejected_hash_mismatch"
                ] += 1
                continue

            if (
                correction.rejected_output
                != attempt.parsed_output
            ):
                exclusions[
                    "hub_correction_rejected_output_mismatch"
                ] += 1
                continue

            if (
                correction.chosen_output
                == attempt.parsed_output
            ):
                exclusions[
                    "hub_correction_no_change"
                ] += 1
                continue

            identity = {
                "trajectory_id": trajectory.trajectory_id,
                "correction_id": correction.correction_id,
                "attempt_id": attempt.attempt_id,
                "messages_sha256": attempt.messages_sha256,
                "rejected_sha256": attempt.raw_response_sha256,
                "chosen_sha256": correction.chosen_response_sha256,
            }

            record_id = (
                "hub-pref-"
                + _sha256_text(
                    canonical_json(
                        identity
                    )
                )[:24]
            )

            model_identity = attempt.model_identity

            required_model_hashes = [
                model_identity.model_artifact_sha256,
                model_identity.model_weights_sha256,
                model_identity.tokenizer_artifact_sha256,
                model_identity.model_profile_sha256,
            ]

            if not all(
                required_model_hashes
            ):
                exclusions[
                    "hub_model_hashes_incomplete"
                ] += 1
                continue

            records.append(
                HubPreferenceRecord(
                    record_id=record_id,
                    created_at=(
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    trajectory_id=(
                        trajectory.trajectory_id
                    ),
                    correction_id=(
                        correction.correction_id
                    ),
                    source_attempt_id=(
                        attempt.attempt_id
                    ),
                    target_model_key=(
                        model_identity.model_key
                    ),
                    user_request=(
                        trajectory.user_request
                    ),
                    prompt_messages=(
                        prompt_messages
                    ),
                    rejected=(
                        attempt.raw_response
                    ),
                    chosen=(
                        correction.chosen_response
                    ),
                    failure_codes=(
                        _failure_codes(
                            trajectory
                        )
                    ),
                    source_attempt_validation_status=(
                        attempt.validation_status
                    ),
                    source_model_artifact_sha256=(
                        model_identity.model_artifact_sha256
                    ),
                    source_model_weights_sha256=(
                        model_identity.model_weights_sha256
                    ),
                    source_tokenizer_artifact_sha256=(
                        model_identity.tokenizer_artifact_sha256
                    ),
                    source_model_profile_sha256=(
                        model_identity.model_profile_sha256
                    ),
                    source_system_prompt_sha256=(
                        attempt.system_prompt_sha256
                    ),
                    source_capability_catalog_sha256=(
                        attempt.capability_catalog_sha256
                    ),
                    source_messages_sha256=(
                        attempt.messages_sha256
                    ),
                    source_attempt_sha256=(
                        _model_sha256(
                            attempt
                        )
                    ),
                    source_trajectory_sha256=(
                        _model_sha256(
                            trajectory
                        )
                    ),
                    source_correction_sha256=(
                        _model_sha256(
                            correction
                        )
                    ),
                    trajectory_review_id=(
                        trajectory_review.review_id
                    ),
                    correction_review_id=(
                        correction_review.review_id
                    ),
                    dataset_eligible=True,
                    training_authorized=False,
                )
            )

        if not records:
            detail = ", ".join(
                f"{key}={value}"
                for key, value in sorted(
                    exclusions.items()
                )
            )

            suffix = (
                f" Exclusions: {detail}"
                if detail
                else ""
            )

            raise ValueError(
                "No trusted reviewed Hub preference records are available."
                + suffix
            )

        seen_ids: set[str] = set()
        deduplicated: list[HubPreferenceRecord] = []

        for record in sorted(
            records,
            key=lambda item: item.record_id,
        ):
            if record.record_id in seen_ids:
                exclusions[
                    "duplicate_hub_preference_record"
                ] += 1
                continue

            seen_ids.add(
                record.record_id
            )
            deduplicated.append(
                record
            )

        version = _next_version(
            self.dataset_root
        )
        target = (
            self.dataset_root
            / version
        )

        target.mkdir(
            parents=True,
            exist_ok=False,
        )

        try:
            content_sha256 = immutable_write_jsonl(
                target
                / "records.jsonl",
                deduplicated,
            )

            manifest = HubPreferenceManifest(
                version=version,
                created_at=(
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                record_count=len(
                    deduplicated
                ),
                content_sha256=(
                    content_sha256
                ),
                promoted_by=(
                    normalized_promoted_by
                ),
                promotion_reason=(
                    normalized_reason
                ),
                source_trajectory_ids=sorted(
                    {
                        item.trajectory_id
                        for item in deduplicated
                    }
                ),
                source_correction_ids=sorted(
                    {
                        item.correction_id
                        for item in deduplicated
                    }
                ),
                source_attempt_ids=sorted(
                    {
                        item.source_attempt_id
                        for item in deduplicated
                    }
                ),
                exclusion_reason_counts=dict(
                    sorted(
                        exclusions.items()
                    )
                ),
                training_authorized=False,
                promotion_authorized=False,
            )

            immutable_write_json(
                target
                / "manifest.json",
                manifest,
            )

        except Exception:
            shutil.rmtree(
                target,
                ignore_errors=True,
            )
            raise

        return HubPreferenceBuildResult(
            manifest=manifest,
            output_directory=str(
                target
            ),
        )
