from __future__ import annotations
import subprocess
import sys

import hashlib
import json
import os
import re
import shutil
import tempfile

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import (
    Any,
    Iterator,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from learning.continual.corpus_decontamination import (
    DEFAULT_DECONTAMINATION_PATH,
    CorpusDecontaminationPolicy,
    decontamination_reason,
    load_decontamination_manifest,
    policy_sha256,
)

from learning.continual.corpus_registry import (
    CorpusRegistrySource,
    load_corpus_registry,
)

from learning.continual.storage import (
    canonical_json,
)

from learning.paths import (
    REPOSITORY_ROOT,
    RUNTIME_LEARNING_ROOT,
)


CORPUS_RUNTIME_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "corpus"
)

CORPUS_SNAPSHOT_ROOT = (
    CORPUS_RUNTIME_ROOT
    / "snapshots"
)

CORPUS_CURSOR_PATH = (
    CORPUS_RUNTIME_ROOT
    / "cursors.json"
)


NormalizedObjective = Literal[
    "retrieval",
    "sft",
    "dpo",
]


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _safe_name(
    value: str,
) -> str:
    return (
        re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            value,
        )
        .strip("-")
        or "source"
    )


def _jsonish_text(
    value: Any,
) -> str | None:

    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        return (
            value
            if value
            else None
        )

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
        ),
    ):
        return canonical_json(
            value
        )

    if isinstance(
        value,
        (
            int,
            float,
            bool,
        ),
    ):
        return str(
            value
        )

    return None


def _first_text(
    row: dict[str, Any],
    *keys: str,
) -> str | None:

    for key in keys:

        value = _jsonish_text(
            row.get(
                key
            )
        )

        if value:
            return value

    return None


def _first_raw(
    row: dict[str, Any],
    *keys: str,
) -> Any:

    for key in keys:

        if key not in row:
            continue

        value = row[
            key
        ]

        if value is not None:
            return value

    return None


def _source_record_id(
    row: dict[str, Any],
) -> str | None:

    return _first_text(
        row,
        "id",
        "instance_id",
        "issue_id",
        "ticket_id",
        "commit",
        "sha",
        "hash",
        "uuid",
        "record_id",
    )


def _language(
    row: dict[str, Any],
) -> str | None:

    return _first_text(
        row,
        "language",
        "lang",
        "programming_language",
    )


def _task(
    row: dict[str, Any],
) -> str | None:

    return _first_text(
        row,
        "task",
        "task_type",
        "category",
        "type",
        "intent",
    )


def _verified_outcome(
    row: dict[str, Any],
) -> bool:

    for key in (
        "verified",
        "success",
        "resolved",
        "tests_passed",
        "passed",
        "is_successful",
        "accepted",
    ):

        value = row.get(
            key
        )

        if value is True:
            return True

        if (
            isinstance(
                value,
                str,
            )
            and value.strip().lower()
            in {
                "true",
                "passed",
                "success",
                "resolved",
                "verified",
            }
        ):
            return True

    reward = row.get(
        "reward"
    )

    if (
        isinstance(
            reward,
            (
                int,
                float,
            ),
        )
        and reward > 0
    ):
        return True

    # SWE-style records may express verification through executable
    # test-transition data rather than one boolean.
    if any(
        row.get(
            key
        )
        for key in (
            "test_patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "fail_to_pass",
            "pass_to_pass",
        )
    ):
        return True

    return False


def _messages(
    value: Any,
) -> list[dict[str, str]]:

    if isinstance(
        value,
        str,
    ):
        try:
            value = json.loads(
                value
            )
        except Exception:
            return []

    if not isinstance(
        value,
        list,
    ):
        return []

    result: list[
        dict[str, str]
    ] = []

    for item in value:

        if not isinstance(
            item,
            dict,
        ):
            continue

        role = item.get(
            "role"
        )

        content = item.get(
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
            and content.strip()
        ):
            continue

        role = (
            role
            .strip()
            .lower()
        )

        if role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            continue

        result.append(
            {
                "role":
                    role,

                "content":
                    content.strip(),
            }
        )

    return result


def _prompt_messages(
    prompt: str,
    *,
    system: str | None = None,
) -> list[dict[str, str]]:

    result: list[
        dict[str, str]
    ] = []

    if system:

        result.append(
            {
                "role":
                    "system",

                "content":
                    system.strip(),
            }
        )

    result.append(
        {
            "role":
                "user",

            "content":
                prompt.strip(),
        }
    )

    return result


class NormalizedCorpusRecord(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "continual-normalized-corpus-record.v1"
        ),
        alias="schema",
    )

    record_id: str
    source_id: str

    source_record_id: str | None = None

    upstream_index: int = Field(
        ge=0
    )

    objective: NormalizedObjective

    target_component: str

    prompt_messages: list[
        dict[str, str]
    ] = Field(
        default_factory=list
    )

    chosen: str | None = None
    rejected: str | None = None
    text: str | None = None

    language: str | None = None
    task: str | None = None

    verified_outcome: bool = False

    content_sha256: str

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    @model_validator(
        mode="after"
    )
    def validate_objective(
        self,
    ) -> "NormalizedCorpusRecord":

        if (
            self.objective
            == "retrieval"
            and not self.text
        ):
            raise ValueError(
                "Retrieval records require text."
            )

        if (
            self.objective
            == "sft"
            and (
                not self.prompt_messages
                or not self.chosen
            )
        ):
            raise ValueError(
                "SFT records require prompt_messages and chosen."
            )

        if (
            self.objective
            == "dpo"
            and (
                not self.prompt_messages
                or not self.chosen
                or not self.rejected
                or (
                    self.chosen.strip()
                    == self.rejected.strip()
                )
            )
        ):
            raise ValueError(
                "DPO records require prompt/chosen/rejected."
            )

        return self


class CorpusSnapshotManifest(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "continual-corpus-snapshot.v1"
        ),
        alias="schema",
    )

    snapshot_id: str
    created_at: str

    source_id: str
    provider: str

    dataset_id: str | None = None
    subset: str | None = None
    revision: str | None = None
    split: str | None = None

    target_component: str
    objectives: list[str]

    source_config_sha256: str

    # None preserves compatibility with snapshots created before
    # decontamination policy hashing was introduced.
    decontamination_policy_sha256: str | None = None

    cursor_start: int
    cursor_end: int

    scanned_count: int
    accepted_count: int

    rejected_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    records_sha256: str

    training_eligible: bool = False

    training_blockers: list[
        str
    ] = Field(
        default_factory=list
    )

    output_directory: str


class CorpusMaterializationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    manifest: CorpusSnapshotManifest

    records_path: str
    manifest_path: str


class CorpusCursorStore:

    def __init__(
        self,
        path: Path = (
            CORPUS_CURSOR_PATH
        ),
    ) -> None:

        self.path = (
            path
            .expanduser()
            .resolve()
        )

    def _load(
        self,
    ) -> dict[str, Any]:

        if not self.path.is_file():

            return {
                "schema":
                    "continual-corpus-cursors.v1",

                "sources":
                    {},
            }

        raw = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            raw,
            dict,
        ):
            raise ValueError(
                "Corpus cursor store is malformed."
            )

        if not isinstance(
            raw.get(
                "sources"
            ),
            dict,
        ):
            raw[
                "sources"
            ] = {}

        return raw

    def get(
        self,
        source: CorpusRegistrySource,
    ) -> int:

        raw = self._load()

        item = (
            raw[
                "sources"
            ]
            .get(
                source.source_id
            )
        )

        if not isinstance(
            item,
            dict,
        ):
            return 0

        previous_revision = (
            item.get(
                "revision"
            )
        )

        current_revision = (
            source.revision
        )

        if (
            previous_revision
            != current_revision
        ):
            return 0

        return max(
            0,
            int(
                item.get(
                    "cursor",
                    0,
                )
            ),
        )

    def set(
        self,
        source: CorpusRegistrySource,
        cursor: int,
    ) -> None:

        raw = self._load()

        raw[
            "sources"
        ][
            source.source_id
        ] = {
            "cursor":
                int(
                    cursor
                ),

            "revision":
                source.revision,

            "updated_at":
                _utc_now(),
        }

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = (
            self.path
            .with_suffix(
                self.path.suffix
                + ".tmp"
            )
        )

        temporary.write_text(
            json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            temporary,
            self.path,
        )


def _iter_local_jsonl(
    source: CorpusRegistrySource,
    *,
    start_index: int,
) -> Iterator[
    tuple[
        int,
        dict[str, Any],
    ]
]:

    assert (
        source.local_path
        is not None
    )

    path = Path(
        source.local_path
    ).expanduser()

    if not path.is_absolute():

        path = (
            REPOSITORY_ROOT
            / path
        )

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for index, line in enumerate(
            handle
        ):

            if index < start_index:
                continue

            line = line.strip()

            if not line:
                continue

            value = json.loads(
                line
            )

            if not isinstance(
                value,
                dict,
            ):
                continue

            yield (
                index,
                value,
            )


def _iter_huggingface(
    source: CorpusRegistrySource,
    *,
    start_index: int,
    max_rows: int,
) -> Iterator[
    tuple[
        int,
        dict[str, Any],
    ]
]:
    """
    Read Hugging Face rows through a disposable child process.

    The child receives its own row budget and exits via os._exit().
    Therefore PyArrow/datasets native teardown never enters the
    authoritative continual-learning process.
    """

    if not source.dataset_id:
        raise ValueError(
            "Hugging Face source has no dataset_id."
        )

    if max_rows <= 0:
        return

    split = str(
        source.metadata.get(
            "split",
            "train",
        )
    )

    command = [
        sys.executable,
        "-m",
        "learning.continual.hf_stream_worker",
        "--dataset-id",
        source.dataset_id,
        "--split",
        split,
        "--start-index",
        str(
            start_index
        ),
        "--max-rows",
        str(
            max_rows
        ),
    ]

    if source.subset:
        command.extend(
            [
                "--subset",
                source.subset,
            ]
        )

    if source.revision:
        command.extend(
            [
                "--revision",
                source.revision,
            ]
        )

    process = subprocess.Popen(
        command,

        cwd=str(
            REPOSITORY_ROOT
        ),

        stdout=subprocess.PIPE,

        # Keep network/HF diagnostics visible, but separate from
        # the JSONL IPC stream on stdout.
        stderr=None,

        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    if process.stdout is None:
        process.kill()
        process.wait()

        raise RuntimeError(
            "Hugging Face reader worker has no stdout pipe."
        )

    reader_completed = False

    try:
        for line in process.stdout:

            line = line.strip()

            if not line:
                continue

            try:
                envelope = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Hugging Face reader emitted malformed JSONL."
                ) from exc

            if not isinstance(
                envelope,
                dict,
            ):
                continue

            index = envelope.get(
                "index"
            )

            row = envelope.get(
                "row"
            )

            if not (
                isinstance(
                    index,
                    int,
                )
                and isinstance(
                    row,
                    dict,
                )
            ):
                continue

            yield (
                index,
                row,
            )

        exit_code = process.wait()

        if exit_code != 0:
            raise RuntimeError(
                (
                    "Hugging Face reader worker exited "
                    f"with code {exit_code}."
                )
            )

        reader_completed = True

    finally:
        try:
            process.stdout.close()
        except Exception:
            pass

        if (
            process.poll()
            is None
        ):
            # Parent stopped early, usually because the accepted-record
            # snapshot budget was reached.
            #
            # Closing stdout causes the worker's next write to raise
            # BrokenPipeError. The worker handles that by os._exit(0).
            #
            # Do NOT send SIGTERM first; that was the previous lifecycle
            # bug which could re-enter Python/native shutdown.
            try:
                process.wait(
                    timeout=5.0
                )

            except subprocess.TimeoutExpired:
                # Only a genuinely stuck disposable reader is killed.
                process.kill()
                process.wait()

        if (
            reader_completed
            and process.returncode != 0
        ):
            raise RuntimeError(
                (
                    "Hugging Face reader worker exited "
                    f"with code {process.returncode}."
                )
            )


def _iter_source(
    source: CorpusRegistrySource,
    *,
    start_index: int,
    max_rows: int,
) -> Iterator[
    tuple[
        int,
        dict[str, Any],
    ]
]:

    if max_rows <= 0:
        return

    if source.provider == "local":

        emitted = 0

        for item in _iter_local_jsonl(
            source,
            start_index=start_index,
        ):
            yield item

            emitted += 1

            if emitted >= max_rows:
                break

        return

    if (
        source.provider
        == "huggingface"
    ):

        yield from _iter_huggingface(
            source,
            start_index=start_index,
            max_rows=max_rows,
        )

        return

    raise ValueError(
        (
            "External materializer supports local/Hugging Face "
            "sources only. Runtime evidence remains in the "
            "trajectory/feedback pipeline."
        )
    )


def _adapter_name(
    source: CorpusRegistrySource,
) -> str:

    explicit = source.metadata.get(
        "adapter"
    )

    if (
        isinstance(
            explicit,
            str,
        )
        and explicit.strip()
    ):
        return (
            explicit
            .strip()
            .lower()
        )

    name = (
        source.source_id
        .lower()
    )

    if (
        "themis" in name
        or "helpsteer" in name
    ):
        return "preference"

    if (
        "xlam" in name
        or "function" in name
    ):
        return "function_call"

    if (
        "rebench" in name
        or "swe-bench" in name
    ):
        return "issue_patch"

    if (
        "commitpack" in name
        or "commit" in name
    ):
        return "code_diff"

    if (
        "jira" in name
        or "issue-writer" in name
    ):
        return "instruction"

    if (
        "support" in name
        or "ticket" in name
    ):
        return "classification"

    if (
        "swe-zero" in name
        or "trajectory" in name
    ):
        return "trajectory"

    if "dpo" in source.objectives:
        return "preference"

    if "sft" in source.objectives:
        return "instruction"

    return "retrieval"


def _record(
    *,
    source: CorpusRegistrySource,
    upstream_index: int,
    objective: NormalizedObjective,
    row: dict[str, Any],
    prompt_messages: list[
        dict[str, str]
    ] | None = None,
    chosen: str | None = None,
    rejected: str | None = None,
    text: str | None = None,
    verified_outcome: bool | None = None,
    adapter: str,
) -> NormalizedCorpusRecord:

    payload = {
        "source_id":
            source.source_id,

        "source_record_id":
            _source_record_id(
                row
            ),

        "upstream_index":
            upstream_index,

        "objective":
            objective,

        "target_component":
            source.target_component,

        "prompt_messages":
            prompt_messages
            or [],

        "chosen":
            chosen,

        "rejected":
            rejected,

        "text":
            text,

        "language":
            _language(
                row
            ),

        "task":
            _task(
                row
            ),

        "verified_outcome":
            (
                _verified_outcome(
                    row
                )
                if verified_outcome
                is None
                else verified_outcome
            ),

        "metadata":
            {
                "adapter":
                    adapter,

                "contamination_group":
                    source.contamination_group,

                "repository":
                    _first_text(
                        row,
                        "repo",
                        "repository",
                        "repo_name",
                    ),

                "base_commit":
                    _first_text(
                        row,
                        "base_commit",
                        "base_sha",
                        "commit",
                    ),

                "repository_license":
                    _first_text(
                        row,
                        "license",
                        "repository_license",
                        "repo_license",
                    ),
            },
    }

    content_sha = (
        _sha256_text(
            canonical_json(
                payload
            )
        )
    )

    return NormalizedCorpusRecord(
        record_id=(
            "corpus-record-"
            + content_sha[
                :24
            ]
        ),

        content_sha256=(
            content_sha
        ),

        **payload,
    )


def _normalize_preference(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    existing_messages = _messages(
        _first_raw(
            row,
            "prompt_messages",
            "messages",
        )
    )

    prompt = _first_text(
        row,
        "prompt",
        "instruction",
        "question",
        "query",
        "problem",
        "input",
    )

    chosen = _first_text(
        row,
        "chosen",
        "chosen_response",
        "preferred",
        "preferred_response",
        "response_j",
        "winner",
    )

    rejected = _first_text(
        row,
        "rejected",
        "rejected_response",
        "non_preferred",
        "non_preferred_response",
        "response_k",
        "loser",
    )

    if not (
        chosen
        and rejected
        and chosen.strip()
        != rejected.strip()
    ):
        return None

    if not existing_messages:

        if not prompt:
            return None

        existing_messages = (
            _prompt_messages(
                prompt
            )
        )

    return _record(
        source=source,
        upstream_index=index,
        objective="dpo",
        row=row,
        prompt_messages=existing_messages,
        chosen=chosen,
        rejected=rejected,
        adapter="preference",
    )


def _normalize_instruction(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    prompt = _first_text(
        row,
        "instruction",
        "prompt",
        "query",
        "question",
        "input",
        "request",
        "problem",
    )

    chosen = _first_text(
        row,
        "output",
        "response",
        "answer",
        "completion",
        "target",
    )

    if not (
        prompt
        and chosen
    ):
        return None

    return _record(
        source=source,
        upstream_index=index,
        objective="sft",
        row=row,
        prompt_messages=(
            _prompt_messages(
                prompt
            )
        ),
        chosen=chosen,
        adapter="instruction",
    )


def _normalize_issue_patch(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    issue = _first_text(
        row,
        "problem_statement",
        "problem",
        "issue",
        "issue_text",
        "description",
        "title",
    )

    patch = _first_text(
        row,
        "patch",
        "gold_patch",
        "solution_patch",
        "diff",
    )

    if not (
        issue
        and patch
    ):
        return None

    prompt = (
        "Resolve the software issue below with a minimal, "
        "testable source patch.\n\n"
        + issue
    )

    return _record(
        source=source,
        upstream_index=index,
        objective="sft",
        row=row,
        prompt_messages=(
            _prompt_messages(
                prompt,
                system=(
                    "You are a software-engineering agent. "
                    "Inspect the issue, preserve unrelated behavior, "
                    "and produce only the necessary patch."
                ),
            )
        ),
        chosen=patch,
        adapter="issue_patch",
    )


def _normalize_code_diff(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    message = _first_text(
        row,
        "commit_message",
        "message",
        "subject",
        "description",
    )

    before = _first_text(
        row,
        "old_contents",
        "old_content",
        "before",
        "source_before",
    )

    after = _first_text(
        row,
        "new_contents",
        "new_content",
        "after",
        "source_after",
        "patch",
        "diff",
    )

    if not after:
        return None

    prompt_parts = [
        (
            "Apply the repository change represented by "
            "this commit while preserving unrelated behavior."
        )
    ]

    if message:

        prompt_parts.append(
            "Commit intent:\n"
            + message
        )

    if before:

        prompt_parts.append(
            "Previous source:\n"
            + before
        )

    return _record(
        source=source,
        upstream_index=index,
        objective="sft",
        row=row,
        prompt_messages=(
            _prompt_messages(
                "\n\n".join(
                    prompt_parts
                ),
                system=(
                    "You are studying a verified repository change. "
                    "Prefer minimal, coherent edits."
                ),
            )
        ),
        chosen=after,
        adapter="code_diff",
    )


def _normalize_function_call(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    prompt = _first_text(
        row,
        "query",
        "prompt",
        "instruction",
        "question",
        "input",
    )

    answer = _first_text(
        row,
        "answer",
        "answers",
        "response",
        "output",
        "function_call",
        "tool_call",
    )

    if not (
        prompt
        and answer
    ):
        return None

    tools = _first_text(
        row,
        "tools",
        "functions",
        "apis",
        "function",
    )

    if tools:

        prompt = (
            prompt
            + "\n\nAvailable tools:\n"
            + tools
        )

    return _record(
        source=source,
        upstream_index=index,
        objective="sft",
        row=row,
        prompt_messages=(
            _prompt_messages(
                prompt,
                system=(
                    "Select only grounded tools and arguments. "
                    "Do not invent values that were not supplied."
                ),
            )
        ),
        chosen=answer,
        adapter="function_call",
    )


def _normalize_classification(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    ticket = _first_text(
        row,
        "text",
        "ticket",
        "description",
        "body",
        "message",
        "request",
        "subject",
    )

    label = _first_text(
        row,
        "label",
        "category",
        "queue",
        "class",
        "type",
        "target",
    )

    if not (
        ticket
        and label
    ):
        return None

    return _record(
        source=source,
        upstream_index=index,
        objective="sft",
        row=row,
        prompt_messages=(
            _prompt_messages(
                (
                    "Classify the following IT service/support "
                    "request using the available taxonomy.\n\n"
                    + ticket
                )
            )
        ),
        chosen=label,
        adapter="classification",
    )


def _normalize_trajectory(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    conversation = _messages(
        _first_raw(
            row,
            "messages",
            "trajectory",
            "conversation",
        )
    )

    if not conversation:
        return None

    final_assistant_index = None

    for offset in range(
        len(
            conversation
        )
        - 1,
        -1,
        -1,
    ):

        if (
            conversation[
                offset
            ][
                "role"
            ]
            == "assistant"
        ):
            final_assistant_index = offset
            break

    if (
        final_assistant_index
        is None
        or final_assistant_index
        <= 0
    ):
        return None

    chosen = (
        conversation[
            final_assistant_index
        ][
            "content"
        ]
    )

    prompt_messages = (
        conversation[
            :final_assistant_index
        ]
    )

    return _record(
        source=source,
        upstream_index=index,
        objective="sft",
        row=row,
        prompt_messages=prompt_messages,
        chosen=chosen,
        adapter="trajectory",
    )


def _normalize_retrieval(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    text = _first_text(
        row,
        "content",
        "text",
        "document",
        "source",
        "body",
        "code",
    )

    if not text:
        return None

    return _record(
        source=source,
        upstream_index=index,
        objective="retrieval",
        row=row,
        text=text,
        adapter="retrieval",
    )


def normalize_source_row(
    source: CorpusRegistrySource,
    *,
    index: int,
    row: dict[str, Any],
) -> NormalizedCorpusRecord | None:

    adapter = _adapter_name(
        source
    )

    if adapter == "preference":

        return _normalize_preference(
            source,
            index=index,
            row=row,
        )

    if adapter == "issue_patch":

        return _normalize_issue_patch(
            source,
            index=index,
            row=row,
        )

    if adapter == "code_diff":

        return _normalize_code_diff(
            source,
            index=index,
            row=row,
        )

    if adapter == "function_call":

        return _normalize_function_call(
            source,
            index=index,
            row=row,
        )

    if adapter == "classification":

        return _normalize_classification(
            source,
            index=index,
            row=row,
        )

    if adapter == "trajectory":

        return _normalize_trajectory(
            source,
            index=index,
            row=row,
        )

    if adapter == "instruction":

        return _normalize_instruction(
            source,
            index=index,
            row=row,
        )

    return _normalize_retrieval(
        source,
        index=index,
        row=row,
    )


_SECRET_PATTERNS = [
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),

    re.compile(
        r"\bAKIA[0-9A-Z]{16}\b"
    ),

    re.compile(
        r"\bghp_[A-Za-z0-9]{20,}\b"
    ),

    re.compile(
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    ),

    re.compile(
        r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"
    ),
]


def _record_text(
    record: NormalizedCorpusRecord,
) -> str:

    parts = [
        item.get(
            "content",
            ""
        )
        for item
        in record.prompt_messages
    ]

    for value in (
        record.chosen,
        record.rejected,
        record.text,
    ):

        if value:
            parts.append(
                value
            )

    return "\n".join(
        parts
    )


def _has_secret(
    record: NormalizedCorpusRecord,
) -> bool:

    value = _record_text(
        record
    )

    return any(
        pattern.search(
            value
        )
        is not None

        for pattern
        in _SECRET_PATTERNS
    )


def _normalized_set(
    values: list[str],
) -> set[str]:

    return {
        value.strip().lower()
        for value in values
        if value.strip()
    }


def _filter_reason(
    source: CorpusRegistrySource,
    record: NormalizedCorpusRecord,
    *,
    seen_hashes: set[str],
    decontamination_policy: (
        CorpusDecontaminationPolicy
        | None
    ) = None,
) -> str | None:

    policy = source.filters

    if (
        policy.deduplicate
        and record.content_sha256
        in seen_hashes
    ):
        return "duplicate"

    if (
        policy.secret_scan
        and _has_secret(
            record
        )
    ):
        return "secret_like_content"

    allowed_languages = (
        _normalized_set(
            policy.languages
        )
    )

    if (
        allowed_languages
        and record.language
        and record.language
        .strip()
        .lower()
        not in allowed_languages
    ):
        return "language_filtered"

    include_tasks = (
        _normalized_set(
            policy.include_tasks
        )
    )

    exclude_tasks = (
        _normalized_set(
            policy.exclude_tasks
        )
    )

    task = (
        record.task
        .strip()
        .lower()

        if record.task

        else None
    )

    if (
        include_tasks
        and task
        and task
        not in include_tasks
    ):
        return "task_not_included"

    if (
        exclude_tasks
        and task
        and task
        in exclude_tasks
    ):
        return "task_excluded"

    if (
        policy.require_verified_outcome
        and not record.verified_outcome
    ):
        return "unverified_outcome"

    if (
        policy.require_permissive_source_license
    ):
        repository_license = (
            record.metadata.get(
                "repository_license"
            )
        )

        if not (
            isinstance(
                repository_license,
                str,
            )
            and repository_license.strip()
        ):
            return (
                "repository_license_missing"
            )

        normalized_license = (
            repository_license
            .strip()
            .lower()
        )

        permissive_licenses = {
            "mit",
            "apache-2.0",
            "apache 2.0",
            "bsd-2-clause",
            "bsd-3-clause",
            "isc",
            "unlicense",
            "cc0-1.0",
        }

        if (
            normalized_license
            not in permissive_licenses
        ):
            return (
                "repository_license_not_permitted"
            )

    if (
        policy.benchmark_decontamination
        and decontamination_policy
        is not None
    ):
        reason = decontamination_reason(
            policy=(
                decontamination_policy
            ),
            source_record_id=(
                record.source_record_id
            ),
            repository=(
                record.metadata.get(
                    "repository"
                )
            ),
            base_commit=(
                record.metadata.get(
                    "base_commit"
                )
            ),
            content_sha256=(
                record.content_sha256
            ),
        )

        if reason is not None:
            return reason

    return None


def snapshot_training_blockers(
    source: CorpusRegistrySource,
    *,
    decontamination_path: Path = (
        DEFAULT_DECONTAMINATION_PATH
    ),
) -> list[str]:

    if source.provider in {
        "runtime",
        "generated",
    }:
        return [
            "runtime_evidence_managed_by_trajectory_pipeline"
        ]

    blockers: list[str] = []

    if not source.training_eligible:

        blockers.append(
            "registry_source_not_training_eligible"
        )

    if source.evaluation_only:

        blockers.append(
            "evaluation_only_source"
        )

    if (
        source.provider
        == "huggingface"
        and not source.revision
    ):
        blockers.append(
            "upstream_revision_unpinned"
        )

    if (
        source.filters
        .benchmark_decontamination
    ):
        decontamination_manifest = (
            load_decontamination_manifest(
                decontamination_path
            )
        )

        if (
            decontamination_manifest
            .policy_for(
                source.source_id
            )
            is None
        ):
            blockers.append(
                "benchmark_decontamination_manifest_not_wired"
            )

    if (
        source.target_component
        not in {
            "hub",
            "developer-specialist",
        }
    ):
        blockers.append(
            "target_specific_trainer_not_wired"
        )

    developer_sft_bridge_supported = (
        source.target_component
        == "developer-specialist"

        and "sft"
        in source.objectives

        and all(
            objective
            == "sft"

            for objective
            in source.objectives
        )
    )

    # Developer SFT now has a candidate-only target-aware bridge.
    #
    # DPO, Hub external corpora, Jira/ITSM and other specialist
    # objectives remain explicitly blocked until their own bridge
    # and evaluation contracts are wired.
    if (
        source.training_eligible
        and not developer_sft_bridge_supported
    ):

        blockers.append(
            "objective_specific_optimizer_bridge_not_wired"
        )

    return blockers


def materialize_corpus_source(
    *,
    registry_path: Path,
    source_id: str,
    scan_limit: int | None = None,
    reset_cursor: bool = False,
    cursor_store: CorpusCursorStore | None = None,
    snapshot_root: Path = (
        CORPUS_SNAPSHOT_ROOT
    ),
    decontamination_path: Path = (
        DEFAULT_DECONTAMINATION_PATH
    ),
) -> CorpusMaterializationResult:

    registry = load_corpus_registry(
        registry_path
    )

    source = next(
        (
            item
            for item
            in registry.sources
            if item.source_id
            == source_id
        ),
        None,
    )

    if source is None:

        raise ValueError(
            (
                "Unknown corpus source: "
                + source_id
            )
        )

    if not source.enabled:

        raise ValueError(
            (
                "Corpus source is disabled: "
                + source_id
            )
        )

    if source.provider in {
        "runtime",
        "generated",
    }:

        raise ValueError(
            (
                "Runtime/generated evidence does not use "
                "the external corpus materializer."
            )
        )

    decontamination_manifest = (
        load_decontamination_manifest(
            decontamination_path
        )
    )

    source_decontamination_policy = (
        decontamination_manifest
        .policy_for(
            source.source_id
        )
    )

    source_decontamination_sha = (
        policy_sha256(
            source_decontamination_policy
        )
        if source_decontamination_policy
        is not None
        else None
    )

    cursor_store = (
        cursor_store
        or CorpusCursorStore()
    )

    cursor_start = (
        0
        if reset_cursor
        else cursor_store.get(
            source
        )
    )

    accepted_budget = (
        source.sampling
        .max_records_per_snapshot
    )

    if accepted_budget <= 0:

        raise ValueError(
            "max_records_per_snapshot must be positive."
        )

    if scan_limit is None:

        scan_limit = max(
            accepted_budget,
            min(
                accepted_budget
                * 10,
                100_000,
            ),
        )

    if scan_limit <= 0:

        raise ValueError(
            "scan_limit must be positive."
        )

    accepted: list[
        NormalizedCorpusRecord
    ] = []

    seen_hashes: set[str] = set()

    rejected_counts: dict[
        str,
        int,
    ] = {}

    scanned_count = 0
    cursor_end = cursor_start

    iterator = _iter_source(
        source,
        start_index=cursor_start,
        max_rows=scan_limit,
    )

    try:

        for index, row in iterator:

            if (
                scanned_count
                >= scan_limit
            ):
                break

            scanned_count += 1

            cursor_end = (
                index
                + 1
            )

            try:

                record = (
                    normalize_source_row(
                        source,
                        index=index,
                        row=row,
                    )
                )

            except Exception:

                rejected_counts[
                    "normalization_error"
                ] = (
                    rejected_counts.get(
                        "normalization_error",
                        0,
                    )
                    + 1
                )

                continue

            if record is None:

                rejected_counts[
                    "unsupported_or_incomplete"
                ] = (
                    rejected_counts.get(
                        "unsupported_or_incomplete",
                        0,
                    )
                    + 1
                )

                continue

            reason = _filter_reason(
                source,
                record,
                seen_hashes=seen_hashes,
                decontamination_policy=(
                    source_decontamination_policy
                ),
            )

            if reason is not None:

                rejected_counts[
                    reason
                ] = (
                    rejected_counts.get(
                        reason,
                        0,
                    )
                    + 1
                )

                continue

            seen_hashes.add(
                record.content_sha256
            )

            accepted.append(
                record
            )

            if (
                len(
                    accepted
                )
                >= accepted_budget
            ):
                break

    finally:

        # Hugging Face streaming commonly wraps PyArrow/native iterators.
        #
        # Bounded continual-learning reads deliberately stop before source
        # exhaustion, so explicitly close the Python generator rather than
        # leaving native iterator teardown until interpreter finalization.
        close = getattr(
            iterator,
            "close",
            None,
        )

        if callable(
            close
        ):
            close()

    records_payload = "".join(
        canonical_json(
            item.model_dump(
                mode="json",
                by_alias=True,
            )
        )
        + "\n"

        for item
        in accepted
    )

    records_sha = (
        _sha256_text(
            records_payload
        )
    )

    source_config = (
        source.model_dump(
            mode="json",
            by_alias=True,
        )
    )

    source_config_sha = (
        _sha256_text(
            canonical_json(
                source_config
            )
        )
    )

    identity = {
        "source_id":
            source.source_id,

        "revision":
            source.revision,

        "cursor_start":
            cursor_start,

        "cursor_end":
            cursor_end,

        "records_sha256":
            records_sha,

        "source_config_sha256":
            source_config_sha,

        "decontamination_policy_sha256":
            source_decontamination_sha,
    }

    snapshot_id = (
        "corpus-snapshot-"
        + _sha256_text(
            canonical_json(
                identity
            )
        )[:24]
    )

    root = (
        snapshot_root
        .expanduser()
        .resolve()
    )

    source_root = (
        root
        / _safe_name(
            source.source_id
        )
    )

    final_directory = (
        source_root
        / snapshot_id
    )

    blockers = (
        snapshot_training_blockers(
            source,
            decontamination_path=(
                decontamination_path
            ),
        )
    )

    manifest = (
        CorpusSnapshotManifest(
            snapshot_id=snapshot_id,
            created_at=_utc_now(),
            source_id=source.source_id,
            provider=source.provider,
            dataset_id=source.dataset_id,
            subset=source.subset,
            revision=source.revision,
            split=(
                str(
                    source.metadata.get(
                        "split",
                        "train",
                    )
                )
                if (
                    source.provider
                    == "huggingface"
                )
                else None
            ),
            target_component=(
                source.target_component
            ),
            objectives=[
                str(
                    item
                )
                for item
                in source.objectives
            ],
            source_config_sha256=(
                source_config_sha
            ),
            decontamination_policy_sha256=(
                source_decontamination_sha
            ),
            cursor_start=cursor_start,
            cursor_end=cursor_end,
            scanned_count=scanned_count,
            accepted_count=len(
                accepted
            ),
            rejected_counts=(
                rejected_counts
            ),
            records_sha256=records_sha,
            training_eligible=(
                bool(
                    accepted
                )
                and not blockers
            ),
            training_blockers=blockers,
            output_directory=str(
                final_directory
            ),
        )
    )

    source_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not final_directory.exists():

        temporary = Path(
            tempfile.mkdtemp(
                prefix=".snapshot-",
                dir=source_root,
            )
        )

        try:

            (
                temporary
                / "records.jsonl"
            ).write_text(
                records_payload,
                encoding="utf-8",
            )

            (
                temporary
                / "manifest.json"
            ).write_text(
                manifest.model_dump_json(
                    by_alias=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            os.replace(
                temporary,
                final_directory,
            )

        except Exception:

            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )

            raise

    # Advance only AFTER immutable snapshot creation succeeds.
    cursor_store.set(
        source,
        cursor_end,
    )

    return CorpusMaterializationResult(
        manifest=manifest,
        records_path=str(
            final_directory
            / "records.jsonl"
        ),
        manifest_path=str(
            final_directory
            / "manifest.json"
        ),
    )


def list_corpus_snapshots(
    *,
    snapshot_root: Path = (
        CORPUS_SNAPSHOT_ROOT
    ),
) -> list[
    CorpusSnapshotManifest
]:

    root = (
        snapshot_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        return []

    result: list[
        CorpusSnapshotManifest
    ] = []

    for path in root.glob(
        "*/*/manifest.json"
    ):

        try:

            manifest = (
                CorpusSnapshotManifest
                .model_validate_json(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        except Exception:
            continue

        result.append(
            manifest
        )

    return sorted(
        result,
        key=lambda item: (
            item.created_at,
            item.snapshot_id,
        ),
    )
