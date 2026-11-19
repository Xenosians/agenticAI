from __future__ import annotations

import hashlib
import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from services.lua_adapters.model_loading import from_pretrained as lua_from_pretrained

from config import Settings
from learning.training.capability_curriculum import (
    build_agent_capability_records,
)
from learning.training.jira_sft import (
    assert_training_versions,
    build_jira_training_arguments,
    build_runtime_messages,
    fingerprint_directory,
)
from subagents.core.definitions.loader import load_agent_definition
from subagents.core.tooling.capabilities import build_agent_capability_catalog
from subagents.core.tooling.parser import parse_tool_calls
from subagents.core.tooling.prompt import build_worker_system_prompt
from subagents.llm.runtime.hf_prompt import render_hf_causal_fallback_prompt


CONFIG_SCHEMA = "specialist-training-config.v1"
CORPUS_SCHEMA = "specialist-sft-corpus.v1"
RECORD_SCHEMA = "specialist-sft-record.v1"
TRAINING_SCHEMA = "specialist-sft-training.v1"

SUPPORTED_PROMPT_PROFILES = {
    "standard",
    "compact",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _normalize_prompt_profile(
    value: str,
) -> str:
    normalized = value.strip().lower()

    if normalized not in SUPPORTED_PROMPT_PROFILES:
        raise ValueError(
            "worker_prompt_profile must be one of: "
            + ", ".join(sorted(SUPPORTED_PROMPT_PROFILES))
        )

    return normalized


class SpecialistExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    user_request: str
    arguments: dict[str, Any]
    tags: list[str] = Field(default_factory=list)


class SpecialistEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    user: str
    allowed_tools: list[str]
    allowed_arguments: dict[str, list[str]] = Field(default_factory=dict)
    expected_tool: str
    expected_arguments: dict[str, Any]
    allowed_extra_arguments: list[str] = Field(default_factory=list)


class SpecialistTrainingConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
    )
    specialist: str
    agent_definition: str
    base_model_key: str
    candidate_model_key: str

    # Optional training/evaluation prompt override.
    #
    # This does NOT mutate the base model profile.  It lets a dedicated
    # specialist candidate train against the compact worker protocol while
    # hub-main continues to use its production "standard" profile.
    worker_prompt_profile: str | None = None

    examples: list[SpecialistExample] = Field(default_factory=list)
    eval_cases: list[SpecialistEvalCase] = Field(default_factory=list)
    max_new_tokens: int = 160


class SpecialistSftRecord(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=RECORD_SCHEMA,
        alias="schema",
    )
    record_id: str
    split: str
    tool_name: str
    user_request: str
    semantic_context: dict[str, Any]
    target_response: str
    tags: list[str] = Field(default_factory=list)


class SpecialistSftCorpusManifest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=CORPUS_SCHEMA,
        alias="schema",
    )
    created_at: str
    specialist: str
    source: str
    agent_name: str
    agent_definition: str
    base_model_key: str
    candidate_model_key: str
    worker_prompt_profile: str
    record_count: int
    train_count: int
    validation_count: int
    per_tool_counts: dict[str, int]
    train_sha256: str
    validation_sha256: str
    agent_definition_sha256: str
    capability_catalog_sha256: str
    system_prompt_sha256: str


class ExternalSftSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    record_count: int


class SpecialistSftTrainingManifest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=TRAINING_SCHEMA,
        alias="schema",
    )
    created_at: str
    specialist: str
    agent_name: str
    base_model_key: str
    candidate_model_key: str
    worker_prompt_profile: str
    base_model_path: str
    base_adapter_path: str | None = None
    base_adapter_sha256: str | None = None
    continued_from_adapter: bool = False
    output_directory: str
    adapter_directory: str
    merged_model_directory: str | None = None
    corpus_directory: str
    corpus_train_sha256: str
    corpus_validation_sha256: str
    external_sft_sources: list[ExternalSftSource] = Field(default_factory=list)
    max_steps: int
    learning_rate: float
    max_length: int
    lora_r: int
    lora_alpha: int
    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float
    adapter_sha256: str
    merged_model_sha256: str | None = None
    training_versions: dict[str, str]


def load_training_config(
    path: Path,
) -> SpecialistTrainingConfig:
    resolved = path.expanduser().resolve()
    payload = json.loads(
        resolved.read_text(encoding="utf-8")
    )
    config = SpecialistTrainingConfig.model_validate(payload)

    if config.worker_prompt_profile is not None:
        config.worker_prompt_profile = _normalize_prompt_profile(
            config.worker_prompt_profile
        )

    return config


def resolve_worker_prompt_profile(
    *,
    config: SpecialistTrainingConfig,
    base_profile,
) -> str:
    configured = config.worker_prompt_profile

    if configured is not None:
        return _normalize_prompt_profile(configured)

    return _normalize_prompt_profile(
        base_profile.worker_prompt_profile
    )


def _resolve_training_profile_paths(
    profile,
    *,
    label: str,
) -> tuple[Path, Path | None]:
    # Resolve base checkpoint and optional promoted PEFT adapter.
    # The adapter is training lineage, never authorization.
    if profile.model_path is None:
        raise ValueError(
            f"{label} model profile has no model_path."
        )

    model_path = (
        profile.model_path
        .expanduser()
        .resolve()
    )

    if not model_path.is_dir():
        raise ValueError(
            f"{label} model directory does not exist: "
            f"{model_path}"
        )

    raw_adapter = getattr(
        profile,
        "adapter_path",
        None,
    )

    if raw_adapter is None:
        return model_path, None

    adapter_path = (
        raw_adapter
        .expanduser()
        .resolve()
    )

    if not adapter_path.is_dir():
        raise ValueError(
            f"{label} adapter directory does not exist: "
            f"{adapter_path}"
        )

    config_path = (
        adapter_path
        / "adapter_config.json"
    )

    weights = [
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    ]

    if (
        not config_path.is_file()
        or not any(
            candidate.is_file()
            for candidate in weights
        )
    ):
        raise ValueError(
            f"{label} adapter is incomplete: "
            f"{adapter_path}"
        )

    return model_path, adapter_path


def canonical_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    return _canonical_json(
        [
            {
                "name": tool_name,
                "arguments": arguments,
            }
        ]
    )


def semantic_context_for(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """The semantic context the runtime router can produce for this call.

    Only the capability's grounded arguments are bound, exactly as the runtime
    intent parser requires; an optional bounded argument (a time window, a
    line count, a working directory) stays in the call and is checked by the
    executable, never bound as a semantic target.
    """
    from subagents.core.orchestration.intent_contract import trusted_grounded_arguments
    from tools.registry import get_tool

    grounded = set(trusted_grounded_arguments(tool_name=tool_name, tool=get_tool(tool_name)))
    allowed_arguments: dict[str, list[str]] = {}

    for key, value in arguments.items():
        if key not in grounded:
            continue
        if isinstance(value, str):
            allowed_arguments[key] = [value]

        # An integer identifier (a Proxmox VMID) is grounded by its exact
        # decimal text, as at runtime; bool is never an identifier.
        elif type(value) is int:
            allowed_arguments[key] = [str(value)]

        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
        ):
            allowed_arguments[key] = list(value)

    return {
        "allowed_tools": [tool_name],
        "allowed_arguments": allowed_arguments,
        "forbidden_tools": [],
        "forbidden_arguments": {},
    }


def _agent_identity_hash(agent) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "name": agent.name,
                "description": agent.description,
                "model": agent.model,
                "tools": list(agent.tools),
                "max_steps": agent.max_steps,
                "system_prompt": agent.system_prompt,
            }
        )
    )


def _write_jsonl(
    path: Path,
    records: list[SpecialistSftRecord],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(
                _canonical_json(
                    record.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )
                + "\n"
            )


def _load_jsonl(
    path: Path,
) -> list[SpecialistSftRecord]:
    records: list[SpecialistSftRecord] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(
                SpecialistSftRecord.model_validate(
                    json.loads(line)
                )
            )

        except Exception as exc:
            raise ValueError(
                f"Invalid specialist SFT record in {path} "
                f"line {line_number}: {exc}"
            ) from exc

    return records


def build_specialist_sft_corpus(
    *,
    project_root: Path,
    config_path: Path,
    output_directory: Path,
    force: bool = False,
) -> SpecialistSftCorpusManifest:
    import shutil

    project_root = project_root.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    config = load_training_config(config_path)

    if output_directory.exists():
        if not force:
            raise FileExistsError(
                f"Specialist SFT corpus already exists: {output_directory}"
            )

        shutil.rmtree(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    settings = Settings()
    base_profile = settings.require_model_profile(
        config.base_model_key
    )

    prompt_profile = resolve_worker_prompt_profile(
        config=config,
        base_profile=base_profile,
    )

    agent_path = (
        project_root
        / config.agent_definition
    ).resolve()

    agent = load_agent_definition(agent_path)

    if agent.name != config.specialist:
        raise ValueError(
            "Training config specialist does not match "
            f"agent definition: {config.specialist!r} != {agent.name!r}"
        )

    capabilities = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capabilities,
        prompt_profile=prompt_profile,
    )

    held_out_requests = {
        case.user.strip()
        for case in config.eval_cases
    }

    records: list[SpecialistSftRecord] = []
    seen_requests: set[str] = set()
    per_tool_index: dict[str, int] = {}

    for example in config.examples:
        request = example.user_request.strip()

        if not request or request in held_out_requests:
            continue

        if request in seen_requests:
            raise ValueError(
                f"Duplicate training request: {request}"
            )

        seen_requests.add(request)

        index = per_tool_index.get(
            example.tool_name,
            0,
        )
        per_tool_index[example.tool_name] = index + 1

        split = (
            "validation"
            if index % 5 == 0
            else "train"
        )

        records.append(
            SpecialistSftRecord(
                record_id=(
                    f"{config.specialist}-sft-"
                    f"{example.tool_name}-{index:04d}-"
                    f"{_sha256_text(request)[:12]}"
                ),
                split=split,
                tool_name=example.tool_name,
                user_request=request,
                semantic_context=semantic_context_for(
                    example.tool_name,
                    example.arguments,
                ),
                target_response=canonical_tool_call(
                    example.tool_name,
                    example.arguments,
                ),
                tags=[
                    *example.tags,
                    "reviewed-seed",
                ],
            )
        )

    generated = build_agent_capability_records(
        project_root=project_root,
        agent_definition=Path(config.agent_definition),
        excluded_requests=held_out_requests,
    )

    for generated_record in generated:
        request = generated_record.user_request.strip()

        if (
            not request
            or request in held_out_requests
            or request in seen_requests
        ):
            continue

        seen_requests.add(request)

        records.append(
            SpecialistSftRecord(
                record_id=(
                    f"{config.specialist}-sft-"
                    f"{generated_record.record_id}"
                ),
                split=generated_record.split,
                tool_name=generated_record.tool_name,
                user_request=request,
                semantic_context=semantic_context_for(
                    generated_record.tool_name,
                    generated_record.arguments,
                ),
                target_response=generated_record.target_response,
                tags=[
                    *generated_record.tags,
                    "capability-language",
                ],
            )
        )

    if not records:
        raise ValueError(
            "No specialist training records were generated."
        )

    train = [
        record
        for record in records
        if record.split == "train"
    ]
    validation = [
        record
        for record in records
        if record.split == "validation"
    ]

    if not train or not validation:
        raise ValueError(
            "Specialist corpus requires non-empty train and validation splits."
        )

    train_path = output_directory / "train.jsonl"
    validation_path = output_directory / "validation.jsonl"

    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)

    per_tool_counts: dict[str, int] = {}

    for record in records:
        per_tool_counts[record.tool_name] = (
            per_tool_counts.get(record.tool_name, 0)
            + 1
        )

    manifest = SpecialistSftCorpusManifest(
        created_at=_utc_now(),
        specialist=config.specialist,
        source="reviewed-seed+capability-language",
        agent_name=agent.name,
        agent_definition=config.agent_definition,
        base_model_key=config.base_model_key,
        candidate_model_key=config.candidate_model_key,
        worker_prompt_profile=prompt_profile,
        record_count=len(records),
        train_count=len(train),
        validation_count=len(validation),
        per_tool_counts=dict(
            sorted(per_tool_counts.items())
        ),
        train_sha256=_sha256_file(train_path),
        validation_sha256=_sha256_file(validation_path),
        agent_definition_sha256=_agent_identity_hash(agent),
        capability_catalog_sha256=_sha256_text(
            _canonical_json(capabilities)
        ),
        system_prompt_sha256=_sha256_text(system_prompt),
    )

    (output_directory / "manifest.json").write_text(
        json.dumps(
            manifest.model_dump(
                mode="json",
                by_alias=True,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def verify_specialist_sft_corpus(
    *,
    project_root: Path,
    corpus_directory: Path,
) -> tuple[
    SpecialistSftCorpusManifest,
    list[SpecialistSftRecord],
    list[SpecialistSftRecord],
    str,
]:
    project_root = project_root.expanduser().resolve()
    corpus_directory = corpus_directory.expanduser().resolve()

    manifest_path = corpus_directory / "manifest.json"
    train_path = corpus_directory / "train.jsonl"
    validation_path = corpus_directory / "validation.jsonl"

    for path in [
        manifest_path,
        train_path,
        validation_path,
    ]:
        if not path.is_file():
            raise ValueError(
                f"Specialist SFT corpus is incomplete: missing {path}"
            )

    manifest = SpecialistSftCorpusManifest.model_validate(
        json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    )

    prompt_profile = _normalize_prompt_profile(
        manifest.worker_prompt_profile
    )

    if _sha256_file(train_path) != manifest.train_sha256:
        raise ValueError(
            "Specialist SFT train split hash mismatch."
        )

    if _sha256_file(validation_path) != manifest.validation_sha256:
        raise ValueError(
            "Specialist SFT validation split hash mismatch."
        )

    settings = Settings()

    # The base model profile may remain "standard".  The corpus manifest
    # freezes the exact prompt profile used for this dedicated candidate.
    settings.require_model_profile(
        manifest.base_model_key
    )

    agent = load_agent_definition(
        project_root
        / manifest.agent_definition
    )

    capabilities = build_agent_capability_catalog(
        agent,
        include_arguments=True,
    )

    system_prompt = build_worker_system_prompt(
        agent,
        capability_catalog=capabilities,
        prompt_profile=prompt_profile,
    )

    if (
        _agent_identity_hash(agent)
        != manifest.agent_definition_sha256
    ):
        raise ValueError(
            "Specialist definition changed after corpus build. "
            "Rebuild the corpus."
        )

    if (
        _sha256_text(
            _canonical_json(capabilities)
        )
        != manifest.capability_catalog_sha256
    ):
        raise ValueError(
            "Capability catalog changed after corpus build. "
            "Rebuild the corpus."
        )

    if (
        _sha256_text(system_prompt)
        != manifest.system_prompt_sha256
    ):
        raise ValueError(
            "Specialist system prompt changed after corpus build. "
            "Rebuild the corpus."
        )

    train = _load_jsonl(train_path)
    validation = _load_jsonl(validation_path)

    if len(train) != manifest.train_count:
        raise ValueError(
            "Specialist SFT train record count mismatch."
        )

    if len(validation) != manifest.validation_count:
        raise ValueError(
            "Specialist SFT validation record count mismatch."
        )

    for record in [
        *train,
        *validation,
    ]:
        calls = parse_tool_calls(
            record.target_response
        )

        if len(calls) != 1:
            raise ValueError(
                "Specialist SFT target must contain exactly one call: "
                f"{record.record_id}"
            )

        call = calls[0]

        if call.get("name") != record.tool_name:
            raise ValueError(
                f"Specialist SFT target tool mismatch: {record.record_id}"
            )

        if (
            record.tool_name
            not in record.semantic_context.get(
                "allowed_tools",
                [],
            )
        ):
            raise ValueError(
                "Specialist semantic-context tool mismatch: "
                f"{record.record_id}"
            )

    return (
        manifest,
        train,
        validation,
        system_prompt,
    )


def _render_prompt(
    tokenizer,
    messages,
) -> str:
    template = getattr(
        tokenizer,
        "chat_template",
        None,
    )

    if isinstance(template, str) and template.strip():
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return render_hf_causal_fallback_prompt(
        messages
    )


def load_external_sft_jsonl(
    paths: list[Path],
) -> tuple[
    list[dict[str, Any]],
    list[ExternalSftSource],
]:
    records: list[dict[str, Any]] = []
    sources: list[ExternalSftSource] = []

    for raw_path in paths:
        path = raw_path.expanduser().resolve()

        if not path.is_file():
            raise ValueError(
                f"External SFT JSONL does not exist: {path}"
            )

        source_count = 0

        for line_number, line in enumerate(
            path.read_text(
                encoding="utf-8"
            ).splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid external SFT JSON in {path} "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                continue

            partition = value.get("partition")

            if partition not in (
                None,
                "train",
                "sft_train",
            ):
                continue

            prompt_messages = value.get(
                "prompt_messages"
            )
            chosen = value.get(
                "chosen"
            )

            if (
                not isinstance(prompt_messages, list)
                or not prompt_messages
                or not isinstance(chosen, str)
                or not chosen.strip()
            ):
                continue

            valid_messages = True

            for item in prompt_messages:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("role"), str)
                    or not isinstance(item.get("content"), str)
                ):
                    valid_messages = False
                    break

            if not valid_messages:
                continue

            records.append(
                {
                    "source_path": str(path),
                    "line_number": line_number,
                    "prompt_messages": prompt_messages,
                    "chosen": chosen.strip(),
                }
            )
            source_count += 1

        if source_count < 1:
            raise ValueError(
                "External SFT source contains no compatible "
                f"training records: {path}"
            )

        sources.append(
            ExternalSftSource(
                path=str(path),
                sha256=_sha256_file(path),
                record_count=source_count,
            )
        )

    return (
        records,
        sources,
    )


class SpecialistTokenDataset:
    def __init__(self, items) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


class SpecialistCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        max_length = max(
            len(item["input_ids"])
            for item in features
        )

        inputs = []
        attention = []
        labels = []

        for item in features:
            padding = (
                max_length
                - len(item["input_ids"])
            )

            inputs.append(
                item["input_ids"]
                + [self.pad_token_id] * padding
            )
            attention.append(
                item["attention_mask"]
                + [0] * padding
            )
            labels.append(
                item["labels"]
                + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                inputs,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def preflight_specialist_sft(
    *,
    project_root: Path,
    corpus_directory: Path,
    max_length: int,
    external_sft_jsonl: list[Path] | None = None,
) -> dict[str, Any]:
    versions = assert_training_versions()

    (
        manifest,
        train,
        validation,
        system_prompt,
    ) = verify_specialist_sft_corpus(
        project_root=project_root,
        corpus_directory=corpus_directory,
    )

    settings = Settings()
    profile = settings.require_model_profile(
        manifest.base_model_key
    )

    (
        model_path,
        base_adapter_path,
    ) = _resolve_training_profile_paths(
        profile,
        label="Base specialist",
    )

    from transformers import AutoTokenizer

    tokenizer = lua_from_pretrained(AutoTokenizer.from_pretrained,
        str(model_path),
        local_files_only=True,
        fix_mistral_regex=True,
    )

    prompt_lengths = []
    target_lengths = []

    for record in [
        *train,
        *validation,
    ]:
        messages = build_runtime_messages(
            system_prompt=system_prompt,
            user_request=record.user_request,
            semantic_context=record.semantic_context,
        )

        prompt = _render_prompt(
            tokenizer,
            messages,
        )

        prompt_lengths.append(
            len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                )["input_ids"]
            )
        )

        target_lengths.append(
            len(
                tokenizer(
                    record.target_response,
                    add_special_tokens=False,
                )["input_ids"]
            )
        )

    external_records: list[dict[str, Any]] = []
    external_sources: list[ExternalSftSource] = []

    if external_sft_jsonl:
        (
            external_records,
            external_sources,
        ) = load_external_sft_jsonl(
            external_sft_jsonl
        )

        for record in external_records:
            prompt = _render_prompt(
                tokenizer,
                record["prompt_messages"],
            )

            prompt_lengths.append(
                len(
                    tokenizer(
                        prompt,
                        add_special_tokens=True,
                    )["input_ids"]
                )
            )

            target_lengths.append(
                len(
                    tokenizer(
                        record["chosen"],
                        add_special_tokens=False,
                    )["input_ids"]
                )
            )

    with tempfile.TemporaryDirectory(
        prefix="specialist-sft-preflight-"
    ) as temporary_output:
        args = build_jira_training_arguments(
            output_dir=Path(temporary_output),
            max_steps=120,
            learning_rate=2e-4,
            gradient_accumulation_steps=8,
            seed=42,
        )

    return {
        "specialist": manifest.specialist,
        "base_model_key": manifest.base_model_key,
        "candidate_model_key": manifest.candidate_model_key,
        "base_model_path": str(model_path),
        "base_adapter_path": (
            str(base_adapter_path)
            if base_adapter_path is not None
            else None
        ),
        "base_adapter_sha256": (
            fingerprint_directory(
                base_adapter_path
            )
            if base_adapter_path is not None
            else None
        ),
        "continued_from_adapter": (
            base_adapter_path is not None
        ),
        "model_class": (
            specialist_training_model_class_name(
                model_path
            )
        ),
        "prompt_profile": manifest.worker_prompt_profile,
        "records": manifest.record_count,
        "train": manifest.train_count,
        "validation": manifest.validation_count,
        "external_train": len(external_records),
        "external_sources": [
            source.model_dump()
            for source in external_sources
        ],
        "max_prompt_tokens": max(prompt_lengths),
        "max_target_tokens": max(target_lengths),
        "configured_max_length": max_length,
        "training_versions": versions,
        "warmup_steps": args.warmup_steps,
        "optimizer": str(args.optim),
    }


def _checkpoint_model_type(
    model_path: Path,
) -> str:
    from transformers import AutoConfig

    config = lua_from_pretrained(AutoConfig.from_pretrained,
        str(model_path),
        local_files_only=True,
    )

    value = getattr(
        config,
        "model_type",
        "",
    )

    return (
        value.strip().lower()
        if isinstance(value, str)
        else ""
    )


def specialist_training_model_class_name(
    model_path: Path,
) -> str:
    if _checkpoint_model_type(model_path) == "mistral3":
        return "Mistral3ForConditionalGeneration"

    return "AutoModelForCausalLM"


def register_candidate_profile(
    *,
    env_path: Path,
    training_manifest_path: Path,
) -> dict[str, Any]:
    env_path = env_path.expanduser().resolve()
    training_manifest_path = (
        training_manifest_path.expanduser().resolve()
    )

    if not env_path.is_file():
        raise ValueError(
            f".env file does not exist: {env_path}"
        )

    manifest = SpecialistSftTrainingManifest.model_validate(
        json.loads(
            training_manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    base_model_directory = Path(
        manifest.base_model_path
    ).expanduser().resolve()

    if not base_model_directory.is_dir():
        raise ValueError(
            "Specialist base model directory does not exist: "
            f"{base_model_directory}"
        )

    adapter_directory = Path(
        manifest.adapter_directory
    ).expanduser().resolve()

    if not adapter_directory.is_dir():
        raise ValueError(
            "Trained specialist adapter directory does not exist: "
            f"{adapter_directory}"
        )

    if (
        fingerprint_directory(
            adapter_directory
        )
        != manifest.adapter_sha256
    ):
        raise ValueError(
            "Specialist adapter fingerprint does not match "
            "the training manifest."
        )

    text = env_path.read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()

    profile_index = None
    raw_value = None

    for index, line in enumerate(lines):
        if line.startswith("MODEL_PROFILES="):
            profile_index = index
            raw_value = line.split("=", 1)[1].strip()
            break

    if profile_index is None or raw_value is None:
        raise ValueError(
            "MODEL_PROFILES is missing from .env."
        )

    if (
        len(raw_value) >= 2
        and raw_value[0] == raw_value[-1]
        and raw_value[0] in {"'", '"'}
    ):
        raw_value = raw_value[1:-1]

    profiles = json.loads(raw_value)

    if not isinstance(profiles, dict):
        raise ValueError(
            "MODEL_PROFILES must decode to an object."
        )

    base_profile = profiles.get(
        manifest.base_model_key
    )

    if not isinstance(base_profile, dict):
        raise ValueError(
            "Base model profile is missing from MODEL_PROFILES: "
            f"{manifest.base_model_key}"
        )

    candidate_profile = dict(
        base_profile
    )

    # Adapter-native specialist serving:
    #
    #     immutable base model
    #       +
    #     trained successor adapter
    #
    # If the base profile already has an adapter because this training
    # continued a specialist, replace it with the newly trained successor
    # adapter. Do not stack the predecessor and successor adapters.
    candidate_profile["model_path"] = str(
        base_model_directory
    )
    candidate_profile["adapter_path"] = str(
        adapter_directory
    )
    candidate_profile["enabled"] = True

    # Critical: runtime prompting must match the protocol the adapter
    # actually learned during training.
    candidate_profile["worker_prompt_profile"] = (
        manifest.worker_prompt_profile
    )

    profiles[
        manifest.candidate_model_key
    ] = candidate_profile

    encoded = json.dumps(
        profiles,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    lines[profile_index] = (
        "MODEL_PROFILES='"
        + encoded
        + "'"
    )

    temporary = env_path.with_name(
        ".env.specialist-registration.tmp"
    )

    temporary.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        env_path,
    )

    return candidate_profile
