"""Job data: training rows, preference pairs and held-out cases (JSON lines, JSON or Parquet)."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROLES = frozenset({"system", "user", "assistant"})


def read_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        rows = parquet.read_table(path).to_pylist()
    elif path.suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        rows = loaded if isinstance(loaded, list) else loaded.get("rows", [])
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} row {number} is not an object")
    return rows


def _messages(value: Any, where: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: prompt messages must be a non-empty list")
    messages = []
    for message in value:
        if not isinstance(message, dict) or message.get("role") not in ROLES \
                or not isinstance(message.get("content"), str):
            raise ValueError(f"{where}: each message needs a system, user or assistant role and text content")
        messages.append({"role": message["role"], "content": message["content"]})
    return messages


def _prompt_and_answer(row: dict[str, Any], where: str) -> tuple[list[dict[str, str]], str]:
    if "prompt_messages" in row:
        chosen = row.get("chosen", row.get("target_response"))
        if not isinstance(chosen, str) or not chosen.strip():
            raise ValueError(f"{where}: chosen must be non-empty text")
        return _messages(row["prompt_messages"], where), chosen
    messages = _messages(row.get("messages"), where)
    if messages[-1]["role"] != "assistant" or len(messages) < 2:
        raise ValueError(f"{where}: a conversation ends with the assistant's answer")
    return messages[:-1], messages[-1]["content"]


def sft_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, row in enumerate(read_rows(path), 1):
        prompt, chosen = _prompt_and_answer(row, f"{Path(path).name} row {number}")
        rows.append({"prompt_messages": prompt, "chosen": chosen})
    return rows


def preference_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, row in enumerate(read_rows(path), 1):
        where = f"{Path(path).name} row {number}"
        prompt, chosen = _prompt_and_answer({k: v for k, v in row.items() if k != "rejected"}, where)
        rejected = row.get("rejected")
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError(f"{where}: rejected must be non-empty text")
        if rejected.strip() != chosen.strip():
            rows.append({"prompt_messages": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def heldout_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for number, row in enumerate(read_rows(path), 1):
        where = f"{Path(path).name} row {number}"
        if not isinstance(row.get("name"), str) or not isinstance(row.get("expected_tool"), str):
            raise ValueError(f"{where}: a case has a name and an expected_tool")
        cases.append({
            "name": row["name"],
            "prompt_messages": _messages(row.get("prompt_messages"), where),
            "expected_tool": row["expected_tool"],
            "expected_arguments": dict(row.get("expected_arguments") or {}),
            "allowed_arguments": dict(row.get("allowed_arguments") or {}),
            "allowed_extra_arguments": list(row.get("allowed_extra_arguments") or []),
            "max_new_tokens": row.get("max_new_tokens"),
        })
    if not cases:
        raise ValueError("the job has no held-out cases")
    return cases


def cases_digest(cases: list[dict[str, Any]]) -> str:
    text = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split(rows: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list, list]:
    """Deterministic train/validation split; at least one validation row once there are two rows."""
    if len(rows) < 2:
        return list(rows), []
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    size = min(len(rows) - 1, max(1, round(len(rows) * fraction)))
    chosen = set(order[:size])
    return ([row for index, row in enumerate(rows) if index not in chosen],
            [row for index, row in enumerate(rows) if index in chosen])


def tokenize_pair(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, list[int]]:
    """Janus's specialist tokenization: the loss covers only the answer; a long prompt keeps its tail."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("an answer tokenized to zero tokens")
    if len(target_ids) >= max_length:
        raise ValueError("an answer is longer than max_length")
    room = max_length - len(target_ids)
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    input_ids = [*prompt_ids, *target_ids]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": [*([-100] * len(prompt_ids)), *target_ids]}


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
