"""Prompt rendering identical to Janus serving.

A copy of ``subagents.llm.runtime.hf_prompt`` (the worker may not import Janus); the worker tests pin
both renderers to each other so a change to one fails until the other follows.

``template_options`` are the extra chat-template arguments the serving backend passes for the model
(Qwen3 is served with ``enable_thinking=False``, which adds an empty think block to the prompt). A model
trained or scored without them sees a different prompt than the one it is served with.
"""
from __future__ import annotations

FALLBACK_SCHEMA = (
    "hf-causal-role-labelled-prompt.v1|"
    "roles:system=SYSTEM,user=USER,assistant=ASSISTANT|"
    "message:{LABEL}:\\n{content.strip()}|"
    "separator:\\n\\n|"
    "generation_suffix:ASSISTANT:\\n"
)
_LABELS = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}


def render_fallback(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        parts.append(f"{_LABELS.get(role, role.upper())}:\n{message.get('content', '').strip()}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def render_prompt(tokenizer, messages: list[dict[str, str]], template_options: dict | None = None) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template.strip():
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **(template_options or {}))
    return render_fallback(messages)


def tokenizes_messages_itself(tokenizer) -> bool:
    """Mistral's tokenizer (MistralCommonBackend) has no text template: it encodes the messages directly, the
    way Ministral serving calls it. Its prompt as text re-tokenizes to other ids, so it is never used as text."""
    return type(tokenizer).__name__ == "MistralCommonBackend"


def encode_prompt(tokenizer, messages: list[dict[str, str]], template_options: dict | None = None) -> list[int]:
    """The prompt's token ids exactly as serving encodes them (special tokens included, once)."""
    if tokenizes_messages_itself(tokenizer):
        encoded = tokenizer.apply_chat_template(messages, return_dict=True)
        return [int(token) for token in encoded["input_ids"]]
    return list(tokenizer(render_prompt(tokenizer, messages, template_options), add_special_tokens=True)["input_ids"])
