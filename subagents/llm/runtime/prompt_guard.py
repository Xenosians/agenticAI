"""Keeps message text from becoming a local model's chat-template control tokens.

A Hugging Face tokenizer splits its special-token strings (``<|im_start|>``,
``<|im_end|>``, ``</s>``...) out of the rendered prompt and encodes them as real
control tokens. Message content carries retrieved passages, crawled pages,
uploaded documents and tool output; any of them containing such a string could
close the current turn and open a forged system turn.

The guard inserts a WORD JOINER (U+2060, invisible) after the first character of
each such string in message content. The text stays readable, the tokenizer no
longer sees the control string, and ordinary text is unchanged. Only the
template, rendered afterwards by the tokenizer itself, produces control tokens.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

WORD_JOINER = "\u2060"


class ControlTokenGuard:
    """Defuses one tokenizer's control-token strings inside message content."""

    def __init__(self, control_tokens: Iterable[str]):
        # Longest first, so a token that contains a shorter one is matched whole.
        tokens = sorted({token for token in control_tokens if isinstance(token, str) and len(token) > 1},
                        key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, tokens))) if tokens else None

    @classmethod
    def for_tokenizer(cls, tokenizer) -> "ControlTokenGuard":
        """The guard for a tokenizer's own control tokens.

        A tokenizer whose special tokens cannot be read is refused: guessing would
        let forged turns through unnoticed.
        """
        try:
            tokens = set(tokenizer.all_special_tokens)
            decoder = getattr(tokenizer, "added_tokens_decoder", None)
            if isinstance(decoder, dict):
                tokens |= {token.content for token in decoder.values() if getattr(token, "special", False)}
        except Exception as error:
            raise RuntimeError("The model tokenizer's control tokens could not be read") from error
        return cls(tokens)

    def content(self, text: str) -> str:
        if self._pattern is None or not text:
            return text
        return self._pattern.sub(lambda match: match.group()[0] + WORD_JOINER + match.group()[1:], text)

    def messages(self, messages: list[dict]) -> list[dict]:
        """A guarded copy; the caller's messages are never changed."""
        return [{**message, "content": self.content(message["content"])}
                if isinstance(message.get("content"), str) else dict(message)
                for message in messages]
