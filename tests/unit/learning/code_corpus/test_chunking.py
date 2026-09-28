from learning.code_corpus.chunking import (
    split_semantic_source,
)


def test_python_is_split_by_symbols_before_tokenization():
    text = """\
import json

def alpha(value):
    return value + 1

class Worker:
    def run(self, value):
        return value * 2
"""

    chunks = split_semantic_source(
        text=text,
        language="python",
        max_chars=8000,
    )

    symbols = {
        item.symbol
        for item in chunks
    }

    assert "alpha" in symbols
    assert "Worker" in symbols
    assert "Worker.run" in symbols
    assert "<module-segment-1>" in symbols


def test_python_methods_preserve_parent_symbol():
    text = """\
class Worker:
    def run(self):
        return True
"""

    chunks = split_semantic_source(
        text=text,
        language="python",
    )

    method = next(
        item
        for item in chunks
        if item.symbol == "Worker.run"
    )

    assert method.parent_symbol == "Worker"
    assert method.semantic_depth == 2


def test_large_symbol_is_split_only_after_semantic_split():
    body = "\n".join(
        f"    value_{index} = {index}"
        for index in range(100)
    )

    text = (
        "def large():\n"
        + body
        + "\n    return value_99\n"
    )

    chunks = split_semantic_source(
        text=text,
        language="python",
        max_chars=300,
        overlap_lines=2,
    )

    parts = [
        item
        for item in chunks
        if item.symbol.startswith(
            "large#part"
        )
    ]

    assert len(parts) > 1
    assert all(
        item.parent_symbol == "large"
        for item in parts
    )


def test_markdown_is_split_by_heading():
    text = """\
Preamble.

# Routing

Details.

## Grounding

More details.
"""

    chunks = split_semantic_source(
        text=text,
        language="markdown",
    )

    symbols = [
        item.symbol
        for item in chunks
    ]

    assert "Routing" in symbols
    assert "Grounding" in symbols
