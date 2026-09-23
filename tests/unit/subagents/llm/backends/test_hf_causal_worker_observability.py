from types import (
    SimpleNamespace,
)

import pytest
import torch

import subagents.llm.backends.hf_causal_worker as hf_module

from subagents.llm.backends.hf_causal_worker import (
    HFCausalWorkerBackend,
)


class FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        prompt,
        return_tensors,
    ):
        assert return_tensors == "pt"

        return {
            "input_ids":
                torch.tensor(
                    [[
                        10,
                        11,
                        12,
                    ]]
                ),
        }

    def decode(
        self,
        token_ids,
        skip_special_tokens,
    ):
        assert skip_special_tokens is True
        assert token_ids.tolist() == [
            21,
            22,
        ]

        return (
            '```json {"name":"ticket_get","arguments":{}} ```'
        )


class FakeModel:
    def __init__(
        self,
    ):
        self._parameter = (
            torch.nn.Parameter(
                torch.zeros(
                    1
                )
            )
        )

        self._embedding = (
            SimpleNamespace(
                weight=(
                    self._parameter
                )
            )
        )

    def eval(
        self,
    ):
        return self

    def parameters(
        self,
    ):
        yield self._parameter

    def get_input_embeddings(
        self,
    ):
        return self._embedding

    def generate(
        self,
        **kwargs,
    ):
        return (
            torch.tensor(
                [[
                    10,
                    11,
                    12,
                    21,
                    22,
                ]]
            )
        )


def test_hf_causal_worker_reports_exact_generated_tokens_and_closes(
    monkeypatch,
    tmp_path,
):
    tokenizer = (
        FakeTokenizer()
    )

    model = (
        FakeModel()
    )

    monkeypatch.setattr(
        hf_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    monkeypatch.setattr(
        hf_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    backend = (
        HFCausalWorkerBackend(
            model_path=tmp_path,
            model_load_kwargs={},
        )
    )

    result = (
        backend
        .generate_observed(
            [
                {
                    "role": "user",
                    "content": "Get KAN-3",
                }
            ],
            max_new_tokens=8,
        )
    )

    assert (
        result.text
        == '{"name":"ticket_get","arguments":{}}'
    )

    assert (
        result.generated_tokens
        == 2
    )

    assert (
        result.first_token_seconds
        is None
    )

    backend.close()

    assert (
        backend._closed
        is True
    )

    assert not hasattr(
        backend,
        "model",
    )

    assert not hasattr(
        backend,
        "tokenizer",
    )

    with pytest.raises(
        RuntimeError,
        match="closed",
    ):
        backend.generate(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )
