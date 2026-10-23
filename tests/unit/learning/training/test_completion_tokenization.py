"""Training must preserve the complete prompt boundary and assistant target."""

import pytest
import torch

from learning.training.hub_hybrid_qlora import (
    _encode_prompt_and_completion,
    _generation_inputs,
)


class TemplateTokenizer:
    chat_template = "configured native template"

    def __init__(self, prompt_ids, full_ids):
        self.prompt_ids = prompt_ids
        self.full_ids = full_ids

    def apply_chat_template(self, messages, **kwargs):
        ids = self.full_ids if messages[-1]["role"] == "assistant" else self.prompt_ids
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def encode(tokenizer, *, max_length=8):
    return _encode_prompt_and_completion(
        tokenizer=tokenizer,
        prompt_messages=[{"role": "user", "content": "Route this request."}],
        completion='{"delegations":[]}',
        max_length=max_length,
    )


def test_complete_prompt_is_masked_and_complete_target_is_supervised():
    ids, attention, labels = encode(TemplateTokenizer([1, 2, 3], [1, 2, 3, 4, 5]))

    assert ids.tolist() == [[1, 2, 3, 4, 5]]
    assert attention.tolist() == [[1, 1, 1, 1, 1]]
    assert labels.tolist() == [[-100, -100, -100, 4, 5]]


@pytest.mark.parametrize(
    ("prompt_ids", "full_ids"),
    [
        ([1, 2, 3], [1, 2, 7, 4, 5]),
        ([1, 2, 3], [1, 2]),
        ([1, 2], [7, 8, 9]),
        ([], [1, 2]),
    ],
)
def test_rejects_an_unstable_or_missing_prompt_boundary(prompt_ids, full_ids):
    with pytest.raises(ValueError, match="complete prompt prefix"):
        encode(TemplateTokenizer(prompt_ids, full_ids))


def test_rejects_truncation_of_a_supervised_completion():
    # The final token can be the closing JSON delimiter or EOS. It must survive.
    with pytest.raises(ValueError, match="exceeds max_length.*truncate"):
        encode(TemplateTokenizer([1, 2], [1, 2, 3, 4, 5]), max_length=4)


def test_exact_sequence_budget_retains_final_completion_token():
    ids, _, labels = encode(TemplateTokenizer([1, 2], [1, 2, 3, 4]), max_length=4)

    assert ids.tolist() == [[1, 2, 3, 4]]
    assert labels.tolist() == [[-100, -100, 3, 4]]


def test_rejects_a_record_with_no_completion_tokens():
    with pytest.raises(ValueError, match="No completion tokens"):
        encode(TemplateTokenizer([1, 2], [1, 2]))


def test_template_without_generation_flag_keeps_its_complete_assistant_boundary():
    class CommonTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, *, return_tensors, return_dict):
            return super().apply_chat_template(
                messages, return_tensors=return_tensors, return_dict=return_dict
            )

    tokenizer = CommonTokenizer([1, 2, 9], [1, 2, 9, 4, 5])
    ids, _, labels = encode(tokenizer)
    generation_inputs = _generation_inputs(
        tokenizer=tokenizer,
        prompt_messages=[{"role": "user", "content": "Route this request."}],
    )

    assert ids.tolist() == [[1, 2, 9, 4, 5]]
    assert labels.tolist() == [[-100, -100, -100, 4, 5]]
    assert generation_inputs["input_ids"].tolist() == [[1, 2, 9]]


class BatchEncodingTokenizer(TemplateTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        from transformers import BatchEncoding

        return BatchEncoding(super().apply_chat_template(messages, **kwargs))

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("A valid runtime chat template must not fall back.")


def test_transformers_batch_encoding_preserves_runtime_training_template():
    ids, _, labels = encode(BatchEncodingTokenizer([1, 2], [1, 2, 3, 4]))

    assert ids.tolist() == [[1, 2, 3, 4]]
    assert labels.tolist() == [[-100, -100, 3, 4]]


def test_transformers_batch_encoding_preserves_runtime_evaluation_template():
    inputs = _generation_inputs(
        tokenizer=BatchEncodingTokenizer([1, 2], [1, 2, 3, 4]),
        prompt_messages=[{"role": "user", "content": "Route this request."}],
    )

    assert inputs["input_ids"].tolist() == [[1, 2]]


@pytest.mark.parametrize("full_ids", [[1, 2, 3, 4], [1, 2, 3, 4, 9]])
def test_native_common_completed_assistant_has_exactly_one_eos(full_ids):
    from transformers import MistralCommonBackend

    class NativeCommonTokenizer(BatchEncodingTokenizer, MistralCommonBackend):
        chat_template = None
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            if messages[-1]["role"] == "assistant":
                assert kwargs.get("continue_final_message") is True
                assert not kwargs.get("add_generation_prompt")
            return super().apply_chat_template(messages, **kwargs)

    ids, _, labels = encode(NativeCommonTokenizer([1, 2], full_ids))

    assert ids.tolist() == [[1, 2, 3, 4, 9]]
    assert labels.tolist() == [[-100, -100, 3, 4, 9]]


@pytest.mark.parametrize("operation", ["training", "evaluation"])
def test_configured_native_template_failure_never_changes_serialization(operation):
    class BrokenTemplateTokenizer(BatchEncodingTokenizer):
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("Native chat template failed")

    tokenizer = BrokenTemplateTokenizer([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="Native chat template failed"):
        if operation == "training":
            encode(tokenizer)
        else:
            _generation_inputs(
                tokenizer=tokenizer,
                prompt_messages=[{"role": "user", "content": "Route this request."}],
            )


@pytest.mark.parametrize("operation", ["training", "evaluation"])
def test_native_internal_type_error_is_not_retried_without_generation_flag(operation):
    class InternallyBrokenTokenizer(BatchEncodingTokenizer):
        calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            if "add_generation_prompt" in kwargs:
                raise TypeError("Native template implementation defect")
            # Retrying without the flag would hide the real defect.
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = InternallyBrokenTokenizer([1, 2], [1, 2, 3])
    with pytest.raises(TypeError, match="Native template implementation defect"):
        if operation == "training":
            encode(tokenizer)
        else:
            _generation_inputs(
                tokenizer=tokenizer,
                prompt_messages=[{"role": "user", "content": "Route this request."}],
            )

    assert tokenizer.calls == 1


def test_malformed_native_evaluation_inputs_never_fall_back():
    class MalformedTemplateTokenizer(BatchEncodingTokenizer):
        def apply_chat_template(self, *_args, **_kwargs):
            return {}

    with pytest.raises(ValueError, match="named input_ids"):
        _generation_inputs(
            tokenizer=MalformedTemplateTokenizer([1, 2], [1, 2, 3]),
            prompt_messages=[{"role": "user", "content": "Route this request."}],
        )


def test_tokenizer_without_a_native_template_can_use_the_declared_fallback():
    class UntemplatedTokenizer(TemplateTokenizer):
        chat_template = None

        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("No chat template configured")

        def __call__(self, text, **_kwargs):
            ids = self.full_ids if text.endswith('{"delegations":[]}') else self.prompt_ids
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    tokenizer = UntemplatedTokenizer([1, 2], [1, 2, 3, 4])
    ids, _, labels = encode(tokenizer)
    inputs = _generation_inputs(
        tokenizer=tokenizer,
        prompt_messages=[{"role": "user", "content": "Route this request."}],
    )

    assert ids.tolist() == [[1, 2, 3, 4]]
    assert labels.tolist() == [[-100, -100, 3, 4]]
    assert inputs["input_ids"].tolist() == [[1, 2]]
