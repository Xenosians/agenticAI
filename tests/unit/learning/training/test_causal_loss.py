from types import SimpleNamespace

import pytest

from learning.training.causal_loss import tail_only_causal_loss, tail_only_trainer_class


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _record(torch, prompt, target):
    return {
        "input_ids": torch.tensor([prompt + target]),
        "attention_mask": torch.ones(1, len(prompt + target), dtype=torch.long),
        "labels": torch.tensor([[-100] * len(prompt) + target]),
    }


def test_suffix_loss_matches_full_causal_loss_and_retains_inputs(torch):
    torch.manual_seed(11)
    full_logits = torch.randn(1, 7, 23, requires_grad=True)
    record = _record(torch, [2, 3, 4, 5], [6, 7, 8])
    calls = []

    def model(**inputs):
        calls.append(inputs)
        return SimpleNamespace(logits=full_logits[:, -inputs["logits_to_keep"]:, :])

    loss, outputs = tail_only_causal_loss(model, record)
    reference = torch.nn.functional.cross_entropy(
        full_logits[:, :-1, :].reshape(-1, 23), record["labels"][:, 1:].reshape(-1)
    )
    torch.testing.assert_close(loss, reference)
    assert "labels" in record
    assert "labels" not in calls[0]
    assert calls[0]["logits_to_keep"] == 4
    assert outputs.logits.shape == (1, 4, 23)


@pytest.mark.parametrize("count", [6, 6.0, "tensor"])
def test_window_count_normalizes_summed_target_loss(torch, count):
    record = _record(torch, [2, 3], [4, 5, 6])
    model = lambda **_: SimpleNamespace(logits=torch.zeros(1, 4, 11))
    count = torch.tensor(6) if count == "tensor" else count
    loss, _ = tail_only_causal_loss(model, record, num_items_in_batch=count)
    expected = torch.log(torch.tensor(11.0)) / 2
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("count", [0, -1, 2, True, 3.5, float("nan"), float("inf"), "3"])
def test_invalid_window_count_is_rejected_before_forward(torch, count):
    def model(**_):
        pytest.fail("An invalid token count must not reach the model.")

    with pytest.raises(ValueError, match="Supervised-token count"):
        tail_only_causal_loss(
            model, _record(torch, [2, 3], [4, 5, 6]), num_items_in_batch=count
        )


def test_nonscalar_window_count_is_rejected(torch):
    with pytest.raises(ValueError, match="must be a scalar"):
        tail_only_causal_loss(
            None, _record(torch, [2, 3], [4, 5, 6]),
            num_items_in_batch=torch.tensor([3, 3]),
        )


@pytest.mark.parametrize("labels, reason", [
    ([[-100, -100, -100]], "supervised target tokens"),
    ([[2, 3, 4]], "contiguous supervised suffix"),
    ([[-100, 3, -100, 4]], "contiguous supervised suffix"),
    ([[-100, 3, 4, -100]], "contiguous supervised suffix"),
    ([[-100, 3], [-100, 4]], "batch size of one"),
    ([-100, 3, 4], "batch size of one"),
])
def test_invalid_target_layout_is_rejected_before_forward(torch, labels, reason):
    def model(**_):
        pytest.fail("An invalid target layout must not reach the model.")

    with pytest.raises(ValueError, match=reason):
        tail_only_causal_loss(model, {"labels": torch.tensor(labels)})


def test_model_must_return_the_requested_suffix_shape(torch):
    model = lambda **_: SimpleNamespace(logits=torch.zeros(1, 7, 11))
    with pytest.raises(ValueError, match="Unexpected tail-logit shape"):
        tail_only_causal_loss(model, _record(torch, [2, 3], [4, 5, 6]))


def test_trainer_return_outputs_and_gathered_count_scaling(torch):
    class Base:
        def __init__(self):
            self.args = SimpleNamespace(average_tokens_across_devices=True, n_gpu=0)
            self.accelerator = SimpleNamespace(num_processes=4, parallelism_config=None)

    trainer = tail_only_trainer_class(Base)()
    model = lambda **_: SimpleNamespace(logits=torch.zeros(1, 4, 11))
    record = _record(torch, [2, 3], [4, 5, 6])
    loss, outputs = trainer.compute_loss(model, record, True, num_items_in_batch=12)
    torch.testing.assert_close(loss, torch.log(torch.tensor(11.0)))
    assert outputs.logits.shape == (1, 4, 11)
    trainer.accelerator.parallelism_config = SimpleNamespace(tp_size=2)
    loss = trainer.compute_loss(model, record, num_items_in_batch=6)
    torch.testing.assert_close(loss, torch.log(torch.tensor(11.0)))


def test_actual_trainer_gradients_match_full_causal_reference_and_partial_window(torch, tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from learning.training.specialist_sft import SpecialistCollator

    rows = [
        _record(torch, [5, 6, 7, 8], [9, 10]),
        _record(torch, [11, 12, 13], [14, 15, 16, 17, 18]),
        _record(torch, [19, 20], [21, 22, 23]),
    ]

    def build_model():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = transformers.BloomForCausalLM(transformers.BloomConfig(
                vocab_size=64, hidden_size=16, n_layer=1, n_head=2,
                hidden_dropout=0.0, attention_dropout=0.0,
            ))
            # The custom loss, rather than model.forward, consumes the count.
            model.accepts_loss_kwargs = False
            return peft.get_peft_model(model, peft.LoraConfig(
                r=2, lora_alpha=4, target_modules=["query_key_value"],
                task_type="CAUSAL_LM", init_lora_weights=False,
            ))

    def reference_gradients(records):
        model = build_model()
        summed = 0.0
        count = 0
        for row in records:
            outputs = model(input_ids=row["input_ids"], attention_mask=row["attention_mask"])
            summed = summed + torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1, :].reshape(-1, 64),
                row["labels"][:, 1:].reshape(-1), reduction="sum",
            )
            count += int(row["labels"].ne(-100).sum())
        (summed / count).backward()
        return {name: param.grad.clone() for name, param in model.named_parameters()
                if param.requires_grad and param.grad is not None}

    captured = []

    class Capture(transformers.TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            captured.append({name: param.grad.detach().clone()
                             for name, param in model.named_parameters()
                             if param.requires_grad and param.grad is not None})

    class OrderedTrainer(tail_only_trainer_class(transformers.Trainer)):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(
                self.train_dataset if train_dataset is None else train_dataset
            )

    trainer = OrderedTrainer(
        model=build_model(),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path), report_to="none", use_cpu=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            max_steps=2, learning_rate=0.0, max_grad_norm=0.0,
            remove_unused_columns=False, save_strategy="no", logging_strategy="no",
            disable_tqdm=True,
        ),
        train_dataset=[{key: value[0].tolist() for key, value in row.items()} for row in rows],
        data_collator=SpecialistCollator(pad_token_id=0), callbacks=[Capture()],
    )
    assert trainer.model_accepts_loss_kwargs is True
    trainer.train()
    assert len(captured) == 2
    for observed, records in zip(captured, (rows[:2], rows[2:]), strict=True):
        expected = reference_gradients(records)
        assert observed.keys() == expected.keys()
        assert any(bool(grad.abs().gt(0).any()) for grad in observed.values())
        for name in expected:
            torch.testing.assert_close(observed[name], expected[name], atol=1e-6, rtol=1e-4)
    assert all(param.grad is None for param in trainer.model.parameters() if not param.requires_grad)
