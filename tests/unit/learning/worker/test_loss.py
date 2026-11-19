"""The worker's answer-only loss: exact, and it never builds logits for prompt positions."""
import pytest

torch = pytest.importorskip("torch")

from learning.worker.loss import answer_only_loss  # noqa: E402


class Recorder(torch.nn.Module):
    """A causal LM stand-in that records how many positions it was asked to score."""

    def __init__(self, vocabulary=11, width=6):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocabulary, width)
        self.head = torch.nn.Linear(width, vocabulary)
        self.kept = []

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        self.kept.append(logits_to_keep)
        hidden = self.embed(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        return type("Out", (), {"logits": self.head(hidden)})()


def full_loss(model, input_ids, labels):
    logits = model(input_ids=input_ids).logits
    shifted, targets = logits[:, :-1, :], labels[:, 1:]
    return torch.nn.functional.cross_entropy(shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
                                             ignore_index=-100, reduction="sum")


def test_the_answer_only_loss_equals_the_full_causal_loss_and_scores_only_the_answer():
    model = Recorder()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 6, 7, 8]])
    loss, _ = answer_only_loss(model, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                                       "labels": labels})
    assert model.kept == [4]                         # three answer tokens plus the position predicting the first
    assert torch.allclose(loss, full_loss(model, input_ids, labels) / 3)


def test_a_batch_with_padding_keeps_each_rows_answer_and_divides_by_the_window_count():
    model = Recorder()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, -100, 4, 5, -100], [-100, -100, -100, -100, 5, 6]])
    loss, _ = answer_only_loss(model, {"input_ids": input_ids, "attention_mask": (labels != -100) | (input_ids > 0),
                                       "labels": labels}, num_items_in_batch=8)
    assert model.kept == [4]                         # from the earliest answer start in the batch to the end
    assert torch.allclose(loss, full_loss(model, input_ids, labels) / 8)


def test_a_row_without_any_answer_token_is_refused():
    model = Recorder()
    with pytest.raises(ValueError, match="answer"):
        answer_only_loss(model, {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[-100] * 3])})


def test_a_per_device_count_from_data_parallel_is_summed_once():
    """transformers expands the batch's item count to one entry per GPU under DataParallel (Kaggle has two
    T4s); the loss must divide by the count, not fail on a two-element tensor (live: 'a Tensor with 2
    elements cannot be converted to Scalar')."""
    model = Recorder()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, -100, 4, 5, 6]])
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}
    expected, _ = answer_only_loss(model, batch, num_items_in_batch=torch.tensor(3))
    expanded, _ = answer_only_loss(model, batch, num_items_in_batch=torch.tensor(3).unsqueeze(0).expand(2, -1))
    assert torch.allclose(expanded, expected)
