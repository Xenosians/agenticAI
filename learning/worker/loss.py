"""Answer-only causal loss that never materialises logits for prompt positions.

The same idea as Janus's ``learning.training.causal_loss`` (the worker cannot import it), extended to
padded batches: only the positions from the earliest answer start in the batch to the end are scored.
On a 150k-token vocabulary this is the difference between gigabytes of logits for a long system prompt
and a few hundred megabytes, which is what lets a 6 GB laptop GPU or a CPU worker train specialists.
"""
from __future__ import annotations


def answer_only_loss(model, inputs, *, num_items_in_batch=None):
    import torch
    import torch.nn.functional as functional

    model_inputs = dict(inputs)
    labels = model_inputs.pop("labels")
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("every row needs at least one answer token")
    first = int(supervised.float().argmax(dim=1).min().item())
    if first == 0:
        raise ValueError("an answer cannot start at the first position; the prompt must come first")
    keep = labels.shape[1] - first + 1          # the position before the first answer token predicts it
    outputs = model(**model_inputs, logits_to_keep=keep)
    logits = outputs.logits[:, :-1, :].float()
    targets = labels[:, first:].to(logits.device)
    denominator = int(supervised.sum().item()) if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.item()
    loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
                                    reduction="sum") / denominator
    return loss, outputs


def answer_only_trainer(base):
    class AnswerOnlyTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True     # the window's answer-token count reaches compute_loss
            self._loss_shifts_labels = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = answer_only_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            return (loss, outputs) if return_outputs else loss

    return AnswerOnlyTrainer
