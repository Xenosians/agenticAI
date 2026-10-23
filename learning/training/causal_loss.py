"""Exact target-suffix causal loss without allocating prompt-position logits.

Imports stay lightweight; training dependencies are loaded only when computing a
loss or constructing the Trainer subclass. No checkpoint or runtime is loaded.
"""

from __future__ import annotations

import math
from numbers import Real


def tail_only_causal_loss(model, inputs, *, num_items_in_batch=None):
    """Return loss and outputs for one masked-prompt/target-suffix record.

    Trainer supplies the supervised-token count for the entire accumulation
    window, including a short final window. A per-record mean here would weight
    unequal targets incorrectly and bypass Trainer's accumulation scaling.
    Outside training, an absent count means the mean over this record's targets.
    """
    import torch
    import torch.nn.functional as functional

    model_inputs = dict(inputs)
    labels = model_inputs.pop("labels")
    if not torch.is_tensor(labels) or labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Tail-only causal loss requires a batch size of one.")

    supervised = labels[0].ne(-100)
    positions = supervised.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("Tail-only causal loss requires supervised target tokens.")
    first_target = int(positions[0].item())
    if first_target <= 0 or not bool(supervised[first_target:].all().item()):
        raise ValueError(
            "Tail-only causal loss requires a contiguous supervised suffix "
            "after a masked prompt."
        )

    target_count = int(positions.numel())
    denominator = target_count if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denominator):
        if denominator.ndim != 0:
            raise ValueError("Supervised-token count must be a scalar.")
        denominator = denominator.item()
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, Real)
        or not math.isfinite(denominator)
        or denominator != int(denominator)
        or denominator < target_count
    ):
        raise ValueError(
            "Supervised-token count must be an integer at least as large "
            "as this record's target count."
        )

    # The predecessor predicts the first target; the final returned logit is
    # beyond the record and is excluded from the causal loss.
    logits_to_keep = target_count + 1
    outputs = model(**model_inputs, logits_to_keep=logits_to_keep)
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[:2] != (1, logits_to_keep):
        raise ValueError(
            f"Unexpected tail-logit shape: {tuple(logits.shape)}; "
            f"expected (1, {logits_to_keep}, vocabulary)."
        )
    predictions = logits[:, :-1, :].contiguous().float()
    targets = labels[:, first_target:].contiguous().to(predictions.device)
    loss = functional.cross_entropy(
        predictions.view(-1, predictions.shape[-1]),
        targets.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / denominator
    return loss, outputs


def tail_only_trainer_class(base_trainer_class):
    """Bind the suffix loss to the installed Trainer's accumulation contract."""

    class TailOnlyCausalTrainer(base_trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # This custom loss consumes the window count, regardless of whether
            # the wrapped model's forward signature advertises loss kwargs.
            self.model_accepts_loss_kwargs = True
            self._loss_shifts_labels = True

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss, outputs = tail_only_causal_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
            # Match Trainer.compute_loss's correction when the count was gathered
            # across devices but gradient synchronization averages their losses.
            if self.args.average_tokens_across_devices and num_items_in_batch is not None:
                scale = self.accelerator.num_processes
                parallelism = getattr(self.accelerator, "parallelism_config", None)
                if parallelism is not None:
                    scale //= parallelism.tp_size
                loss *= scale if self.args.n_gpu <= 1 else self.args.n_gpu
            return (loss, outputs) if return_outputs else loss

    return TailOnlyCausalTrainer
