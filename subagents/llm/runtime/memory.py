from __future__ import annotations

import gc

import torch


def release_unused_accelerator_memory(
) -> None:
    """
    Reclaim accelerator memory that is no longer referenced by
    live Python objects.

    This function deliberately does NOT evict live models.

    Semantics:

        Python objects that are no longer reachable
            -> collected

        CUDA allocator blocks that are no longer occupied
            -> returned to the CUDA driver

        live tensors / live model backends
            -> untouched

    This matters before automatic Transformers device placement.

    `device_map="auto"` reasons about currently available device
    memory. PyTorch may retain already-unused CUDA blocks in its
    allocator cache, which can otherwise make a later model load
    appear to have substantially less available VRAM than the
    process really has.

    Cleanup is best-effort. Failure to empty the allocator cache
    must not itself make the model lifecycle unusable.
    """

    gc.collect()

    if not (
        torch.cuda.is_available()
    ):
        return

    try:
        torch.cuda.empty_cache()

    except Exception as exc:
        print(
            "[MODEL] CUDA cache cleanup "
            f"failed: {exc!r}"
        )