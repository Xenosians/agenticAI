"""Compatibility re-export. New code should import learning.integrations."""

from learning.integrations.runtime_hooks import (
    DEFAULT_CONTEXT_RECORDS_PATH,
    ContinualLearningRuntimeHook,
    ContinualLearningRuntimeHooks,
)

DEFAULT_CONTEXT_EVENTS_PATH = DEFAULT_CONTEXT_RECORDS_PATH

__all__ = [
    "DEFAULT_CONTEXT_EVENTS_PATH",
    "DEFAULT_CONTEXT_RECORDS_PATH",
    "ContinualLearningRuntimeHook",
    "ContinualLearningRuntimeHooks",
]
