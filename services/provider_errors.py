"""Typed provider read errors shared by adapters and tool result mapping."""


class ProviderResourceNotFound(RuntimeError):
    """A provider completed a read but did not contain the exact target."""
