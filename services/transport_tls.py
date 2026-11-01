"""Operator-selected TLS trust, independent of HTTP proxy environment policy."""
from __future__ import annotations

import os
from pathlib import Path
import ssl


def configured_bundle() -> str | None:
    """The deployment's CA bundle file (ITSM_HTTP_CA_BUNDLE), or None for the system trust store."""
    bundle = os.environ.get("ITSM_HTTP_CA_BUNDLE")
    if bundle is None:
        return None
    try:
        path = Path(bundle)
        if path.is_absolute() and path.is_file():
            return str(path)
    except (OSError, ValueError):
        pass
    raise ValueError("transport_ca_bundle_invalid")


def configured_context() -> ssl.SSLContext | None:
    """Use a mounted public/private CA bundle without weakening verification.

    HTTPX ignores SSL_CERT_FILE when trust_env=False. An explicit deployment
    setting lets HTTP and WebSocket callbacks share trust while proxy settings
    and credentials remain with their existing host-owned transports.
    """
    bundle = configured_bundle()
    if bundle is None:
        return None
    try:
        return ssl.create_default_context(cafile=bundle)
    except (OSError, ssl.SSLError):
        raise ValueError("transport_ca_bundle_invalid") from None
