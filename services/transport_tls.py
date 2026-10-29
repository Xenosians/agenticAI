"""Operator-selected TLS trust, independent of HTTP proxy environment policy."""
from __future__ import annotations

import os
from pathlib import Path
import ssl


def configured_context() -> ssl.SSLContext | None:
    """Use a mounted public/private CA bundle without weakening verification.

    HTTPX ignores SSL_CERT_FILE when trust_env=False. An explicit deployment
    setting lets HTTP and WebSocket callbacks share trust while proxy settings
    and credentials remain with their existing host-owned transports.
    """
    bundle = os.environ.get("ITSM_HTTP_CA_BUNDLE")
    if bundle is None:
        return None
    try:
        path = Path(bundle)
        if not path.is_absolute() or not path.is_file():
            raise ValueError
        return ssl.create_default_context(cafile=str(path))
    except (OSError, ValueError, ssl.SSLError):
        raise ValueError("transport_ca_bundle_invalid") from None
