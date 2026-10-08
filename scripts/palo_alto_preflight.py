from __future__ import annotations

import sys
import time

from pathlib import Path


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


from config import get_settings
from services.palo_alto import (
    build_palo_alto_service,
    palo_alto_integration_status,
)


def main() -> None:
    settings = get_settings()

    status = palo_alto_integration_status(
        settings
    )

    print("Palo Alto E2E Preflight")
    print("=======================")
    print(
        "configured:",
        status["configured"],
    )
    print(
        "host:",
        status["host"],
    )
    print(
        "verify_tls:",
        status["verify_tls"],
    )
    print(
        "mode:",
        status["mode"],
    )

    service = build_palo_alto_service(
        settings
    )

    if service is None:
        raise SystemExit(
            "Palo Alto integration is not configured. Set PALO_ALTO_BASE_URL and "
            "PALO_ALTO_API_KEY for a real PAN-OS instance in the AI environment. "
            "See docs/palo_alto_lab_setup.md."
        )

    started = time.perf_counter()
    try:
        info = service.system_info()
    finally:
        print("provider_elapsed_seconds:", round(time.perf_counter() - started, 3))
        service.close()

    print(
        "reachable:",
        bool(info.get("ok")),
    )
    print(
        "hostname:",
        info.get("hostname"),
    )
    print(
        "model:",
        info.get("model"),
    )
    print(
        "pan_os:",
        info.get("sw_version"),
    )


if __name__ == "__main__":
    main()
