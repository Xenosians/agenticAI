from __future__ import annotations

from config import get_settings
from services.palo_alto import build_palo_alto_service, palo_alto_integration_status


def main() -> None:
    settings = get_settings()
    status = palo_alto_integration_status(settings)

    print("Palo Alto E2E Preflight")
    print("=======================")
    print("configured:", status["configured"])
    print("host:", status["host"])
    print("verify_tls:", status["verify_tls"])
    print("mode:", status["mode"])

    service = build_palo_alto_service(settings)
    if service is None:
        raise SystemExit("Palo Alto integration is not configured.")

    try:
        info = service.system_info()
    finally:
        service.close()

    print("reachable:", bool(info.get("ok")))
    print("hostname:", info.get("hostname"))
    print("model:", info.get("model"))
    print("pan_os:", info.get("sw_version"))


if __name__ == "__main__":
    main()
