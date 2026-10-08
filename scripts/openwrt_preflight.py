"""Verify real read-only OpenWrt calls and inspect the session's write permission."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings
from services.openwrt import build_openwrt_service


def main():
    service = build_openwrt_service(Settings())
    if service is None:
        raise SystemExit("OpenWrt is not configured.")
    try:
        board = service.system_info()
        interface = service.interface_status("lan")
        firewall = service.firewall_config()
        login = service._rpc("0" * 32, "session", "login", {
            "username": service.config.username, "password": service.config.password})
        permission = service._rpc(login["ubus_rpc_session"], "session", "access", {
            "scope": "ubus", "object": "uci", "function": "set"})
        if permission.get("access") is not False:
            raise RuntimeError("API account must not have UCI write permission.")
        report = {"schema": "openwrt-preflight.v1", "provider": "openwrt",
            "release": board["data"]["release"], "interface_read": interface["ok"],
            "firewall_read": firewall["ok"], "uci_write_allowed": False,
            "dataplane_verified": False}
        print(json.dumps(report, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
