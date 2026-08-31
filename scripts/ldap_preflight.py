import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import get_bool_env, get_env
from services.directory.ldap import LdapDirectoryService


def check_config() -> dict:
    required = [
        "AD_HOST",
        "AD_BIND_PASSWORD",
        "AD_BASE_DN",
        "AD_TEST_USER",
    ]

    missing = [
        name
        for name in required
        if not get_env(name)
    ]

    # Accept either naming convention.
    bind_identity = (
        get_env("AD_BIND_USER")
        or get_env("AD_BIND_DN")
    )

    if not bind_identity:
        missing.append(
            "AD_BIND_USER or AD_BIND_DN"
        )

    return {
        "ok": not missing,
        "missing": missing,
    }


def check_tcp(
    host: str,
    port: int,
) -> dict:
    try:
        with socket.create_connection(
            (host, port),
            timeout=5,
        ):
            return {
                "ok": True,
                "message": (
                    f"TCP connection to "
                    f"{host}:{port} succeeded."
                ),
            }

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def main():
    print("LDAP Preflight")
    print("==============")
    print()

    # --------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------

    config_result = check_config()

    if not config_result["ok"]:
        print("[FAIL] Configuration")
        print(
            "Missing:",
            ", ".join(
                config_result["missing"]
            ),
        )
        return

    print("[OK] Configuration present")

    host = get_env("AD_HOST")

    use_ssl = get_bool_env(
        "AD_USE_SSL",
        True,
    )

    default_port = (
        636
        if use_ssl
        else 389
    )

    port = int(
        get_env(
            "AD_PORT",
            str(default_port),
        )
        or default_port
    )

    test_user = get_env(
        "AD_TEST_USER"
    )

    bind_identity = (
        get_env("AD_BIND_USER")
        or get_env("AD_BIND_DN")
    )

    print(
        f"Host: {host}:{port}"
    )

    print(
        f"SSL: {use_ssl}"
    )

    print(
        f"Bind identity: {bind_identity}"
    )

    print(
        f"Test user: {test_user}"
    )

    # Never print the password.
    print()

    # --------------------------------------------------
    # TCP / NETWORK
    # --------------------------------------------------

    tcp_result = check_tcp(
        host,
        port,
    )

    if not tcp_result["ok"]:
        print("[FAIL] TCP connection")
        print(
            tcp_result["error"]
        )
        return

    print(
        "[OK]",
        tcp_result["message"],
    )

    # --------------------------------------------------
    # LDAP ADAPTER INITIALIZATION
    # --------------------------------------------------

    try:
        directory = (
            LdapDirectoryService()
        )

    except Exception as exc:
        print(
            "[FAIL] LDAP configuration"
        )
        print(exc)
        return

    print(
        "[OK] LDAP adapter initialized"
    )

    # --------------------------------------------------
    # LDAP BIND
    # --------------------------------------------------

    connection = None

    try:
        connection = (
            directory._connect()
        )

        print(
            "[OK] LDAP bind succeeded"
        )

    except Exception as exc:
        print(
            "[FAIL] LDAP bind"
        )
        print(exc)
        return

    finally:
        if connection is not None:
            connection.unbind()

    # --------------------------------------------------
    # TEST ACCOUNT LOOKUP
    # --------------------------------------------------

    result = directory.account_status(
        test_user
    )

    if not result.get(
        "ok",
        False,
    ):
        print(
            "[FAIL] Test account lookup"
        )

        print(
            result.get(
                "error",
                result,
            )
        )

        return

    print(
        "[OK] Test account lookup"
    )

    print()
    print("Result:")
    print(result)

    print()

    print(
        "LDAP read-only preflight passed."
    )


if __name__ == "__main__":
    main()