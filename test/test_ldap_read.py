import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import get_env
from services.directory import get_directory_service


def main():
    backend = (
        get_env(
            "DIRECTORY_BACKEND",
            "mock",
        )
        or "mock"
    ).strip().lower()

    if backend != "ldap":
        print("LDAP test skipped.")
        print(
            "DIRECTORY_BACKEND is not set to 'ldap'."
        )
        return

    test_user = get_env(
        "AD_TEST_USER"
    )

    if not test_user:
        raise RuntimeError(
            "AD_TEST_USER is not configured."
        )

    directory = get_directory_service()

    print(
        "Directory backend:",
        type(directory).__name__,
    )

    print(
        "Testing account:",
        test_user,
    )

    print()

    result = directory.account_status(
        test_user
    )

    print("Account status result:")
    print(result)


if __name__ == "__main__":
    main()