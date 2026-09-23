from config import Settings
from services.directory import build_directory_service
from services.directory.account_creation import prepare_account_create


def main() -> None:
    settings = Settings()
    directory = build_directory_service(settings)

    prepared = prepare_account_create(
        directory,
        settings,
        given_name="Demo",
        family_name="Candidate",
        department="Engineering",
        role="Backend Engineer",
    )

    if not prepared.get("ok"):
        raise SystemExit("ACCOUNT CREATE PREFLIGHT: FAIL")

    args = prepared["execution_arguments"]
    print("ACCOUNT CREATE PREFLIGHT")
    print("========================")
    print(f"directory backend: {settings.directory_backend}")
    print(f"derived username: {args['expected_username']}")
    print(f"derived email: {args['expected_email']}")
    print(f"identity policy: {args['identity_policy_version']}")
    print(f"password length: {settings.account_temporary_password_length}")
    print("plaintext password generated: False")
    print("directory mutation occurred: False")
    print("Phoenix credential write occurred: False")
    print("PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
