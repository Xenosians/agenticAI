from config import Settings
from services.atlassian import build_atlassian_admin_service


def main() -> int:
    settings = Settings()
    service = build_atlassian_admin_service(settings)

    print("Atlassian J1 Preflight")
    print("======================")

    status = service.credential_status()
    print()
    print("Credential status:")
    print("  Admin API key configured:", status["admin_api_key_configured"])
    print("  Jira basic token configured:", status["jira_basic_configured"])
    print("  Default org configured:", status["default_org_configured"])
    print("  Secret values exposed:", status["secret_values_exposed"])

    if not status["admin_api_key_configured"]:
        print()
        print("ATLASSIAN_ADMIN_API_KEY is not configured.")
        return 2

    print()
    print("Organizations:")
    organizations = service.list_organizations(limit=10)
    if not organizations["ok"]:
        print("  FAILED:", organizations.get("error"))
        return 1

    for item in organizations["organizations"]:
        print(" ", item.get("id"), "-", item.get("name"))

    if status["default_org_configured"]:
        print()
        print("Workspaces in configured default org:")
        workspaces = service.list_workspaces(limit=10)
        if not workspaces["ok"]:
            print("  FAILED:", workspaces.get("error"))
            return 1
        for item in workspaces["workspaces"]:
            print(" ", item.get("id"), "-", item.get("name"), "-", item.get("host_url"))

    print()
    print("J1 Atlassian read-only preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
