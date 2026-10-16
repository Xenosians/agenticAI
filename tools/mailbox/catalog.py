from typing import Any


def format_mailbox_result(result: dict[str, Any]) -> str:
    if (result.get("ok") and result.get("operation") == "mailbox_create"
            and all(result.get(flag) is True for flag in
                    ("credential_persisted", "smtp_verified", "imap_verified", "delivery_verified"))):
        return f"Local mailbox {result.get('email')} created and authenticated SMTP/IMAP delivery verified. Its credential is encrypted; the .test address has no public mail reachability."
    if result.get("ok") and result.get("operation") == "mailbox_status":
        return (
            f"Local mailbox {result.get('email')}: credentials {'present' if result.get('credential_exists') else 'absent'}, "
            f"IMAP storage {'present' if result.get('mailbox_exists') else 'absent'}. "
            "This presence check does not verify authenticated delivery or public reachability."
        )
    return result.get("error") or "Mailbox operation did not complete."


def format_mailbox_approval(arguments: dict[str, Any]) -> str:
    return (
        f"Create the local SMTP/IMAP mailbox {arguments.get('email')}, generate and encrypt its credential, "
        "and send one verification message to itself. This .test address cannot receive public GitHub or Atlassian verification mail."
    )


MAILBOX_TOOLS = {
    "mailbox_create": {
        "description": "Create a real local SMTP/IMAP mailbox at the configured .test lab domain, separate from directory identity creation. Requires the user's exact email and approval. Trusted code generates and encrypts the password, then verifies authenticated local self-delivery. This mailbox cannot receive public Internet email or public GitHub/Atlassian verification messages.",
        "risk": "high", "requires_approval": True, "grounded_arguments": ["email"],
        "parameters": {"email": {"type": "str", "description": "Exact lowercase .test mailbox email explicitly supplied by the user. Never invent an address or use a public domain."}},
        "result_formatter": format_mailbox_result,
        "approval_formatter": format_mailbox_approval,
    },
    "mailbox_status": {
        "description": "Read whether credentials and IMAP storage exist for an exact local .test mailbox email. Does not create an account, authenticate, send mail, reveal secrets, or prove public email reachability.",
        "risk": "read", "requires_approval": False, "grounded_arguments": ["email"],
        "condition_fields": ["credential_exists", "mailbox_exists"],
        "parameters": {"email": {"type": "str", "description": "Exact lowercase .test mailbox email explicitly supplied by the user."}},
        "result_formatter": format_mailbox_result,
    },
}
