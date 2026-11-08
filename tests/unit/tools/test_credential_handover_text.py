"""A result that saved a temporary password says how its requester hands it over.

The new person's own mailbox does not exist yet, so the password is never mailed
to it: the person who asked reveals it once on the request in Janus.
"""
import pytest

from tools.account.catalog import format_account_create_result, format_password_reset_result
from tools.mailbox.catalog import format_mailbox_result

HANDOVER = "Reveal it once on this request in Janus"
SECRET = "Never-Shown-Secret!42"

RESULTS = [
    (format_account_create_result, {"ok": True, "user_id": "jdoe", "email": "jdoe@itsm.test",
                                    "credential_id": "cred", "temporary_password": SECRET}),
    (format_password_reset_result, {"ok": True, "user_id": "jdoe", "credential_id": "cred",
                                    "temporary_password": SECRET}),
    (format_mailbox_result, {"ok": True, "operation": "mailbox_create", "email": "box@itsm.test",
                             "credential_persisted": True, "smtp_verified": True, "imap_verified": True,
                             "delivery_verified": True, "credential_id": "cred", "temporary_password": SECRET}),
]


@pytest.mark.parametrize(("formatter", "result"), RESULTS)
def test_a_saved_temporary_password_is_handed_over_in_janus(formatter, result):
    text = formatter(result)

    assert HANDOVER in text
    assert "first sign-in" in text
    assert SECRET not in text


def test_an_account_without_a_saved_credential_promises_no_hand_over():
    text = format_account_create_result({"ok": True, "user_id": "jdoe", "email": "jdoe@itsm.test"})

    assert HANDOVER not in text
