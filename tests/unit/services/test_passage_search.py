"""Passage search answers questions, and returns nothing rather than unrelated text."""
from services.retrieval.fts import FtsRecord, query_terms, search_passages

HANDBOOK = {
    "freeze": "Payroll cluster maintenance. The quarterly change freeze for the payroll cluster starts on "
              "the second Tuesday at 18:00 UTC and lasts 72 hours.",
    "backup": "Backups of the payroll database run nightly to the vault-b pool and are retained for 35 days.",
    "vpn": "When a remote user cannot connect to the corporate VPN, confirm the account is not locked. "
           "Error code 809 usually means a home router is blocking the tunnel.",
    "password": "Password and account policy. Passwords expire every 90 days. After five failed sign-in "
                "attempts an account is locked for 15 minutes.",
    "wifi": "Wi-Fi network. Guests connect to CORP-GUEST, which gives internet access for 24 hours.",
}


def search(question, limit=3):
    records = [FtsRecord(key, text) for key, text in HANDBOOK.items()]
    return search_passages(records, question, limit=limit).identifiers


def test_a_question_finds_the_passage_that_answers_it():
    assert search("When does the payroll change freeze start?")[0] == "freeze"
    assert search("What does VPN error 809 mean?")[0] == "vpn"


def test_word_forms_match_their_stem():
    assert search("freezes") == ("freeze",)
    assert search("How often does a password expire?")[0] == "password"


def test_a_hyphenated_word_matches_its_joined_spelling():
    assert search("guest wifi") == ("wifi",)
    assert search("signin attempts")[0] == "password"


def test_numbers_are_search_terms():
    assert search("error 809") == ("vpn",)


def test_grammatical_words_alone_find_nothing():
    assert search("How do I do it, and what is that?") == ()


def test_sharing_one_word_with_an_unrelated_question_is_not_a_match():
    assert search("What is the parental leave policy?") == ()
    assert search("How do I rotate the TLS certificate on the load balancer?") == ()


def test_a_question_outside_the_corpus_finds_nothing():
    assert search("How do I configure BGP peering on the core switch?") == ()


def test_results_are_bounded_and_report_truncation():
    records = [FtsRecord(str(i), f"VPN gateway note {i}") for i in range(5)]
    matches = search_passages(records, "vpn gateway", limit=2)
    assert len(matches.identifiers) == 2 and matches.truncated


def test_a_kind_filter_limits_results():
    records = [FtsRecord("kb", "VPN gateway setup", kind="knowledge"),
               FtsRecord("rb", "VPN gateway restart", kind="runbook")]
    assert search_passages(records, "vpn gateway", limit=5, kind="runbook").identifiers == ("rb",)


def test_a_passage_is_found_by_the_title_of_its_page():
    records = [FtsRecord("handover", "The weekly handover happens on Monday at 10:00.", title="On-call handbook"),
               FtsRecord("leases", "Static leases give a printer a fixed address.", title="DHCP and DNS")]
    assert search_passages(records, "When is the on-call handover?", limit=3).identifiers == ("handover",)


def test_letters_and_digits_match_either_spelling():
    spaced = [FtsRecord("radio", "Enable the 5 GHz radio and choose channel 36.")]
    joined = [FtsRecord("radio", "Enable the 5GHz radio and choose channel 36.")]
    assert search_passages(spaced, "Which channel for the 5GHz radio?", limit=3).identifiers == ("radio",)
    assert search_passages(joined, "Which channel for the 5 GHz radio?", limit=3).identifiers == ("radio",)
    assert search_passages([FtsRecord("p1", "A P1 incident is acknowledged within 15 minutes.")],
                           "P1 acknowledgement", limit=3).identifiers == ("p1",)


def test_quantifiers_are_grammatical_words():
    assert query_terms("How much on-call pay do engineers receive?") == ["oncall", "pay", "engineers", "receive"]
    assert query_terms("How many failed sign-ins lock an account?") == ["failed", "signins", "lock", "account"]


def test_a_passage_that_only_mentions_the_topic_is_not_returned_beside_the_answer():
    records = [FtsRecord("answer", "Restart the print spooler service from the Services console."),
               FtsRecord("topic", "The print spooler service writes its log to the event viewer.")]
    assert search_passages(records, "How do I restart the print spooler service?", limit=3).identifiers == (
        "answer",)
