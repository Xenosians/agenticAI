"""Reference datasets are split into passages that keep one topic together.

A dataset admitted before passage chunking keeps its fixed-window chunks, so its
citations reopen the exact text they named.
"""
import pytest

from services.extensions.retrieval import PASSAGE_CHUNKING, dataset_chunks
from services.extensions.store import ExtensionStore

SECTIONS = [
    "Payroll cluster maintenance. The quarterly change freeze starts on the second Tuesday at 18:00 UTC.",
    "VPN troubleshooting. Error code 809 usually means a home router is blocking the tunnel.",
    "Printer support. Restart the print spooler service when print jobs stay pending.",
]


def test_paragraphs_stay_whole_and_each_passage_is_bounded():
    text = "\n\n".join(SECTIONS * 12)
    chunks = dataset_chunks(text, PASSAGE_CHUNKING)
    assert all(len(chunk) <= 1200 for chunk in chunks)
    assert all(any(section in chunk for section in SECTIONS) for chunk in chunks)
    assert not any(chunk.startswith(" ") or chunk.endswith(" ") for chunk in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_a_heading_or_question_line_stays_with_the_text_after_it():
    text = ("VPN troubleshooting\n\nError code 809 usually means a home router is blocking the tunnel.\n\n"
            "How do I reset my password?\nUse the self-service portal and your employee number.\n\n"
            "Printer support\nRestart the print spooler service when jobs stay pending.\n"
            "Toner orders go through the facilities portal.")
    assert dataset_chunks(text, PASSAGE_CHUNKING) == [
        "VPN troubleshooting\n\nError code 809 usually means a home router is blocking the tunnel.",
        "How do I reset my password?\nUse the self-service portal and your employee number.",
        "Printer support\nRestart the print spooler service when jobs stay pending.\n"
        "Toner orders go through the facilities portal.",
    ]


def test_a_crawled_page_keeps_its_title_and_address_with_its_first_section():
    page = ("# Router setup\nSource: https://docs.example.test/router\n\nConnect the WAN port to the modem.\n"
            "Firmware\nDownload the release that matches the board.")
    assert dataset_chunks(page, PASSAGE_CHUNKING) == [
        "# Router setup\nSource: https://docs.example.test/router\n\nConnect the WAN port to the modem.",
        "Firmware\nDownload the release that matches the board.",
    ]


def test_a_list_stays_with_the_sentence_that_introduces_it():
    text = ("Restart the print spooler when print jobs stay pending.\n- Clear stuck jobs older than an hour\n"
            "- Check that the queue PRN-3F-01 is online\nWi-Fi network\nGuests use CORP-GUEST for 24 hours.")
    assert dataset_chunks(text, PASSAGE_CHUNKING) == [
        "Restart the print spooler when print jobs stay pending.\n- Clear stuck jobs older than an hour\n"
        "- Check that the queue PRN-3F-01 is online",
        "Wi-Fi network\nGuests use CORP-GUEST for 24 hours.",
    ]


def test_a_paragraph_longer_than_a_passage_is_split_at_sentences():
    sentence = "The VPN gateway accepts connections on UDP port 1194 from managed laptops. "
    chunks = dataset_chunks(sentence * 40, PASSAGE_CHUNKING)
    assert len(chunks) > 1 and all(len(chunk) <= 1200 for chunk in chunks)
    assert all(chunk.rstrip().endswith(".") for chunk in chunks)


def test_a_text_without_breaks_is_still_cut_into_bounded_passages():
    chunks = dataset_chunks("x" * 5000, PASSAGE_CHUNKING)
    assert all(0 < len(chunk) <= 1200 for chunk in chunks) and "".join(chunks) == "x" * 5000


def test_the_original_windows_are_unchanged_for_older_datasets():
    text = "a" * 4000
    assert dataset_chunks(text) == [text[0:2000], text[1800:3800], text[3600:4000]]


def test_new_datasets_record_passage_chunking(tmp_path):
    store = ExtensionStore(tmp_path)
    row = store.dataset("alice", name="Handbook", source="Owned", license="Owned", purpose="rag",
                        reviewed=True, text="\n\n".join(SECTIONS))
    assert row["chunking"] == PASSAGE_CHUNKING
    hits = store.search("alice", row["id"], "VPN error 809")
    assert [hit["content"] for hit in hits] == [SECTIONS[1]]
    assert store.resolve("alice", row["id"], hits[0]["chunk"], row["sha256"])["content"] == SECTIONS[1]


def test_a_section_of_a_crawled_page_is_found_by_the_page_title(tmp_path):
    store = ExtensionStore(tmp_path)
    text = ("# On-call handbook\nSource: https://handbook.example.test/on-call\n\n"
            "The on-call engineer acknowledges pages within 5 minutes.\n"
            "Handover\nThe weekly handover happens on Monday at 10:00.\n\n"
            "# DHCP and DNS\nSource: https://docs.example.test/dhcp\n\n"
            "Handover of leases\nLeases are handed over to the backup router at midnight.")
    row = store.dataset("alice", name="Crawled", source="https://handbook.example.test", license="Owned",
                        purpose="rag", reviewed=True, text=text)
    hits = store.search("alice", row["id"], "When is the on-call handover?")
    assert [hit["content"] for hit in hits] == ["Handover\nThe weekly handover happens on Monday at 10:00."]


def test_a_dataset_saved_before_passage_chunking_resolves_its_old_citations(tmp_path):
    store = ExtensionStore(tmp_path)
    text = "\n\n".join(SECTIONS)
    row = store.dataset("alice", name="Handbook", source="Owned", license="Owned", purpose="rag",
                        reviewed=True, text=text)
    # The record as it was stored before passage chunking existed.
    stored = store.get("alice", row["id"])
    stored.pop("chunking")
    stored.pop("credential_configured")
    with store.connect() as db:
        import json
        db.execute("UPDATE resource SET data=? WHERE owner=? AND id=?", (json.dumps(stored), "alice", row["id"]))
    assert store.resolve("alice", row["id"], 1, row["sha256"])["content"] == text[:2000]
    assert store.search("alice", row["id"], "VPN error 809")[0]["content"] == text[:2000]


def test_an_unknown_chunking_is_refused(tmp_path):
    with pytest.raises(ValueError):
        dataset_chunks("text", "unknown-chunking")
