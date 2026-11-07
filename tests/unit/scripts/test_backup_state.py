"""Backup of Janus state: retention, SQLite snapshots and the restore check, offline."""
import io
import json
import sqlite3
import tarfile
from datetime import datetime, timedelta, timezone

import pytest

from scripts import backup_state as backup


def make_db(path, rows=3):
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT)")
        db.executemany("INSERT INTO item(name) VALUES (?)", [(f"n{i}",) for i in range(rows)])
    return path


def test_retention_keeps_recent_daily_and_one_per_older_week():
    now = datetime(2026, 11, 30, 2, 0, tzinfo=timezone.utc)
    names = [f"janus-{(now - timedelta(days=d)):%Y%m%dT%H%M%SZ}.tar" for d in range(0, 120)]
    keep = backup.retained(names, now=now, daily=14, weekly=8)
    ages = sorted((now - backup.archive_time(name)).days for name in keep)
    assert ages[:14] == list(range(14))           # every one of the last 14 days
    assert len(keep) == 14 + 8                     # plus one per week for 8 older weeks
    assert max(ages) < 14 + 8 * 7
    assert backup.retained(["janus-not-a-date.tar", "other.tar"], now=now, daily=1, weekly=1) == set()


def test_retention_never_keeps_an_archive_older_than_the_weekly_window():
    # With sparse history the weekly slots must not reach back past the window.
    now = datetime(2026, 11, 8, 2, 30, tzinfo=timezone.utc)
    stamps = [now, now - timedelta(hours=1), now - timedelta(days=1), now - timedelta(days=20),
              now - timedelta(days=200)]
    names = [f"janus-{stamp:%Y%m%dT%H%M%SZ}.tar" for stamp in stamps]
    keep = backup.retained(names, now=now, daily=14, weekly=8)
    assert keep == {names[0], names[2], names[3]}  # newest per day, the 20-day-old weekly; not 200 days


def test_a_live_wal_database_is_snapshotted_consistently(tmp_path):
    source = make_db(tmp_path / "live.sqlite3", rows=5)
    with sqlite3.connect(source) as writer:            # an open writer with uncheckpointed WAL pages
        writer.execute("INSERT INTO item(name) VALUES ('late')")
        writer.commit()
        data = backup.sqlite_snapshot(source)
    copy = tmp_path / "copy.sqlite3"
    copy.write_bytes(data)
    with sqlite3.connect(copy) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT count(*) FROM item").fetchone()[0] == 6


def test_an_archive_records_what_it_holds_and_verifies(tmp_path):
    db = make_db(tmp_path / "a.sqlite3")
    key = tmp_path / "credentials.key"
    key.write_bytes(b"k" * 32)
    archive = tmp_path / "out" / "janus-20261108T020000Z.tar"
    manifest = backup.write_archive(archive, database_export=b"-- export\nDEFINE TABLE item;\n",
                                    sqlite_files={"extensions/extensions.sqlite3": db},
                                    plain_files={"extensions/credentials.key": key})
    assert manifest["files"]["database.surql"]["bytes"] > 0
    assert set(manifest["files"]) == {"database.surql", "extensions/extensions.sqlite3", "extensions/credentials.key"}
    report = backup.verify_archive(archive)
    assert report["ok"] is True and report["sqlite"] == {"extensions/extensions.sqlite3": "ok"}
    assert oct(archive.stat().st_mode & 0o777) == "0o600"


def test_a_damaged_archive_fails_verification(tmp_path):
    db = make_db(tmp_path / "a.sqlite3")
    archive = tmp_path / "janus-20261108T020000Z.tar"
    backup.write_archive(archive, database_export=b"-- export\n", sqlite_files={"a.sqlite3": db}, plain_files={})
    with tarfile.open(archive) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    members["a.sqlite3"] = members["a.sqlite3"][:100] + b"\x00" * 50 + members["a.sqlite3"][150:]
    with tarfile.open(archive, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    report = backup.verify_archive(archive)
    assert report["ok"] is False and "a.sqlite3" in report["mismatched"]


def test_an_empty_or_failed_database_export_is_refused():
    with pytest.raises(backup.BackupError, match="export"):
        backup.check_export(b"")
    with pytest.raises(backup.BackupError, match="export"):
        backup.check_export(b'{"code":403,"details":"Authentication failed"}')
    # An error document that happens to quote a statement is still an error, not an export.
    with pytest.raises(backup.BackupError, match="export"):
        backup.check_export(b'{"code":400,"details":"Parse error near DEFINE TABLE"}')
    backup.check_export(b"-- ------------------------------\n-- OPTION\nDEFINE TABLE job TYPE ANY;\n")


def test_the_manifest_never_holds_secret_values(tmp_path):
    key = tmp_path / "credentials.key"
    key.write_bytes(b"super-secret-key-material-000000")
    archive = tmp_path / "janus-20261108T020000Z.tar"
    manifest = backup.write_archive(archive, database_export=b"-- export\n", sqlite_files={},
                                    plain_files={"extensions/credentials.key": key})
    assert b"super-secret" not in json.dumps(manifest).encode()
