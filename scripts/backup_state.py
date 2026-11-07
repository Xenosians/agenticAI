"""Back up Janus state into one archive, keep a bounded history and verify it.

What is saved, at the locations the services are configured with:

- the SurrealDB database, exported over HTTP with the backend's own credentials
  (SURREALDB_* in the shared .env);
- the AI owner store (EXTENSION_STORAGE_ROOT: extension SQLite and its
  credentials.key) and, from the AI service's Settings, the approval and
  completion-outbox databases, the asset inventory and the directory CA.

Run it from the AI checkout with the AI service's Python, so Settings resolves
exactly as it does for the running service.

SQLite files are copied with SQLite's online backup API, so a database in use
(WAL mode, open writer) is captured consistently; raw file copies of a live WAL
database are not. Each archive carries a manifest (sizes and SHA-256 of every
member, never their contents) and is verified after writing: every member's
digest is checked and every SQLite copy must pass `PRAGMA integrity_check`.

The archive contains credentials.key, which decrypts stored provider keys. Keep
the backup directory restricted to the owner.

Usage: python scripts/backup_state.py --dest <backup directory> [--daily 14] [--weekly 8]
       python scripts/backup_state.py --verify <archive>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARCHIVE_PATTERN = re.compile(r"^janus-(\d{8}T\d{6}Z)\.tar$")
MANIFEST = "manifest.json"


class BackupError(RuntimeError):
    """A backup step failed; the message never includes secrets or stored content."""


def archive_time(name: str) -> datetime | None:
    match = ARCHIVE_PATTERN.match(Path(name).name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def retained(names, *, now: datetime, daily: int, weekly: int) -> set[str]:
    """Archives to keep: the newest of each of the last `daily` days, then the newest of each older week."""
    dated: list[tuple[datetime, str]] = sorted(
        ((stamp, name) for name in names if (stamp := archive_time(name)) is not None), reverse=True)
    keep: dict[tuple, str] = {}
    for stamp, name in dated:
        age_days = (now.date() - stamp.date()).days
        if age_days < daily:
            keep.setdefault(("day", stamp.date()), name)
    cutoff = now.date() - timedelta(days=daily)
    oldest = cutoff - timedelta(weeks=weekly)
    for stamp, name in dated:
        # One archive per ISO week, only for the `weekly` weeks before the daily window;
        # sparse history must not stretch the window back in time.
        if oldest <= stamp.date() < cutoff:
            keep.setdefault(("week", stamp.isocalendar()[:2]), name)
    return set(keep.values())


def sqlite_snapshot(path: Path) -> bytes:
    """A consistent copy of a (possibly live) SQLite database, as bytes."""
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "copy.sqlite3"
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            copy = sqlite3.connect(target)
            try:
                source.backup(copy)
                copy.execute("PRAGMA journal_mode=DELETE")
            finally:
                copy.close()
        finally:
            source.close()
        return target.read_bytes()


def check_export(data: bytes) -> None:
    """Refuse an empty export or an error document instead of saving it as a backup."""
    head = data[:4096].decode("utf-8", errors="replace").lstrip()
    if not head or head.startswith("{") or "DEFINE" not in data[:200_000].decode("utf-8", errors="replace"):
        raise BackupError("The database export is empty or an error response.")


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> dict:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    tar.addfile(info, io.BytesIO(data))
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_archive(path: Path, *, database_export: bytes, sqlite_files: dict[str, Path],
                  plain_files: dict[str, Path]) -> dict:
    """Write the archive atomically (temporary name, then rename) and return its manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "janus-backup.v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "files": {}, "sqlite": sorted(sqlite_files)}
    partial = path.with_name(path.name + ".partial")
    with tarfile.open(partial, "w") as tar:
        manifest["files"]["database.surql"] = _add(tar, "database.surql", database_export)
        for name, source in sorted(sqlite_files.items()):
            manifest["files"][name] = _add(tar, name, sqlite_snapshot(source))
        for name, source in sorted(plain_files.items()):
            manifest["files"][name] = _add(tar, name, Path(source).read_bytes())
        _add(tar, MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode())
    os.chmod(partial, 0o600)
    os.replace(partial, path)
    return manifest


def _read(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise BackupError(f"Archive member {member.name} is not a regular file.")
    return handle.read()


def verify_archive(path: Path) -> dict:
    """Check every member against the manifest and every SQLite copy's integrity."""
    report = {"ok": False, "mismatched": [], "missing": [], "sqlite": {}}
    with tarfile.open(path) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        if MANIFEST not in members:
            report["missing"].append(MANIFEST)
            return report
        manifest = json.loads(_read(tar, members[MANIFEST]))
        for name, expected in manifest["files"].items():
            if name not in members:
                report["missing"].append(name)
                continue
            data = _read(tar, members[name])
            if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                report["mismatched"].append(name)
                continue
            if name in manifest.get("sqlite", []):
                with tempfile.TemporaryDirectory() as scratch:
                    copy = Path(scratch) / "check.sqlite3"
                    copy.write_bytes(data)
                    with sqlite3.connect(copy) as db:
                        report["sqlite"][name] = db.execute("PRAGMA integrity_check").fetchone()[0]
    report["ok"] = not report["mismatched"] and not report["missing"] and all(
        value == "ok" for value in report["sqlite"].values())
    return report


def export_database(env: dict) -> bytes:
    import httpx

    url = (env.get("SURREALDB_URL") or "").rstrip("/")
    if not url:
        raise BackupError("SURREALDB_URL is not configured.")
    response = httpx.get(url + "/export", timeout=300, trust_env=False,
                         auth=(env.get("SURREALDB_USERNAME") or "", env.get("SURREALDB_PASSWORD") or ""),
                         headers={"surreal-ns": env.get("SURREALDB_NAMESPACE") or "",
                                  "surreal-db": env.get("SURREALDB_DATABASE") or "",
                                  "Accept": "application/octet-stream"})
    if response.status_code != 200:
        raise BackupError(f"The database export failed with HTTP {response.status_code}.")
    check_export(response.content)
    return response.content


def state_files(ai_root: Path, env: dict) -> tuple[dict[str, Path], dict[str, Path]]:
    """The AI state to save, at the locations the AI service itself is configured with.

    Paths come from the service's Settings (relative paths resolve against the AI
    checkout, as for the running service) and the extension store's root variable;
    files that do not exist are skipped.
    """
    from config.settings import Settings  # the AI checkout is on sys.path (see main)

    settings = Settings()

    def located(path) -> Path | None:
        if path is None:
            return None
        path = Path(path)
        return path if path.is_absolute() else ai_root / path

    extensions = Path(env.get("EXTENSION_STORAGE_ROOT") or Path.home() / ".local/share/agentic-extensions")
    sqlite_candidates = {
        "extensions/extensions.sqlite3": extensions / "extensions.sqlite3",
        "ai/approvals.sqlite3": located(settings.approval_store_path),
        "ai/completion_outbox.sqlite3": located(settings.completion_outbox_path),
        "ai/assets/inventory.sqlite3": located(settings.asset_sqlite_path),
    }
    plain_candidates = {
        "extensions/credentials.key": extensions / "credentials.key",
        "ai/trust/directory-ca.pem": located(settings.ad_tls_ca_file),
    }
    return ({name: path for name, path in sqlite_candidates.items() if path is not None and path.is_file()},
            {name: path for name, path in plain_candidates.items() if path is not None and path.is_file()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Back up Janus state into one verified archive.")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--env-file", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--daily", type=int, default=14)
    parser.add_argument("--weekly", type=int, default=8)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        report = verify_archive(args.verify)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 1
    if args.dest is None:
        parser.error("--dest is required")

    from dotenv import dotenv_values

    env = {key: value for key, value in dotenv_values(args.env_file).items() if value is not None}
    ai_root = Path(__file__).resolve().parents[1]
    if str(ai_root) not in sys.path:
        sys.path.insert(0, str(ai_root))
    os.chdir(ai_root)  # relative Settings paths resolve as they do for the running service
    now = datetime.now(timezone.utc)
    archive = args.dest / f"janus-{now:%Y%m%dT%H%M%SZ}.tar"
    try:
        sqlite_files, plain_files = state_files(ai_root, env)
        manifest = write_archive(archive, database_export=export_database(env),
                                 sqlite_files=sqlite_files, plain_files=plain_files)
    except (BackupError, OSError, sqlite3.Error) as error:
        print(json.dumps({"ok": False, "error": type(error).__name__, "detail": str(error)[:200]}))
        return 1
    report = verify_archive(archive)
    if not report["ok"]:
        archive.rename(archive.with_name(archive.name + ".failed"))
        print(json.dumps({"ok": False, "verify": report}))
        return 1
    existing = [path.name for path in args.dest.glob("janus-*.tar")]
    keep = retained(existing, now=now, daily=args.daily, weekly=args.weekly)
    removed = [name for name in existing if name not in keep]
    for name in removed:
        (args.dest / name).unlink()
    print(json.dumps({"ok": True, "archive": archive.name, "members": len(manifest["files"]),
                      "bytes": archive.stat().st_size, "kept": len(keep), "removed": len(removed)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
