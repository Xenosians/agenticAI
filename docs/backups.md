# Backups

`scripts/backup_state.py` saves Janus state into one archive and verifies it.
A deployment schedules it; on the development machine a systemd user timer,
`janus-backup.timer`, runs it nightly at 02:30 (a run missed while the machine
was off happens at the next start) and appends a one-line JSON result per run
to `backup.log` in the backup directory.

## What an archive holds

| Member | Source |
| --- | --- |
| `database.surql` | SurrealDB export over HTTP, with the backend's `SURREALDB_*` account |
| `extensions/extensions.sqlite3`, `extensions/credentials.key` | The AI owner store (`EXTENSION_STORAGE_ROOT`): models, skills, plugins, sites, encrypted provider keys and the key that decrypts them |
| `ai/approvals.sqlite3`, `ai/completion_outbox.sqlite3`, `ai/assets/inventory.sqlite3` | The AI service's approval store, completion outbox and asset inventory (paths from its Settings) |
| `ai/trust/directory-ca.pem` | The directory CA (`AD_TLS_CA_FILE`) |
| `manifest.json` | Size and SHA-256 of every member, never contents |

SQLite databases are copied with SQLite's online backup API, so a database in
use is captured consistently. An archive is written under a temporary name and
renamed only when complete; it is then verified (every digest, and
`PRAGMA integrity_check` on every SQLite copy). An archive that fails
verification is renamed `*.failed` and kept out of rotation. An empty export or
an error document from the database is refused rather than saved.

Rotation keeps the newest archive of each of the last 14 days and one per week
for the 8 weeks before that (`--daily`, `--weekly`).

**The archive contains `credentials.key`**, which decrypts stored provider
keys. Keep the backup directory readable only by the owner and do not copy
archives to shared storage unencrypted.

## Commands

```bash
# one backup now (from the AI checkout, with the AI service's Python)
python scripts/backup_state.py --dest <backup directory>
# check an archive
python scripts/backup_state.py --verify <backup directory>/janus-<time>.tar
# the timer
systemctl --user list-timers janus-backup.timer
systemctl --user start janus-backup.service
```

## Restore

Restore into a stopped stack; never import into a database the backend is
writing to.

1. Stop the services: `systemctl --user stop janus.target`.
2. Check the archive with `--verify` and extract it to a private directory.
3. Database: create the namespace and database if they do not exist
   (`DEFINE NAMESPACE …; USE NS …; DEFINE DATABASE …;`; SurrealDB 3 does not
   create them on import), then `POST /import` with `database.surql` and the
   `surreal-ns`/`surreal-db` headers. Importing into a non-empty database
   merges; restore into an empty one.
4. AI state: put `extensions/*` back under `EXTENSION_STORAGE_ROOT` (owner-only
   permissions; `credentials.key` mode 600) and the `ai/*` files at the paths
   the AI Settings name.
5. Start the services: `systemctl --user start janus.target`.

Practise the restore on a throwaway SurrealDB container (steps 2–3 against
it, then compare table row counts with the live database) after changing the
schema or the backup script; an untested backup is not a backup.
