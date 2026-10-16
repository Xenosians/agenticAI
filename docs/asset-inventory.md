# Persistent asset inventory

The existing `asset_get`, `asset_search`, `asset_assign`, and `asset_unassign` tools can use a local SQLite inventory. Records and assignments survive MCP/API restarts. This is an application-managed inventory, with no external CMDB synchronization.

## Enable

Configure the AI service and restart it after provisioning inventory:

```dotenv
ASSET_BACKEND=sqlite
ASSET_SQLITE_PATH=.runtime/assets/inventory.sqlite3
```

Relative paths resolve against the AI repository. Windows and WSL path forms are supported. Use a dedicated persistent database file and keep its directory private and backed up. An explicit path is required; in-memory databases are rejected. The existing `mock` backend remains available for tests and demos, and is the default. The previously accepted but unimplemented `local` value is replaced by the explicit `sqlite` backend.

The SQLite inventory starts empty. It never seeds the mock assets or discovers inventory automatically. Selecting this provider adds no model tools, authority, approval bypass, or directory-account provisioning.

## Import inventory

Prepare an operator-reviewed JSON array of complete records using the fields below. The provider is always `sqlite`; do not include a `provider` field. Unknown fields, duplicate JSON field names, duplicate normalized asset IDs, and inconsistent ownership/status are rejected.

| Field | Required | Format |
| --- | --- | --- |
| `asset_id` | Yes | Visible text, at most 100 characters; normalized to uppercase |
| `name` | Yes | Visible text, at most 200 characters |
| `asset_type` | Yes | Visible text, at most 100 characters |
| `status` | Yes | `available`, `in_use`, `repair`, or `retired` |
| `owner_id` | For `in_use` | Visible text, at most 100 characters; normalized to lowercase |
| `serial_number`, `platform`, `location` | No | Visible text, at most 200 characters, or `null` |

An `available` asset must have no owner. A repair/retired record may retain its recorded owner, but tool-driven assignment changes remain blocked until an operator updates its inventory status. This provider records a supplied owner identifier; it does not create or verify directory identities.

Run these commands from the AI repository, with the same provider settings as the service:

```bash
python scripts/asset_inventory.py /path/to/reviewed-inventory.json --validate-only
python scripts/asset_inventory.py /path/to/reviewed-inventory.json
```

Validation-only mode does not create/open the database. Imports accept 1–10000 records in at most 8 MiB of JSON. The entire batch is validated before any insert. One transaction imports all records or none; an existing ID rejects the whole batch by default.

For an intentional operator update of supplied IDs, use `--replace`. This replaces every field of each supplied record and can change its assignment. Other records remain untouched. The import command is an operator interface, never an LLM capability.

## Tool behavior

- Lookups normalize asset IDs and read the persisted record.
- Search combines optional text, owner, type, and status filters. Text matching is literal and case insensitive, including Unicode case folding. Results are ordered by asset ID, limited to 1–25, and indicate truncation.
- Assignment uses a write transaction. An asset owned by another user is denied; explicitly unassign it before assigning a different owner. Assigning the same owner again succeeds without changing state.
- Unassignment clears the owner and changes `in_use` to `available`. Repeating an already completed unassignment succeeds without changing state.
- Repair/retired assignment changes are denied. Storage failures return errors; they never report a simulated success.

ToolGateway, the semantic guard, and existing approvals still govern mutation calls. SQLite does not change which caller may access an asset. Local and selected outbound inference adapters use these same application tools.

## Verify

```bash
python -m pytest -q tests/unit/services/test_sqlite_assets.py tests/unit/tools/assets/test_assets.py tests/unit/config/test_configuration_boundaries.py
```

These tests use disposable synthetic records and verify persisted readback, reopen behavior, competing writers, filters and limits, atomic imports, and registration of all four existing MCP tools. This is provider/tool evidence; successful model routing and the complete Phoenix/browser lifecycle need separate live acceptance checks.
