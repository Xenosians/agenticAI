# Import real knowledge and runbooks

The existing `knowledge_search`, `knowledge_get`, and `runbook_get` tools can use the local provider instead of the mock examples. The provider searches and returns actual operator-imported text with its source, license, content digest, and pinned snapshot digest. Importing does not train a model or grant retrieved text permission to execute tools.

Prepare a UTF-8 JSONL file with one reviewed document per line. Each record must have `document_id`, `kind` (`knowledge` or `runbook`), `title`, `summary`, `content`, and optional `tags`. Admission also requires an accurate `source`, `license`, `visibility: "public"`, and `reviewed: true`. Do not change a private document's classification to make it pass: these tools currently have no per-document authorization context and admit public documents only. Secret scanner findings are refused.

```sh
python scripts/import_knowledge.py \
  --source /path/to/operator-reviewed-public-documents.jsonl \
  --output-dir /path/to/knowledge-snapshots
```

The importer validates the source through the production provider, publishes its exact bytes atomically under `<sha256>.jsonl`, and verifies the stored snapshot. It prints only counts and the configuration needed to select it:

```dotenv
KNOWLEDGE_BACKEND=local
KNOWLEDGE_SNAPSHOT_PATH=/absolute/path/to/knowledge-snapshots/<sha256>.jsonl
KNOWLEDGE_SNAPSHOT_SHA256=<sha256>
```

Apply the returned values to the local deployment environment and restart the AI service after current jobs finish. The command does not edit the environment, seed example answers, or overwrite previous pins. Re-importing identical bytes reuses the snapshot; reviewed changes create a new pin while preserving the old file. Existing content with a mismatched digest is refused. The import limit is 32 MiB per snapshot; larger dataset ingestion belongs to a separate pipeline.

After configuration, test with a phrase and document/runbook IDs actually present in your imported dataset. A search with no match and an unknown ID must remain empty/not found. Import success proves validated storage and provider read-back; real LLM routing and browser acceptance are separate checks.
