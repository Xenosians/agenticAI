# Agentic AI Runtime

Python/FastAPI service for the Agentic Developer Hub and ITSM platform. It turns chat requests into specialist proposals, validates them against server-side policy, and executes supported tools after any required approval.

This is an evolving local-first application. Model output is a proposal, never authorization.

## Features

- Hub and specialist routing for directory accounts, access, tickets, Jira, developer workspaces, knowledge, assets, and networking.
- Compact routing for selected model connections: local catalog search narrows the prompt, while the full registry still validates every proposed tool action.
- Typed tools for LDAP/Jira integrations, workspace inspection, builds/tests, and governed Git changes. Provider availability depends on configuration.
- Durable approvals that retain the exact proposed action, plus a completion outbox for reliable delivery to Phoenix.
- Local Transformers models, quantization profiles, LoRA adapters, and serialized GPU scheduling.
- Per-account model connections: compatible chat-completion APIs, including Groq; installed local profiles; and Pi/OpenCode inference adapters.
- Shared provider usage queues, editable quota settings, and verified reasoning controls for supported connections.
- Validated uploads of local safetensors model folders, private reference datasets, and reviewed JSONL training data.
- Local full-text retrieval for reference datasets, with source and trust metadata kept separate from execution authority.
- Training/evaluation infrastructure for curated datasets, candidate adapters, behavioral checks, and promotion gates.

## How the repositories fit together

```text
agenticFrontend (Nim/Karax browser)
    -> agenticBackend / itsm_backend (Phoenix API + SurrealDB)
        -> agenticaiPersonal / itsm-agent (this service)
            -> policy, approvals, tools and configured providers
```

Phoenix owns authentication, chats, and durable user-facing jobs. This service owns inference and guarded execution. The browser communicates with Phoenix only. Internal service calls use a shared token.

## Stack

Python, FastAPI, Pydantic, PyTorch/Transformers, MCP, HTTPX, and SQLite for local approval, completion, and extension storage. LDAP and Jira are optional provider integrations; networking adapters expose read-only OpenWrt and PAN-OS capabilities.

## Local setup

Use a Python environment with a PyTorch build compatible with your hardware. Local model memory requirements depend on the configured checkpoint and quantization. Model weights and provider credentials are not bundled.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements-dev.txt fastapi uvicorn
```

FastAPI and Uvicorn currently need explicit installation; they are not listed in `requirements.txt`. Dependencies are not fully locked, so a fresh environment may need version alignment.

Create a private `.env` in this repository, using [config/settings.py](config/settings.py) as the configuration reference. There is currently no tracked environment template here. Configure at least:

| Setting | Purpose |
| --- | --- |
| `PHOENIX_BASE_URL` | Backend origin, commonly `http://127.0.0.1:4000` |
| `ITSM_INTERNAL_JOB_TOKEN` | Same private token as Phoenix |
| `MODEL_PROFILES` | JSON profiles with actual model paths and runtime settings |
| `HUB_MODEL_KEY` | Hub profile; default `hub-main` |
| Provider settings | Enable only the directory, ticketing, knowledge, or network services you intend to use |

Profiles must cover the enabled [agent definitions](subagents/agents). Configure repository/workspace mappings before using developer tools. Remote model selection does not remove the need to configure the local runtime, including the networking model.

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Inspect `http://127.0.0.1:8000/ready` after startup. Run one AI process per extension/model storage root. Start Phoenix separately with matching service settings.

Optional training dependencies:

```bash
python -m pip install -r requirements-training.txt
```

## Models, harnesses, and datasets

The frontend lets signed-in users save, probe, and select their own model connections. Local models remain the default. API credentials are encrypted in the owner-scoped extension store; hosted services still require suitable account access, credentials, and quota.

Pi and OpenCode provide inference only: their native tool execution is disabled. Application policy, approvals, and ToolGateway continue to govern tools. The networking specialist keeps its existing model and receives no additional private dataset context.

The MVP focuses on completing supported ITSM actions through this guarded tool path. Explicit inference stages keep broad catalog routing local and apply separate routing/reasoning budgets to external adapters. See [compact routing and verification](docs/compact-routing.md) for current evidence and remaining acceptance work.

Reference uploads accept text/Markdown; training uploads accept reviewed JSONL. Importing training data does not start training or promote a model. Private model/dataset jobs are excluded from automatic shared training capture. See [user extensions](docs/user-extensions.md) for formats, upload limits, allowed origins, storage, and API commands.

## Tests

```bash
python -m pytest -q tests/unit
```

Real-model, GPU, provider, and container integration checks require their own configured environments. A successful connection probe confirms that request only; it is not a tool-use or model-quality evaluation.

The [tool acceptance matrix](docs/tool-acceptance-matrix.md) inventories all 65 tools and separates proposal, approval, execution, and durable app-flow evidence. Fixtures alone never count as model or tool success.

## Current boundaries

- This is an MVP with model-dependent routing and argument accuracy; configured integrations are not proof of live connectivity.
- Dataset retrieval uses local full-text search, not embeddings or an external vector database.
- The n8n/workflow surface validates manifests only. It neither runs workflows nor registers webhooks.
- Uploaded models reject custom executable code and pickle weights. Format admission does not guarantee inference quality or sufficient GPU memory.
- Ambiguous mutation outcomes require reconciliation; they are not automatically retried.
- Multi-worker sharing of extension/upload storage and a unified three-repository deployment are not implemented.

## More detail

- [User models, harnesses, datasets, and workflow validation](docs/user-extensions.md)
- [Compact routing, inference stages, and ITSM verification](docs/compact-routing.md)
- [Cross-repository acceptance checks](docs/cross-repo-acceptance.md)
- [Architecture and trust model](ARCHITECTURE.md)
- [Security policy](SECURITY.md)
- [Capability learning architecture](docs/CAPABILITY_LEARNING_ARCHITECTURE.md)
- [Training methods](docs/training-methods.md)
- [PAN-OS lab setup](docs/palo_alto_lab_setup.md)
