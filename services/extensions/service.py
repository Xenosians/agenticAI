from __future__ import annotations

import base64
import asyncio
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import threading

from services.lua_adapters.model_loading import from_pretrained as lua_from_pretrained

from services.extensions.store import ExtensionStore
from services.extensions.document_ingestion import (
    DocumentIngestionError,
    MAX_BASE64_BYTES,
    extract_document,
)
from services.extensions.transport import request_json, validate_url
from services.extensions.harness import harness_binary, harness_generate
from services.extensions.options import MODEL_OPTION_FIELDS, inference_option_support, networking_inference_options, validate_model_options

_upload_lock = threading.Lock()


@lru_cache
def extension_store():
    return ExtensionStore(Path(os.environ.get("EXTENSION_STORAGE_ROOT", Path.home()/".local/share/agentic-extensions")))



def specialist_roles(settings) -> list[dict]:
    """Registered specialists a model connection can be assigned to."""
    from subagents.core.definitions.loader import load_agent_directory

    agents_dir = getattr(settings, "agents_dir", None)
    if agents_dir is None:
        return []
    return [{"id": agent.name, "description": agent.description, "local_model": agent.model}
            for agent in sorted(load_agent_directory(agents_dir), key=lambda agent: agent.name)]


def workflow_capabilities(settings) -> list[dict]:
    """Public descriptors for explicit workflow review, never provider health."""
    from subagents.core.definitions.loader import load_agent_directory
    from tools.capability_contract import build_capability_contract
    from tools.registry import TOOLS, get_tool
    from subagents.core.orchestration.intent_contract import trusted_success_continuations
    from services.integration_readiness import provider_for_capability

    agents_dir = getattr(settings, "agents_dir", None)
    agents = load_agent_directory(agents_dir) if agents_dir is not None else []
    result = []
    for name, tool in sorted(TOOLS.items()):
        metadata = build_capability_contract(name, tool)
        # Every registered owner is disclosed; a capability can belong to a
        # provider-specific specialist and a provider-neutral fallback.
        owners = sorted(agent.name for agent in agents if name in agent.tools)
        result.append({
            "id": name, "description": tool.get("description", ""),
            "effect": metadata.effect, "permission": metadata.permission,
            "risk": metadata.risk, "requires_approval": metadata.requires_approval,
            "resource_type": metadata.resource_type,
            "agent_names": owners,
            # The provider whose setup this operation needs (see /v1/integrations).
            "provider": provider_for_capability(name),
            "grounded_arguments": list(metadata.grounded_arguments),
            "required_arguments": list(metadata.required_arguments),
            "parameters": {
                field: {key: value for key, value in descriptor.items()
                        if key in {"type", "description", "enum"}}
                for field, descriptor in tool.get("parameters", {}).items()
                if field in metadata.grounded_arguments
            },
            "success_continuations": trusted_success_continuations(
                tool_name=name, tool=tool, tool_lookup=get_tool,
            ),
        })
    return result

def replaced_dataset(payload: dict) -> str | None:
    """The dataset version a new upload replaces; absent or empty means a new dataset."""
    replaces = payload.get("replaces")
    if replaces in (None, ""):
        return None
    if not isinstance(replaces, str) or not re.fullmatch(r"[0-9a-f]{32}", replaces):
        raise ValueError("replaces must name a dataset version")
    return replaces


def dataset_version(row: dict) -> dict:
    """A dataset's version identity: its history, its number and the version it replaced."""
    identity = {"dataset_id": row["id"], "lineage_id": row["lineage_id"], "version": row["version"]}
    if row.get("previous_id"):
        identity.update(previous_id=row["previous_id"], previous_sha256=row["previous_sha256"])
    return identity


def public_resource(row):
    # No dataset bodies, split history, credentials or server filesystem paths in list responses.
    public = {key: value for key, value in row.items()
              if key not in {"api_key", "content", "model_path", "retired_splits"}}
    if public.get("kind") == "model" and public.get("backend") == "http":
        from services.extensions.transport import is_insecure_transport
        public["insecure_transport"] = is_insecure_transport(public.get("url", ""))
    return public


_HOSTED_HARNESS_PROVIDERS = {
    "groq",
    "openrouter",
    "anthropic",
    "google",
    "mistral",
    "openai",
    "xai",
    "cerebras",
    "opencode",
}


def _connection_provider(
    backend: str,
    model: str,
) -> str:
    if backend not in {
        "pi",
        "opencode",
    }:
        return ""

    provider, separator, identifier = (
        model.partition(
            "/"
        )
    )

    if (
        not separator
        or not provider
        or not identifier
    ):
        raise ValueError(
            "Harness models require an explicit "
            "provider/model identifier"
        )

    return provider


def _model_connection_requires_credential(
    backend: str,
    model: str,
    url: str = "",
) -> bool:
    if backend == "http":
        from urllib.parse import urlsplit

        normalized = validate_url(
            url
        )

        hostname = (
            urlsplit(
                normalized
            )
            .hostname
        )

        # Local OpenAI-compatible endpoints remain keyless.
        return hostname != "127.0.0.1"

    if backend in {
        "pi",
        "opencode",
    }:
        return (
            _connection_provider(
                backend,
                model,
            )
            in _HOSTED_HARNESS_PROVIDERS
        )

    return False


def upload_root(owner: str, identity: str):
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid upload identity")
    root = Path(os.environ.get("USER_MODEL_ROOT", Path(__file__).resolve().parents[3]/"models/user-imports"))
    return root / hashlib.sha256(owner.encode()).hexdigest() / identity


def validate_model_bundle(root: Path):
    """Check untrusted metadata before passing any paths to Transformers."""
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Model bundle must contain only regular files")
        if path.suffix != ".json":
            continue
        if path.stat().st_size > 50_000_000:
            raise ValueError("Model configuration exceeds limit")
        config = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(config, dict) and config.get("auto_map"):
            raise ValueError("Custom model/tokenizer code is not admitted")
        if path.name.endswith(".index.json"):
            mapping = config.get("weight_map") if isinstance(config, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Weight index requires a nonempty weight map")
            for filename in mapping.values():
                if (not isinstance(filename, str) or
                    not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors", filename) or
                    not (root/filename).is_file()):
                    raise ValueError("Weight index must reference uploaded safetensors files")
        # Tokenizer file references must never escape the admitted bundle.
        if path.name in {"tokenizer_config.json", "config.json"} and isinstance(config, dict):
            for key, value in config.items():
                if key.endswith("_file") and value is not None:
                    if not isinstance(value, str) or Path(value).name != value or "/" in value or "\\" in value or not (root/value).is_file():
                        raise ValueError("Configuration file references must stay inside the bundle")


async def command(owner: str, action: str, payload: dict, settings) -> dict:
    from services.lua_adapters.host_boundaries import inspect_data_async
    # Reject an oversized binary envelope before Lua inspects it; accepted
    # document bytes still cross the configured Lua admission boundary before parsing.
    if action == "dataset" and "base64" in payload:
        encoded = payload["base64"]
        if not isinstance(encoded, str):
            raise ValueError("Expected a text field")
        if len(encoded) > MAX_BASE64_BYTES:
            raise DocumentIngestionError("document_upload_too_large")
    await inspect_data_async("source" if action.startswith(("dataset", "upload", "transform")) else "connection",
                             "admission", payload, {"owner": owner, "action": action})
    # Initialize the durable store on the application loop before dispatching
    # blocking filesystem/tokenizer/weight inspection to a worker thread.
    extension_store()
    if action.startswith("upload_"):
        def upload():
            with _upload_lock:
                return asyncio.run(_command(owner, action, payload, settings))
        return await asyncio.to_thread(upload)
    return await _command(owner, action, payload, settings)


async def _command(owner: str, action: str, payload: dict, settings) -> dict:
    for key in ("name", "model", "backend", "url", "api_key", "api_format", "text", "source", "license", "purpose", "id", "upload_id", "filename", "base64", "query"):
        if key in payload and not isinstance(payload[key], str):
            raise ValueError("Expected a text field")
    for key in ("model_id", "dataset_id"):
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            raise ValueError("Expected a resource identifier")
    if len(payload.get("api_key", "")) > 16_384:
        raise ValueError("Credential exceeds size limit")
    if "offset" in payload and (type(payload["offset"]) is not int or payload["offset"] < 0):
        raise ValueError("Invalid upload offset")
    store = extension_store()
    if action == "list":
        from services.extensions.providers import (
            connection_presets,
            provider_presets,
        )
        from services.extensions.transform import transforms_available
        from services.lua_adapters.profile import source_policy
        return {"schema": "extension-catalog.v1", "resources": [public_resource(r) for r in store.list(owner)],
                "selected": store.selected(owner),
                "role_assignments": store.role_assignments(owner),
                "specialist_roles": specialist_roles(settings),
                "provider_presets": provider_presets(),
                "connection_presets": connection_presets(),
                "inference_option_support": inference_option_support(),
                "networking_inference_options": networking_inference_options(),
                "local_profiles": [{"id": key, "backend": profile.backend} for key, profile in settings.model_profiles.items() if profile.enabled and not key.startswith("user-upload-")],
                "capabilities": workflow_capabilities(settings),
                "harnesses": [{"id": kind, "available": _installed(kind)} for kind in ["opencode", "pi"]],
                "transforms": [{"id": "lua-row-transform", "available": transforms_available()}],
                # Sites this machine's profile lets a user crawl into reference knowledge.
                "crawl_sources": source_policy()["allowed_origins"],
                "connections": [{"id": "rag", "state": "ready", "contract": "owner-retrieval.v1"},
                                {"id": "n8n", "state": "prepared", "contract": "governed-workflow.v1"}]}
    if action == "model":
        backend = payload.get(
            "backend"
        )

        if backend not in {
            "http",
            "local",
            "opencode",
            "pi",
        }:
            raise ValueError(
                "Choose a supported model "
                "or harness backend"
            )

        name = (
            payload.get(
                "name",
                "",
            )
            .strip()
        )

        model = (
            payload.get(
                "model",
                "",
            )
            .strip()
        )

        api_key = (
            payload.get(
                "api_key",
                "",
            )
            or ""
        )

        if (
            not name
            or not model
            or len(
                model
            ) > 300
            or model.startswith(
                "-"
            )
        ):
            raise ValueError(
                "Model name and identifier "
                "are required"
            )

        profile = {
            "name":
                name[:120],

            "backend":
                backend,

            "model":
                model,

            "state":
                "ready",
        }

        profile.update({key: payload[key] for key in MODEL_OPTION_FIELDS
                        if key in payload and (payload[key] is not None or key == "networking_inference")
                        and not (key == "reasoning_effort" and payload[key] == "")})

        if backend == "http":
            profile[
                "url"
            ] = validate_url(
                payload.get(
                    "url",
                    "",
                )
            )

            api_format = payload.get("api_format")
            if api_format is not None:
                if api_format not in {"anthropic", "openai"}:
                    raise ValueError("api_format must be anthropic or openai")
                profile["api_format"] = api_format

            provider_id = str(payload.get("provider_id") or "").strip().lower()
            if provider_id:
                if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", provider_id):
                    raise ValueError("provider_id must be 1-40 letters, digits, '-' or '_'")
                profile["provider_id"] = provider_id

        validate_model_options(profile)

        if backend == "local":
            if (
                model
                not in settings.model_profiles
                or model.startswith(
                    "user-upload-"
                )
                or not settings
                .model_profiles[
                    model
                ]
                .enabled
            ):
                raise ValueError(
                    "Choose an installed local "
                    "profile or upload a model first"
                )

        if backend == "pi":
            # Validate the stored provider/model shape
            # even before the first inference request.
            from services.extensions.harness import (
                pi_credential_env,
            )

            pi_credential_env(
                model
            )

        elif backend == "opencode":
            from services.extensions.harness import (
                harness_provider,
            )

            harness_provider(
                model
            )

        if (
            backend
            != "local"
            and _model_connection_requires_credential(
                backend,
                model,
                profile.get(
                    "url",
                    "",
                ),
            )
            and not api_key.strip()
        ):
            raise ValueError(
                "The selected hosted provider "
                "requires an API key"
            )

        if backend in {
            "opencode",
            "pi",
        }:
            # Installation is checked only after the
            # connection itself has passed admission.
            harness_binary(
                backend
            )

        return public_resource(
            store.put(
                owner,
                "model",
                profile,
                api_key,
            )
        )


    if action == "model_options":
        if set(payload) - (MODEL_OPTION_FIELDS | {"id"}):
            raise ValueError("Only model inference options can be updated")
        return public_resource(store.update_model_options(owner, payload["id"],
            {key: value for key, value in payload.items() if key != "id"}))

    if action == "select":
        store.select(owner, payload.get("model_id") or None, payload.get("dataset_id") or None)
        return store.selected(owner)
    if action == "assign_role":
        if set(payload) - {"role", "model_id"}:
            raise ValueError("Only a role and a model connection can be assigned")
        role = payload.get("role")
        if role not in {row["id"] for row in specialist_roles(settings)}:
            raise ValueError("Unknown specialist role")
        return {"role_assignments": store.assign_role(owner, role, payload.get("model_id") or None)}
    if action == "delete":
        store.delete(owner, payload["id"])
        return {"deleted": True}
    if action == "probe":
        profile = store.get(
            owner,
            payload["id"],
            secret=True,
        )

        if profile["kind"] != "model":
            raise ValueError(
                "Select a model connection"
            )

        if profile["backend"] == "http":
            # Connectivity probes deliberately use a different contract
            # from runtime inference. A provider that authenticates and
            # processes the request is reachable even when a reasoning-heavy
            # model spends the small probe budget before visible output.
            #
            # Actual chat/agent inference continues through remote_generate()
            # and still rejects truncated or missing visible output.
            from services.extensions.probe import remote_probe

            return await remote_probe(
                profile,
                max_tokens=256,
            )

        if profile["backend"] in {
            "opencode",
            "pi",
        }:
            from services.lua_adapters.host_boundaries import model_request, model_response
            # A harness probe is a model request: Lua first, then the harness.
            binding = {"connection_id": profile.get("id"), "stage": "CONNECTION_PROBE"}
            messages, budget = await model_request(
                [{"role": "user", "content": "Reply with exactly: HARNESS_OK"}], 256, binding=binding)
            text = await model_response(await harness_generate(profile, messages, budget), binding=binding)

            return {
                "schema":
                    "provider-probe.v1",

                "state":
                    "reachable",

                "probe_complete":
                    True,

                "response":
                    text[:2000],

                "message":
                    (
                        "Harness connection is reachable "
                        "and returned visible output."
                    ),
            }

        return {
            "schema":
                "provider-probe.v1",

            "state":
                "configured",

            "probe_complete":
                False,

            "message":
                (
                    "Local inference uses the existing GPU "
                    "scheduler; test it through a chat job."
                ),
        }

    if action == "models":
        # Lists models the supplied (or saved) credential can use. Credentials
        # are never echoed; only public identifiers and display names return.
        from services.extensions import anthropic
        if payload.get("id"):
            profile = store.get(owner, payload["id"], secret=True)
            if profile.get("kind") != "model":
                raise ValueError("Select a model connection")
        else:
            profile = {"backend": "http", "url": validate_url(payload.get("url", "")),
                       "api_key": payload.get("api_key", "")}
            if payload.get("api_format") is not None:
                profile["api_format"] = payload["api_format"]
        if not anthropic.uses_messages_api(profile):
            raise ValueError("Model listing is available for Anthropic connections")
        from services.extensions.diagnostics import ExternalProviderError
        from services.extensions.probe import _category_error
        try:
            return {"models": await anthropic.list_models(profile["url"], profile.get("api_key", ""))}
        except ExternalProviderError as exc:
            raise _category_error(exc) from None
    if action == "dataset":
        fields = {key: payload.get(key) for key in ("name", "source", "license", "purpose", "reviewed")}
        if not all(isinstance(fields[k], str) for k in ("name", "source", "license", "purpose")):
            raise ValueError("Dataset fields must be text")
        replaces = replaced_dataset(payload)
        if "base64" in payload:
            if fields["purpose"] != "rag":
                raise DocumentIngestionError("document_training_not_supported")
            if "text" in payload:
                raise DocumentIngestionError("document_format_not_supported")
            extracted = await extract_document(payload.get("filename", ""), payload["base64"])
            return public_resource(store.dataset(
                owner, **fields, text=extracted.text, replaces=replaces, document=extracted.metadata,
            ))
        fields["text"] = payload.get("text")
        if not isinstance(fields["text"], str):
            raise ValueError("Dataset fields must be text")
        return public_resource(store.dataset(owner, **fields, replaces=replaces))
    if action == "dataset_transform":
        # The owner's Lua script reshapes their own rows in an isolated worker. A preview
        # admits nothing; with apply, the output goes through the same admission as any upload.
        from services.extensions.transform import TransformError, run_transform
        source, text, apply = payload.get("script"), payload.get("text"), payload.get("apply", False)
        if not isinstance(source, str) or not isinstance(text, str) or type(apply) is not bool:
            raise TransformError("transform_request_invalid", "Send the script, the rows and whether to apply.")
        result = await asyncio.to_thread(run_transform, source, text)
        if not apply:
            preview = [json.loads(line) for line in result["text"].splitlines()[:5]]
            return {key: value for key, value in result.items() if key != "text"} | {"preview": preview}
        fields = {key: payload.get(key) for key in ("name", "source", "license", "purpose", "reviewed")}
        if not all(isinstance(fields[k], str) for k in ("name", "source", "license", "purpose")):
            raise ValueError("Dataset fields must be text")
        provenance = {"schema": result["schema"], "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "input_rows": result["input_rows"], "output_rows": result["output_rows"]}
        return public_resource(store.dataset(owner, **fields, text=result["text"], replaces=replaced_dataset(payload),
                                             provenance=provenance))
    if action == "dataset_url":
        replaces = replaced_dataset(payload)
        data = await request_json(payload.get("url", ""), api_key=payload.get("api_key", ""))
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError("Dataset API must return an object containing a text string")
        return public_resource(store.dataset(owner, name=payload.get("name", ""), text=data["text"],
            source=payload["url"], license=payload.get("license", ""), purpose=payload.get("purpose", "rag"), reviewed=payload.get("reviewed"),
            replaces=replaces))
    if action == "dataset_crawl":
        # Reads one site this machine's profile allows into reference knowledge. Crawled
        # pages are third-party text, so they are admitted for retrieval only, never training.
        from services.knowledge.crawler import crawl
        if payload.get("purpose", "rag") != "rag":
            raise ValueError("Crawled sites are admitted as reference knowledge only")
        from services.extensions.store import sensitive_content
        from services.knowledge.crawler import CrawlError
        replaces = replaced_dataset(payload)
        crawled = await crawl(payload.get("url", ""))
        # A page that admission would refuse (a credential or personal identifier) is
        # withheld and counted, so one page cannot block the rest of a site.
        crawled.withhold(lambda page: sensitive_content(page) is None)
        if not crawled.pages:
            raise CrawlError("crawl_all_pages_withheld", "Every page read contained possible credentials or "
                             "personal identifiers, so nothing was saved.")
        return public_resource(store.dataset(owner, name=payload.get("name", ""), text=crawled.corpus(),
            source=crawled.start_url, license=payload.get("license", ""), purpose="rag",
            reviewed=payload.get("reviewed"), replaces=replaces, crawl=crawled.provenance()))
    if action == "search":
        return {"hits": store.search(owner, payload["id"], str(payload.get("query", ""))[:200])}
    if action == "resolve":
        # Reopen one cited passage: the owner's dataset at the exact cited version.
        chunk, digest = payload.get("chunk"), payload.get("sha256")
        if type(chunk) is not int or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid citation")
        return {"hit": store.resolve(owner, payload["id"], chunk, digest)}
    if action == "workflow_validate":
        from services.extensions.workflow import admit_workflow
        from tools.registry import TOOLS
        return admit_workflow(payload, TOOLS)
    if action == "export_dataset":
        row = store.get(owner, payload["id"])
        if row["kind"] != "dataset" or row["purpose"] != "training":
            raise ValueError("Only admitted training datasets have fixed training/validation splits")
        records = [{"id": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                    **record, "split": split, "reviewed": True, "source": row["source"], "license": row["license"]}
                   for split, rows in row["content"].items() for record in rows]
        return {"schema": "private-training-dataset.v1", "source": row["source"], "license": row["license"],
                "records": records,
                "sha256": row["sha256"], "splits": row["content"], "promotion": "requires_held_out_evaluation_and_separate_approval",
                **dataset_version(row)}
    if action == "upload_begin":
        identity = uuid.uuid4().hex
        upload_root(owner, identity).mkdir(parents=True, mode=0o700)
        return {"upload_id": identity, "chunk_bytes": 1_000_000}
    if action == "upload_chunk":
        root = upload_root(owner, payload["upload_id"])
        if not root.is_dir():
            raise ValueError("Upload session not found")
        if (root/".sealed").exists():
            raise ValueError("Admitted model uploads are immutable")
        name = payload.get("filename", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,150}", name) or name.startswith(".") or Path(name).suffix not in {".json", ".safetensors", ".txt", ".model"}:
            raise ValueError("Only safe model data files are accepted; executable and pickle files are prohibited")
        blob = base64.b64decode(payload.get("base64", ""), validate=True)
        if not 0 < len(blob) <= 1_000_000:
            raise ValueError("Invalid upload chunk size")
        path = root/name
        if path.is_symlink() or payload.get("offset") != (path.stat().st_size if path.exists() else 0):
            raise ValueError("Upload offset mismatch")
        if sum(p.stat().st_size for p in root.iterdir()) + len(blob) > 20_000_000_000:
            raise ValueError("Model bundle exceeds 20 GB")
        with path.open("ab") as stream:
            stream.write(blob)
        return {"offset": path.stat().st_size}
    if action == "upload_finish":
        root = upload_root(owner, payload["upload_id"])
        if not root.is_dir() or not (root/"config.json").is_file() or not list(root.glob("*.safetensors")) or not (root/"tokenizer_config.json").is_file():
            raise ValueError("Model requires config.json, tokenizer configuration and safetensors weights")
        validate_model_bundle(root)
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        import torch
        configuration = lua_from_pretrained(AutoConfig.from_pretrained, str(root), local_files_only=True, trust_remote_code=False)
        if type(configuration) not in AutoModelForCausalLM._model_mapping:
            raise ValueError("Model architecture does not support causal language generation")
        lua_from_pretrained(AutoTokenizer.from_pretrained, str(root), local_files_only=True, trust_remote_code=False)
        parameters = 0
        for p in root.glob("*.safetensors"):
            with safe_open(p, framework="pt", device="cpu") as weights:
                if not weights.keys():
                    raise ValueError("Empty safetensors file")
                import math
                parameters += sum(math.prod(weights.get_slice(key).get_shape()) for key in weights.keys())
        estimated_vram = int(parameters * 0.75 + 1_073_741_824)
        if not torch.cuda.is_available() or estimated_vram > torch.cuda.get_device_properties(0).total_memory:
            raise ValueError("Uploaded model exceeds this host's estimated quantized GPU capacity")
        hashes = {}
        for p in root.iterdir():
            if p.name == ".sealed":
                raise ValueError("Upload was already admitted")
            with p.open("rb") as stream:
                hashes[p.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        (root/".sealed").write_text(json.dumps(hashes, sort_keys=True))
        return public_resource(store.put(owner, "model", {"name": str(payload.get("name", "Uploaded local model"))[:120],
            "backend": "uploaded", "model": payload["upload_id"], "model_path": str(root), "files": hashes,
            "state": "ready", "quantization": "bnb4", "estimated_vram_bytes": estimated_vram,
            "preflight": "format, architecture and estimated GPU capacity checked; semantic quality requires evaluation"}))
    raise ValueError("Unknown extension action")


def _installed(kind):
    try:
        harness_binary(kind)
        return True
    except ValueError:
        return False
