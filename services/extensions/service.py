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

from services.extensions.store import ExtensionStore
from services.extensions.transport import request_json, validate_url
from services.extensions.harness import harness_binary, harness_generate
from services.extensions.options import INFERENCE_OPTION_FIELDS, inference_option_support, validate_inference_options

_upload_lock = threading.Lock()


@lru_cache
def extension_store():
    return ExtensionStore(Path(os.environ.get("EXTENSION_STORAGE_ROOT", Path.home()/".local/share/agentic-extensions")))


def public_resource(row):
    # No dataset bodies, credentials or server filesystem paths in list responses.
    return {key: value for key, value in row.items() if key not in {"api_key", "content", "model_path"}}


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
    for key in ("name", "model", "backend", "url", "api_key", "text", "source", "license", "purpose", "id", "upload_id", "filename", "base64", "query"):
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
        from tools.registry import TOOLS
        from services.extensions.providers import (
            connection_presets,
            provider_presets,
        )
        return {"schema": "extension-catalog.v1", "resources": [public_resource(r) for r in store.list(owner)],
                "selected": store.selected(owner),
                "provider_presets": provider_presets(),
                "connection_presets": connection_presets(),
                "inference_option_support": inference_option_support(),
                "local_profiles": [{"id": key, "backend": profile.backend} for key, profile in settings.model_profiles.items() if profile.enabled and not key.startswith("user-upload-")],
                "capabilities": [{"id": name, "description": tool.get("description", ""),
                                  "effect": tool.get("effect"), "permission": tool.get("permission")}
                                 for name, tool in sorted(TOOLS.items())],
                "harnesses": [{"id": kind, "available": _installed(kind)} for kind in ["opencode", "pi"]],
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

        profile.update({key: payload[key] for key in INFERENCE_OPTION_FIELDS
                        if key in payload and payload[key] is not None
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

        validate_inference_options(profile)

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
        if set(payload) - (INFERENCE_OPTION_FIELDS | {"id"}):
            raise ValueError("Only model inference options can be updated")
        return public_resource(store.update_model_options(owner, payload["id"],
            {key: value for key, value in payload.items() if key != "id"}))

    if action == "select":
        store.select(owner, payload.get("model_id") or None, payload.get("dataset_id") or None)
        return store.selected(owner)
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
            text = await harness_generate(
                profile,
                [
                    {
                        "role":
                            "user",

                        "content":
                            (
                                "Reply with exactly: "
                                "HARNESS_OK"
                            ),
                    }
                ],
                256,
            )

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

    if action == "dataset":
        fields = {key: payload.get(key) for key in ("name", "text", "source", "license", "purpose", "reviewed")}
        if not all(isinstance(fields[k], str) for k in ("name", "text", "source", "license", "purpose")):
            raise ValueError("Dataset fields must be text")
        return public_resource(store.dataset(owner, **fields))
    if action == "dataset_url":
        data = await request_json(payload.get("url", ""), api_key=payload.get("api_key", ""))
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError("Dataset API must return an object containing a text string")
        return public_resource(store.dataset(owner, name=payload.get("name", ""), text=data["text"],
            source=payload["url"], license=payload.get("license", ""), purpose=payload.get("purpose", "rag"), reviewed=payload.get("reviewed")))
    if action == "search":
        return {"hits": store.search(owner, payload["id"], str(payload.get("query", ""))[:200])}
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
                "sha256": row["sha256"], "splits": row["content"], "promotion": "requires_held_out_evaluation_and_separate_approval"}
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
        configuration = AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        if type(configuration) not in AutoModelForCausalLM._model_mapping:
            raise ValueError("Model architecture does not support causal language generation")
        AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
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
