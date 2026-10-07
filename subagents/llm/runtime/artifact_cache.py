from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import time
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Any


_WEIGHT_SUFFIXES = {
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
}

_SKIP_NAMES = {
    "training_args.bin",
}

_READY_MANIFEST = "cache-manifest.json"


@dataclass(frozen=True)
class ArtifactResolution:
    key: str
    path: Path
    hit: bool


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _source_signature(directory: Path) -> str:
    """
    Fast deployment-cache invalidation signature.

    This intentionally does not hash multi-gigabyte weight shards on every
    load. It hashes stable file metadata plus small configuration files.

    Retraining/promotion normally changes the artifact path and/or shard
    metadata. This cache is disposable; authoritative model provenance stays
    in the source model/promotion system.
    """
    directory = directory.expanduser().resolve()

    if not directory.is_dir():
        raise ValueError(
            f"Model source directory does not exist: {directory}"
        )

    entries: list[dict[str, Any]] = []

    for path in sorted(
        (p for p in directory.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(directory)),
    ):
        stat = path.stat()
        relative = str(path.relative_to(directory))

        entry: dict[str, Any] = {
            "path": relative,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

        if (
            path.suffix.lower() not in _WEIGHT_SUFFIXES
            and stat.st_size <= 4 * 1024 * 1024
        ):
            try:
                entry["small_file_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            except OSError:
                pass

        entries.append(entry)

    return _json_hash(entries)


def _profile_value(profile, name: str):
    return getattr(profile, name, None)


def _profile_identity(profile) -> dict[str, Any]:
    model_path = _profile_value(profile, "model_path")

    if model_path is None:
        raise ValueError("A cacheable model profile requires model_path.")

    source_path = Path(model_path).expanduser().resolve()

    return {
        "schema": "model-deployment-artifact.v1",
        "source_path": str(source_path),
        "source_signature": _source_signature(source_path),
        "backend": _profile_value(profile, "backend"),
        "quantization": _profile_value(profile, "quantization"),
        "model_dtype": _profile_value(profile, "model_dtype"),
        "compute_dtype": _profile_value(profile, "compute_dtype"),
        "bnb_4bit_quant_type": _profile_value(
            profile,
            "bnb_4bit_quant_type",
        ),
        "bnb_4bit_use_double_quant": _profile_value(
            profile,
            "bnb_4bit_use_double_quant",
        ),
        "dequantize_fp8": _profile_value(
            profile,
            "dequantize_fp8",
        ),
        "runtime": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "bitsandbytes": _package_version("bitsandbytes"),
        },
    }


def physical_model_signature(profile) -> str:
    """
    Signature used to decide whether two logical profiles can reuse the same
    already-loaded physical base.

    Adapter identity is deliberately excluded. The adapter is the switchable
    logical layer on top of the shared physical base.
    """
    model_path = _profile_value(profile, "model_path")

    if model_path is None:
        return ""

    value = {
        "backend": _profile_value(profile, "backend"),
        "model_path": str(
            Path(model_path).expanduser().resolve()
        ),
        "quantization": _profile_value(profile, "quantization"),
        "model_dtype": _profile_value(profile, "model_dtype"),
        "compute_dtype": _profile_value(profile, "compute_dtype"),
        "device_map": _profile_value(profile, "device_map"),
        "offload_folder": (
            None
            if _profile_value(profile, "offload_folder") is None
            else str(
                Path(
                    _profile_value(profile, "offload_folder")
                ).expanduser().resolve()
            )
        ),
        "dequantize_fp8": _profile_value(
            profile,
            "dequantize_fp8",
        ),
        "bnb_4bit_quant_type": _profile_value(
            profile,
            "bnb_4bit_quant_type",
        ),
        "bnb_4bit_use_double_quant": _profile_value(
            profile,
            "bnb_4bit_use_double_quant",
        ),
    }

    return _json_hash(value)


class ModelArtifactCache:
    """
    Persistent deployment-weight cache.

    Authoritative model artifacts remain under the configured model_path.
    This cache only stores disposable serving artifacts such as a serialized
    pre-quantized checkpoint.

    SQLite stores metadata/index information only. Tensor files remain normal
    files so Transformers/safetensors can load them efficiently.
    """

    def __init__(
        self,
        root: Path,
        *,
        materialize_quantized: bool = True,
        prefetch: bool = False,
    ) -> None:
        self.root = (
            Path(root)
            .expanduser()
            .resolve()
        )

        self.artifacts_root = (
            self.root
            / "artifacts"
        )

        self.index_path = (
            self.root
            / "index.sqlite3"
        )

        self.materialize_quantized = bool(
            materialize_quantized
        )

        self.prefetch = bool(
            prefetch
        )

        self.artifacts_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_index()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.index_path
        )

        conn.execute(
            "PRAGMA journal_mode=WAL"
        )

        return conn

    def _initialize_index(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    cache_key TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    cache_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    error TEXT
                )
                """
            )

    def key_for(self, profile) -> str:
        return _json_hash(
            _profile_identity(profile)
        )

    def cache_path_for_key(
        self,
        cache_key: str,
    ) -> Path:
        return (
            self.artifacts_root
            / cache_key
        )

    @staticmethod
    def _manifest_path(
        cache_path: Path,
    ) -> Path:
        return (
            cache_path
            / _READY_MANIFEST
        )

    @staticmethod
    def _looks_prequantized_bnb4(
        cache_path: Path,
    ) -> bool:
        config_path = (
            cache_path
            / "config.json"
        )

        if not config_path.is_file():
            return False

        try:
            config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return False

        quant = config.get(
            "quantization_config"
        )

        if not isinstance(
            quant,
            dict,
        ):
            return False

        if quant.get(
            "load_in_4bit"
        ) is True:
            return True

        if quant.get(
            "_load_in_4bit"
        ) is True:
            return True

        method = quant.get(
            "quant_method"
        )

        if isinstance(
            method,
            str,
        ):
            normalized = (
                method
                .strip()
                .lower()
            )

            if (
                "bitsandbytes"
                in normalized
                and (
                    "4"
                    in normalized
                    or quant.get(
                        "_load_in_4bit"
                    ) is True
                )
            ):
                return True

        return False

    def _valid_cache(
        self,
        *,
        cache_key: str,
        cache_path: Path,
    ) -> bool:
        manifest_path = (
            self._manifest_path(
                cache_path
            )
        )

        if not manifest_path.is_file():
            return False

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return False

        if (
            manifest.get(
                "cache_key"
            )
            != cache_key
        ):
            return False

        if (
            manifest.get(
                "state"
            )
            != "ready"
        ):
            return False

        if not self._looks_prequantized_bnb4(
            cache_path
        ):
            return False

        return any(
            path.is_file()
            for path
            in cache_path.glob(
                "*.safetensors"
            )
        )

    def resolve(
        self,
        profile,
    ) -> ArtifactResolution:
        source = Path(
            _profile_value(
                profile,
                "model_path",
            )
        ).expanduser().resolve()

        cache_key = (
            self.key_for(
                profile
            )
        )

        cache_path = (
            self.cache_path_for_key(
                cache_key
            )
        )

        hit = (
            self._valid_cache(
                cache_key=cache_key,
                cache_path=cache_path,
            )
        )

        now = time.time()

        if hit:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO model_artifacts (
                        cache_key,
                        source_path,
                        cache_path,
                        state,
                        created_at,
                        last_used_at,
                        error
                    )
                    VALUES (?, ?, ?, 'ready', ?, ?, NULL)
                    ON CONFLICT(cache_key)
                    DO UPDATE SET
                        last_used_at=excluded.last_used_at,
                        state='ready',
                        error=NULL
                    """,
                    (
                        cache_key,
                        str(source),
                        str(cache_path),
                        now,
                        now,
                    ),
                )

            if self.prefetch:
                self.prefetch_files(
                    cache_path
                )

            print(
                "[MODEL_CACHE] HIT "
                f"key='{cache_key[:16]}...' "
                f"path='{cache_path}'"
            )

            return ArtifactResolution(
                key=cache_key,
                path=cache_path,
                hit=True,
            )

        print(
            "[MODEL_CACHE] MISS "
            f"key='{cache_key[:16]}...' "
            f"source='{source}'"
        )

        return ArtifactResolution(
            key=cache_key,
            path=source,
            hit=False,
        )

    def prefetch_files(
        self,
        directory: Path,
    ) -> None:
        """
        Best-effort OS page-cache warming for deployment weight shards.

        This does not pin tensors in VRAM and does not retain model objects.
        """
        try:
            for path in sorted(
                directory.glob(
                    "*.safetensors"
                )
            ):
                with path.open(
                    "rb"
                ) as handle:
                    while handle.read(
                        8 * 1024 * 1024
                    ):
                        pass
        except OSError as exc:
            print(
                "[MODEL_CACHE] Prefetch skipped "
                f"error={exc!r}"
            )

    @staticmethod
    def _copy_metadata_files(
        source: Path,
        destination: Path,
    ) -> None:
        for path in source.rglob("*"):
            if not path.is_file():
                continue

            if path.name in _SKIP_NAMES:
                continue

            if (
                path.suffix.lower()
                in _WEIGHT_SUFFIXES
            ):
                continue

            relative = (
                path.relative_to(
                    source
                )
            )

            target = (
                destination
                / relative
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:
                shutil.copy2(
                    path,
                    target,
                )
            except OSError:
                pass

    def materialize(
        self,
        *,
        profile,
        backend,
        cache_key: str,
    ) -> Path | None:
        if not self.materialize_quantized:
            return None

        if (
            str(
                _profile_value(
                    profile,
                    "backend",
                )
            )
            .strip()
            .lower()
            != "ministral"
        ):
            return None

        if (
            str(
                _profile_value(
                    profile,
                    "quantization",
                )
            )
            .strip()
            .lower()
            != "bnb4"
        ):
            return None

        model = getattr(
            backend,
            "model",
            None,
        )

        if model is None:
            return None

        save_pretrained = getattr(
            model,
            "save_pretrained",
            None,
        )

        if not callable(
            save_pretrained
        ):
            return None

        final_path = (
            self.cache_path_for_key(
                cache_key
            )
        )

        if self._valid_cache(
            cache_key=cache_key,
            cache_path=final_path,
        ):
            return final_path

        source = Path(
            _profile_value(
                profile,
                "model_path",
            )
        ).expanduser().resolve()

        building_path = (
            self.artifacts_root
            / (
                f".building-"
                f"{cache_key[:16]}-"
                f"{os.getpid()}-"
                f"{uuid.uuid4().hex[:8]}"
            )
        )

        building_path.mkdir(
            parents=True,
            exist_ok=False,
        )

        started = time.time()

        try:
            self._copy_metadata_files(
                source,
                building_path,
            )

            save_pretrained(
                str(
                    building_path
                ),
                safe_serialization=True,
                max_shard_size="2GB",
            )

            tokenizer = getattr(
                backend,
                "tokenizer",
                None,
            )

            tokenizer_save = getattr(
                tokenizer,
                "save_pretrained",
                None,
            )

            if callable(
                tokenizer_save
            ):
                try:
                    tokenizer_save(
                        str(
                            building_path
                        )
                    )
                except Exception:
                    pass

            if not self._looks_prequantized_bnb4(
                building_path
            ):
                raise RuntimeError(
                    "Serialized checkpoint did not preserve a "
                    "4-bit bitsandbytes quantization configuration."
                )

            if not any(
                path.is_file()
                for path
                in building_path.glob(
                    "*.safetensors"
                )
            ):
                raise RuntimeError(
                    "Serialized checkpoint contains no safetensors weights."
                )

            manifest = {
                "schema":
                    "model-deployment-artifact.v1",
                "cache_key":
                    cache_key,
                "state":
                    "ready",
                "source_path":
                    str(source),
                "created_at_unix":
                    time.time(),
                "materialization_seconds":
                    round(
                        time.time()
                        - started,
                        3,
                    ),
                "profile_identity":
                    _profile_identity(
                        profile
                    ),
            }

            self._manifest_path(
                building_path
            ).write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            if final_path.exists():
                shutil.rmtree(
                    final_path
                )

            os.replace(
                building_path,
                final_path,
            )

            now = time.time()

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO model_artifacts (
                        cache_key,
                        source_path,
                        cache_path,
                        state,
                        created_at,
                        last_used_at,
                        error
                    )
                    VALUES (?, ?, ?, 'ready', ?, ?, NULL)
                    ON CONFLICT(cache_key)
                    DO UPDATE SET
                        source_path=excluded.source_path,
                        cache_path=excluded.cache_path,
                        state='ready',
                        last_used_at=excluded.last_used_at,
                        error=NULL
                    """,
                    (
                        cache_key,
                        str(source),
                        str(final_path),
                        now,
                        now,
                    ),
                )

            print(
                "[MODEL_CACHE] Materialized "
                f"key='{cache_key[:16]}...' "
                f"path='{final_path}'"
            )

            return final_path

        except Exception as exc:
            shutil.rmtree(
                building_path,
                ignore_errors=True,
            )

            now = time.time()

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO model_artifacts (
                        cache_key,
                        source_path,
                        cache_path,
                        state,
                        created_at,
                        last_used_at,
                        error
                    )
                    VALUES (?, ?, ?, 'failed', ?, ?, ?)
                    ON CONFLICT(cache_key)
                    DO UPDATE SET
                        state='failed',
                        last_used_at=excluded.last_used_at,
                        error=excluded.error
                    """,
                    (
                        cache_key,
                        str(source),
                        str(final_path),
                        now,
                        now,
                        repr(exc),
                    ),
                )

            # Cache construction is an optimization, never a correctness
            # dependency. Serving continues from the authoritative source.
            print(
                "[MODEL_CACHE] Materialization skipped "
                f"key='{cache_key[:16]}...' "
                f"error={exc!r}"
            )

            return None

    def purge_stale(
        self,
        *,
        keep_keys: set[str],
    ) -> list[Path]:
        removed: list[Path] = []

        for path in self.artifacts_root.iterdir():
            if not path.is_dir():
                continue

            if path.name.startswith(
                ".building-"
            ):
                shutil.rmtree(
                    path,
                    ignore_errors=True,
                )
                removed.append(
                    path
                )
                continue

            if path.name in keep_keys:
                continue

            shutil.rmtree(
                path,
                ignore_errors=True,
            )

            removed.append(
                path
            )

        return removed
