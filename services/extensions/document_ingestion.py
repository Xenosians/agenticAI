"""Owner-requested PDF/DOCX extraction before normal RAG dataset admission.

The document is parsed in ``document_worker.py``, a separate ``python -I`` process
with the Lua workers' sandbox (no files, sockets or processes once locked down;
CPU and memory limits), a minimal environment and a wall-clock limit. Only the
extracted text and the source's digest and parser version leave this module.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import re
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path

from services.extensions.transform import worker_environment


MAX_DOCUMENT_BYTES = 2_000_000
MAX_BASE64_BYTES = ((MAX_DOCUMENT_BYTES + 2) // 3) * 4
MAX_ANSWER_BYTES = 2 * 6 * MAX_DOCUMENT_BYTES  # JSON-escaped text; the worker caps the text itself.
WORKER_WALL_SECONDS = 15.0
WORKER = Path(__file__).with_name("document_worker.py")
# The worker's lockdown must include at least these layers, as for the Lua workers.
REQUIRED_ISOLATION = frozenset({"process", "no_new_privileges", "resource_limits", "no_new_descriptors"})


_PARSER_SLOTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class DocumentIngestionError(ValueError):
    """A stable, content-free refusal safe to return to the owner."""

    _MESSAGES = {
        "document_format_not_supported": "Choose a PDF or DOCX reference document.",
        "document_training_not_supported": "PDF and DOCX uploads are for RAG references; training data must be JSONL.",
        "document_upload_too_large": "Document files must be 2 MB or smaller.",
        "document_file_invalid": "The uploaded file is not a valid supported document.",
        "document_encrypted": "Password-protected documents cannot be indexed.",
        "document_extraction_not_permitted": "This PDF does not permit text extraction.",
        "document_content_empty": "No selectable text was found. Scanned-image OCR is not supported.",
        "document_resource_limit": "The document exceeds the parser's resource limits.",
        "document_text_too_large": "Extracted text exceeds the 2 MB dataset limit.",
        "document_parser_unavailable": "The isolated document parser is not available on this host.",
        "document_parse_failed": "The document could not be parsed.",
    }

    def __init__(self, code: str):
        if code not in self._MESSAGES:
            code = "document_parse_failed"
        self.code = code
        self.message = self._MESSAGES[code]
        super().__init__(code)

    def public_detail(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    metadata: dict[str, str]


def _worker_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _PARSER_SLOTS.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(2)
        _PARSER_SLOTS[loop] = semaphore
    return semaphore


def _decode(filename: str, encoded: str) -> tuple[str, bytes]:
    if not isinstance(filename, str) or not isinstance(encoded, str):
        raise DocumentIngestionError("document_file_invalid")
    if len(encoded) > MAX_BASE64_BYTES or not encoded:
        raise DocumentIngestionError("document_upload_too_large")
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise DocumentIngestionError("document_file_invalid")
    suffix = Path(filename).suffix.lower()
    if (suffix not in {".pdf", ".docx"} or len(filename) > 150 or filename.startswith(".")
            or re.search(r"[\x00-\x1f\x7f]", filename)):
        raise DocumentIngestionError("document_format_not_supported")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise DocumentIngestionError("document_file_invalid") from None
    if not data:
        raise DocumentIngestionError("document_file_invalid")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentIngestionError("document_upload_too_large")
    return suffix[1:], data


async def extract_document(filename: str, encoded: str) -> ExtractedDocument:
    """Extract selectable text in the isolated, resource-limited document worker."""
    kind, data = _decode(filename, encoded)
    if os.name != "posix":
        raise DocumentIngestionError("document_parser_unavailable")

    async with _worker_semaphore():
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-B", str(WORKER), kind,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd="/",
                env=worker_environment(),
                close_fds=True,
                start_new_session=True,
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(data), WORKER_WALL_SECONDS)
            except asyncio.TimeoutError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.communicate()
                raise DocumentIngestionError("document_resource_limit") from None
            except asyncio.CancelledError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.communicate()
                raise
        except DocumentIngestionError:
            raise
        except (OSError, RuntimeError):
            raise DocumentIngestionError("document_parser_unavailable") from None

    text = _worker_text(process.returncode, output)
    package = "pypdf" if kind == "pdf" else "python-docx"
    try:
        parser_version = version(package)
    except PackageNotFoundError:
        raise DocumentIngestionError("document_parser_unavailable") from None
    metadata = {
        "filename": filename,
        "file_format": kind,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "parser": f"{kind}-text-extractor",
        "parser_version": parser_version,
    }
    return ExtractedDocument(text=text, metadata=metadata)


def _worker_text(returncode: int | None, output: bytes) -> str:
    """The worker's extracted text, or the stable refusal its answer names."""
    if len(output) > MAX_ANSWER_BYTES:
        raise DocumentIngestionError("document_text_too_large")
    try:
        answer = json.loads(output) if output else None
    except (ValueError, UnicodeDecodeError):
        answer = None
    if not isinstance(answer, dict):
        # Killed by its CPU or memory limit before it could answer.
        raise DocumentIngestionError("document_resource_limit" if returncode and returncode < 0
                                     else "document_parse_failed")
    if "error" in answer:
        raise DocumentIngestionError(answer["error"] if isinstance(answer["error"], str) else "document_parse_failed")
    text, layers = answer.get("text"), answer.get("isolation")
    if (returncode != 0 or not isinstance(text, str) or not isinstance(layers, list)
            or not all(isinstance(layer, str) for layer in layers)):
        raise DocumentIngestionError("document_parse_failed")
    if not REQUIRED_ISOLATION <= set(layers):
        raise DocumentIngestionError("document_parser_unavailable")
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise DocumentIngestionError("document_text_too_large")
    if not text.strip():
        raise DocumentIngestionError("document_content_empty")
    return text
