"""Isolated worker that reads the selectable text of one PDF or DOCX upload.

Started by ``services.extensions.document_ingestion`` as ``python -I -B`` with a
minimal environment and only its stdin/stdout/stderr pipes. It reads the document
from stdin, loads both parsers and every codec they may ask for, then locks itself
down with the sandbox the Lua workers use (no new privileges, files, sockets or
processes; CPU time and memory limits) before parsing the document. It writes one
JSON answer: the text and the isolation layers it ran under, or a stable
``document_*`` refusal code. It never writes files, follows links, runs macros,
renders images or performs OCR.
"""
from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import pkgutil
import runpy
import sys
import zipfile

# Reuse the established sandbox; a parser of untrusted files must not run in a weaker one.
# run_name keeps the row worker's main entrypoint inactive.
shared = runpy.run_path(str(Path(__file__).with_name("transform_worker.py")), run_name="document_isolation")

MAX_INPUT_BYTES = 2_000_000
MAX_TEXT_BYTES = 2_000_000
MAX_PDF_PAGES = 100
MAX_DOCX_ENTRIES = 512
MAX_DOCX_UNCOMPRESSED_BYTES = 16_000_000
MAX_DOCX_MEMBER_BYTES = 8_000_000
CPU_SECONDS = 5
ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
# After lockdown no file can be opened, so a module imported lazily while parsing (a
# codec a document names, a parser helper) would fail. Every module of these packages
# is loaded first.
PRELOADED_PACKAGES = ("encodings", "pypdf", "docx")


class Refused(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def preload() -> None:
    for name in PRELOADED_PACKAGES:
        package = importlib.import_module(name)
        for module in pkgutil.walk_packages(package.__path__, name + "."):
            try:
                importlib.import_module(module.name)
            except ImportError:
                pass  # Another platform's codec (mbcs) or an optional backend (pycryptodome).


class Text:
    """Printable text gathered within the size limit."""

    def __init__(self):
        self.parts: list[str] = []
        self.size = 0

    def add(self, text: str) -> None:
        clean = "".join(char for char in text if char in "\n\r\t" or char.isprintable())
        if not clean.strip():
            return
        self.size += len(clean.encode("utf-8")) + (1 if self.parts else 0)
        if self.size > MAX_TEXT_BYTES:
            raise Refused("document_text_too_large")
        self.parts.append(clean)

    def value(self) -> str:
        return "\n".join(self.parts)


def read_pdf(data: bytes) -> str:
    from pypdf import PasswordType, PdfReader
    from pypdf.constants import UserAccessPermissions
    from pypdf.errors import LimitReachedError, PyPdfError

    if not data.startswith(b"%PDF-"):
        raise Refused("document_file_invalid")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            # Many vendor manuals are encrypted only to declare permissions, with an
            # empty user password; those open. A document that needs a password does not.
            if reader.decrypt("") == PasswordType.NOT_DECRYPTED:
                raise Refused("document_encrypted")
            permissions = reader.user_access_permissions
            if permissions is not None and not permissions & UserAccessPermissions.EXTRACT:
                raise Refused("document_extraction_not_permitted")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise Refused("document_resource_limit")
        text = Text()
        for page in reader.pages:
            text.add(page.extract_text() or "")
        return text.value()
    except LimitReachedError:
        raise Refused("document_resource_limit") from None
    except PyPdfError:
        raise Refused("document_file_invalid") from None


def _check_archive(data: bytes) -> None:
    """Refuse archives that are oversized, oddly named, encrypted or macro-enabled before parsing."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_ENTRIES:
                raise Refused("document_resource_limit")
            total, members = 0, {}
            for entry in entries:
                name = entry.filename.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/") or entry.flag_bits & 0x1 or name in members:
                    raise Refused("document_file_invalid")
                total += entry.file_size
                if entry.file_size > MAX_DOCX_MEMBER_BYTES or total > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise Refused("document_resource_limit")
                members[name] = entry
            if "[Content_Types].xml" not in members or "word/document.xml" not in members:
                raise Refused("document_file_invalid")
            if members["[Content_Types].xml"].file_size > 256_000:
                raise Refused("document_resource_limit")
            content_types = archive.read("[Content_Types].xml")
            if b"macroenabled" in content_types.lower() or any(
                    name.lower().endswith("vbaproject.bin") for name in members):
                raise Refused("document_file_invalid")
    except zipfile.BadZipFile:
        raise Refused("document_file_invalid") from None


def read_docx(data: bytes) -> str:
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    _check_archive(data)
    try:
        document = Document(io.BytesIO(data))
    except (PackageNotFoundError, KeyError, ValueError, zipfile.BadZipFile):
        raise Refused("document_file_invalid") from None
    text = Text()
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text.add(block.text)
        elif isinstance(block, Table):
            for row in block.rows:
                text.add("\t".join(cell.text for cell in row.cells))
    return text.value()


READERS = {"pdf": read_pdf, "docx": read_docx}


def main() -> int:
    kind = sys.argv[1] if len(sys.argv) == 2 else ""
    data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    try:
        if kind not in READERS:
            raise Refused("document_format_not_supported")
        if not data or len(data) > MAX_INPUT_BYTES:
            raise Refused("document_upload_too_large")
        preload()
        layers = shared["lockdown"](CPU_SECONDS, ADDRESS_SPACE_BYTES)
        text = READERS[kind](data)
        if not text.strip():
            raise Refused("document_content_empty")
        answer = {"text": text, "isolation": layers}
    except Refused as exc:
        answer = {"error": exc.code}
    except shared["Failure"]:
        answer = {"error": "document_parser_unavailable"}
    except (MemoryError, RecursionError):
        answer = {"error": "document_resource_limit"}
    except Exception:  # A parser defect on hostile input: a stable code, never its details.
        answer = {"error": "document_parse_failed"}
    sys.stdout.write(json.dumps(answer, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
