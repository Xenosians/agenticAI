"""The packaged Lua transform must honor Phoenix's versioned v1 contracts."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from services.extensions import extension_contracts, transform
from services.extensions.extension_contracts import ExtensionContractError, load_extension_contract

CONTRACT_FILES = (
    "manifest.schema.json",
    "invocation.schema.json",
    "result.schema.json",
    "lua-row-transform.manifest.json",
)


def _copy_contracts(destination: Path) -> Path:
    shutil.copytree(extension_contracts.CONTRACTS_ROOT, destination)
    return destination


def _refuse_before_worker(monkeypatch):
    def unexpected_worker(*args, **kwargs):
        raise AssertionError("Lua worker must not run for a contract refusal")

    monkeypatch.setattr(transform.subprocess, "run", unexpected_worker)


def test_packaged_contract_files_match_phoenix_canonical_files():
    workspace = Path(__file__).resolve().parents[5]
    canonical = workspace / "agenticBackend/itsm_backend/contracts/v1/extensions"
    if not canonical.is_dir():
        pytest.skip("the sibling Phoenix repository is not present in this checkout")

    for filename in CONTRACT_FILES:
        assert (extension_contracts.CONTRACTS_ROOT / filename).read_bytes() == (canonical / filename).read_bytes()


def test_declared_manifest_and_schema_references_load():
    manifest, invocation, result = load_extension_contract()

    assert manifest["id"] == "lua-row-transform"
    assert manifest["version"] == "1.0.0"
    assert manifest["input_schema"] == invocation["$id"]
    assert manifest["output_schema"] == result["$id"]


@pytest.mark.parametrize("change", [
    lambda manifest: manifest["limits"].update(max_input_bytes=2_000_001),
    lambda manifest: manifest.update(contract_version=2),
    lambda manifest: manifest.update(id="unregistered-transform"),
])
def test_invalid_incompatible_or_undeclared_manifest_is_refused_before_lua(tmp_path, monkeypatch, change):
    root = _copy_contracts(tmp_path / "contracts")
    path = root / extension_contracts.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(extension_contracts, "CONTRACTS_ROOT", root)
    _refuse_before_worker(monkeypatch)

    with pytest.raises(transform.TransformError, match="extension_manifest_invalid") as caught:
        transform.run_transform("return function(row) return row end", '{"ticket":"SUP-12"}')

    assert caught.value.code == "extension_manifest_invalid"


def test_invocation_schema_mismatch_is_refused_before_lua(tmp_path, monkeypatch):
    root = _copy_contracts(tmp_path / "contracts")
    path = root / "invocation.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    schema["properties"]["input"]["properties"]["script"]["const"] = "approved script"
    path.write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setattr(extension_contracts, "CONTRACTS_ROOT", root)
    _refuse_before_worker(monkeypatch)

    with pytest.raises(transform.TransformError) as caught:
        transform.run_transform("return function(row) return row end", '{"ticket":"SUP-12"}')

    assert caught.value.code == "extension_invocation_invalid"


def test_malformed_worker_result_is_refused_by_the_declared_result_schema(monkeypatch):
    def invalid_result(*args, **kwargs):
        answer = {"rows": [{"ticket": "SUP-12"}], "elapsed_ms": "fast", "isolation": []}
        return subprocess.CompletedProcess(args[0], 0, json.dumps(answer).encode(), b"")

    monkeypatch.setattr(transform.subprocess, "run", invalid_result)

    with pytest.raises(transform.TransformError) as caught:
        transform.run_transform("return function(row) return row end", '{"ticket":"SUP-12"}')

    assert caught.value.code == "extension_result_invalid"


def test_configured_worker_limits_cannot_exceed_manifest_ceilings():
    with pytest.raises(transform.TransformError) as caught:
        transform.run_transform(
            "return function(row) return row end",
            '{"ticket":"SUP-12"}',
            transform.TransformLimits(instructions=20_000_001),
        )

    assert caught.value.code == "extension_limits_invalid"
