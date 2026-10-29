"""The `dataset_transform` extension action: a Lua script reshapes the owner's rows before admission.

A preview admits nothing. Applying admits the output exactly like any upload,
so the same secret/PII scan and training checks apply to what the script made.
The public boundary carries only a transform code and a bounded message.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("lupa.lua54")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import extensions  # noqa: E402
from services.extensions.service import command, extension_store  # noqa: E402

TOKEN = "transform-test-token"
RAW = "\n".join(json.dumps({"subject": f"Printer jam {index}", "fix": f"Open tray {index}"}) for index in range(12))
TO_MESSAGES = """
return function(row)
  return {messages = {{role = "user", content = row.subject}, {role = "assistant", content = row.fix}}}
end
"""
SETTINGS = SimpleNamespace(model_profiles={}, require_internal_job_token=lambda: TOKEN)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    extension_store.cache_clear()
    try:
        yield extension_store()
    finally:
        extension_store.cache_clear()


def transform(owner="alice", **payload):
    body = {"script": TO_MESSAGES, "text": RAW, **payload}
    return asyncio.run(command(owner, "dataset_transform", body, SETTINGS))


def apply(owner="alice", **payload):
    fields = {"apply": True, "name": "Printer fixes", "source": "Owned tickets", "license": "Owned",
              "purpose": "training", "reviewed": True}
    return transform(owner, **(fields | payload))


def test_a_preview_shows_the_first_rows_and_admits_nothing(store):
    preview = transform()

    assert (preview["input_rows"], preview["output_rows"]) == (12, 12)
    assert len(preview["preview"]) == 5
    assert preview["preview"][0] == {"messages": [{"role": "user", "content": "Printer jam 0"},
                                                  {"role": "assistant", "content": "Open tray 0"}]}
    assert "text" not in preview
    assert store.list("alice") == []


def test_applying_admits_the_output_as_an_ordinary_versioned_dataset_with_its_provenance(store):
    row = apply()

    assert (row["kind"], row["purpose"], row["state"], row["version"]) == ("dataset", "training", "ready", 1)
    stored = store.get("alice", row["id"])
    assert len(stored["content"]["train"]) + len(stored["content"]["validation"]) == 12
    assert stored["transform"]["schema"] == "lua-row-transform.v1"
    assert (stored["transform"]["input_rows"], stored["transform"]["output_rows"]) == (12, 12)
    assert len(stored["transform"]["script_sha256"]) == len(stored["transform"]["input_sha256"]) == 64
    # Only another owner's view is empty: transforms are owner-scoped like every dataset.
    assert store.list("bob") == []


def test_what_a_script_produces_still_passes_secret_and_training_admission(store):
    leak = 'return function(row) return {messages = {{role = "user", content = row.subject}, ' \
           '{role = "assistant", content = "password=" .. "hunter2-secret"}}} end'
    with pytest.raises(ValueError, match="secret"):
        apply(script=leak)
    with pytest.raises(ValueError, match="messages"):
        apply(script="return function(row) return {text = row.subject} end")
    assert store.list("alice") == []


def test_unreviewed_or_unlicensed_output_is_refused_before_anything_is_stored(store):
    with pytest.raises(ValueError):
        apply(reviewed=False)
    with pytest.raises(ValueError):
        apply(license="")
    assert store.list("alice") == []


@pytest.mark.parametrize("payload", [{"script": 7}, {"script": None}, {"apply": "yes"}, {"apply": 1}])
def test_malformed_requests_are_refused(store, payload):
    from services.extensions.transform import TransformError
    with pytest.raises(TransformError) as caught:
        transform(**payload)
    assert caught.value.code == "transform_request_invalid"


def test_rows_that_are_not_text_are_refused_like_every_text_field(store):
    with pytest.raises(ValueError, match="Expected a text field"):
        transform(text=None)


def test_the_catalog_says_whether_lua_transforms_are_available(store):
    catalog = asyncio.run(command("alice", "list", {}, SimpleNamespace(
        model_profiles={}, require_internal_job_token=lambda: TOKEN, tool_registry=None)))
    assert {"id": "lua-row-transform", "available": True} in catalog["transforms"]


def test_the_public_boundary_carries_only_the_transform_code_and_message(store, monkeypatch):
    monkeypatch.setattr(extensions, "get_settings", lambda: SETTINGS)
    app = FastAPI()
    app.include_router(extensions.router)
    client = TestClient(app)
    body = {"owner": "alice", "action": "dataset_transform",
            "payload": {"script": "return function(row) error('row is wrong') end", "text": RAW}}

    reply = client.post("/internal/v1/extensions", json=body, headers={"x-internal-token": TOKEN})

    assert reply.status_code == 422
    detail = reply.json()["detail"]
    assert set(detail) == {"code", "message"} and detail["code"] == "transform_failed"
    assert detail["message"].startswith("Row 1:") and "row is wrong" in detail["message"]
    assert "Traceback" not in reply.text and "stack" not in reply.text
