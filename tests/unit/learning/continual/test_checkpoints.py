from pathlib import Path

import pytest

from learning.continual.checkpoints import AdapterCheckpointStore


def test_checkpoint_registration_is_hash_pinned(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    store = AdapterCheckpointStore(
        root=tmp_path / "checkpoints",
        active_pointer=tmp_path / "active.json",
        history_path=tmp_path / "history.jsonl",
        promotion_root=tmp_path / "promotions",
    )
    manifest = store.register(
        adapter_directory=adapter,
        base_model_sha256="a" * 64,
        source_cycle_id="cycle-1",
        source_split_id="split-1",
        target_agent="account-specialist",
        target_model_key="qwen2.5-0.5b-funccall",
        label="test",
    )
    store.verify_adapter(manifest.checkpoint_id)

    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        store.verify_adapter(manifest.checkpoint_id)
