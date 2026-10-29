"""Required Lua admission for local tokenizer, base and adapter weight loads."""
from pathlib import Path

from services.lua_adapters.host_boundaries import inspect_data


def from_pretrained(load, *args, **kwargs):
    # Credentials, runtime objects and quantizer internals remain in the host.
    artifacts = [str(value) for value in args if isinstance(value, (str, Path))]
    options = {key: str(kwargs[key]) for key in (
        "pretrained_model_name_or_path", "model_id", "adapter_id", "revision",
        "local_files_only", "trust_remote_code", "device_map", "torch_dtype", "dtype",
    ) if key in kwargs and isinstance(kwargs[key], (str, Path, bool, int, float))}
    loader = getattr(load, "__qualname__", type(load).__name__)
    inspect_data("model", "warm", {"artifacts": artifacts, "options": options},
                 {"loader": loader, "module": getattr(load, "__module__", "unknown")})
    return load(*args, **kwargs)
