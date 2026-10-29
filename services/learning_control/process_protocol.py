"""Standard-library-only durability helpers shared with the trainer launcher."""
from __future__ import annotations

import json
import os
import ctypes
import errno
import platform
import signal
from pathlib import Path
import uuid


def _syscall(number: int, *arguments) -> int:
    """Linux pidfd ABI for Python builds compiled without its optional wrappers."""
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "aarch64"}:
        raise OSError(errno.ENOSYS, "Process handles are unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def open_process_handle(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    return native(pid) if native is not None else _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def signal_process_handle(handle: int, signum: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(handle, signum)
    else:
        _syscall(424, ctypes.c_int(handle), ctypes.c_int(signum), ctypes.c_void_p(None), ctypes.c_uint(0))


def process_identity(pid: int) -> str | None:
    """Pin the kernel boot and start tick; a reused PID is a different process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot}:{fields[19]}"
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (IndexError, ValueError) as exc:
        raise OSError("Process identity is unreadable") from exc


def write_json(path: Path, value: dict) -> None:
    """Persist a complete record before publishing its name."""
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
