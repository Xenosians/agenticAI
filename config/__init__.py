import os
from pathlib import Path

from dotenv import load_dotenv

from .settings import Settings, get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Preserve the old behavior for modules that still use
# get_env()/get_bool_env().
load_dotenv(PROJECT_ROOT / ".env")


def get_env(
    name: str,
    default: str | None = None,
) -> str | None:
    return os.getenv(name, default)


def get_bool_env(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}"
    )


def get_int_env(
    name: str,
    default: int | None = None,
) -> int | None:
    value = os.getenv(name)

    if value is None:
        return default

    return int(value)


def get_float_env(
    name: str,
    default: float | None = None,
) -> float | None:
    value = os.getenv(name)

    if value is None:
        return default

    return float(value)


__all__ = [
    "Settings",
    "get_settings",
    "get_env",
    "get_bool_env",
    "get_int_env",
    "get_float_env",
]