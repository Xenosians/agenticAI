import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def get_env(
    name: str,
    default: str | None = None,
) -> str | None:
    return os.environ.get(
        name,
        default,
    )


def get_bool_env(
    name: str,
    default: bool = False,
) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }