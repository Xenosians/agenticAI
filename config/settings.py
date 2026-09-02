from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================================
    # AI service
    # ============================================================

    ai_host: str = "127.0.0.1"

    ai_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    # ============================================================
    # Agent definitions
    # ============================================================

    agents_dir: Path = (
        PROJECT_ROOT
        / "subagents"
        / "agents"
    )

    # ============================================================
    # Main Hub
    #
    # Ministral 3B
    # ============================================================

    hub_backend: str = "ministral"

    hub_model_path: Path | None = None

    hub_dequantize_fp8: bool = True

    # Used when device_map="auto" needs to offload
    # part of Ministral to disk.
    #
    # Example:
    # /tmp/itsm-ministral-offload
    hub_offload_folder: Path | None = None

    # ============================================================
    # Account specialist
    #
    # Qwen2.5-0.5B FuncCall
    # ============================================================

    account_backend: str = "qwen-funccall"

    account_model_key: str = (
        "qwen2.5-0.5b-funccall"
    )

    account_model_path: Path | None = None

    # ============================================================
    # Access specialist
    #
    # Qwen3-0.6B
    # ============================================================

    access_enabled: bool = False

    access_backend: str = "qwen3"

    access_model_key: str = "qwen3-0.6b"

    access_model_path: Path | None = None

    # ============================================================
    # Helpers
    # ============================================================

    def require_path(
        self,
        value: Path | None,
        setting_name: str,
    ) -> Path:
        """
        Resolve and validate a path that must already exist.

        Appropriate for:
        - model directories
        - agent directories

        Do NOT use this helper for the Hub offload folder,
        because that directory may legitimately not exist yet.
        The Ministral backend creates it when necessary.
        """

        if value is None:
            raise RuntimeError(
                f"{setting_name} is not configured."
            )

        path = value.expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        path = path.resolve()

        if not path.exists():
            raise RuntimeError(
                f"{setting_name} does not exist: {path}"
            )

        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()