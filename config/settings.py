from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODELS_ROOT = (
    PROJECT_ROOT.parent
    / "Models"
)


class Settings(
    BaseSettings
):
    model_config = (
        SettingsConfigDict(
            env_file=(
                PROJECT_ROOT
                / ".env"
            ),
            env_file_encoding=(
                "utf-8"
            ),
            case_sensitive=False,
            extra="ignore",
        )
    )

    # ============================================================
    # AI service
    # ============================================================

    ai_host: str = (
        "127.0.0.1"
    )

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

    hub_backend: str = (
        "ministral"
    )

    hub_model_path: (
        Path | None
    ) = None

    hub_dequantize_fp8: bool = (
        True
    )

    hub_offload_folder: (
        Path | None
    ) = None

    # ============================================================
    # Account specialist
    #
    # Qwen2.5-0.5B FuncCall
    # ============================================================

    account_backend: str = (
        "qwen-funccall"
    )

    account_model_key: str = (
        "qwen2.5-0.5b-funccall"
    )

    account_model_path: (
        Path | None
    ) = None

    # ============================================================
    # Access specialist
    #
    # Qwen3-0.6B
    # ============================================================

    access_enabled: bool = (
        False
    )

    access_backend: str = (
        "qwen3"
    )

    access_model_key: str = (
        "qwen3-0.6b"
    )

    access_model_path: (
        Path | None
    ) = None

    # ============================================================
    # Developer specialist
    #
    # Qwen2.5-Coder-0.5B-Instruct
    #
    # Registered lazily.
    # The model is NOT loaded during application startup.
    # ============================================================

    developer_enabled: bool = (
        True
    )

    developer_backend: str = (
        "qwen-coder"
    )

    developer_model_key: str = (
        "qwen2.5-coder-0.5b"
    )

    developer_model_path: (
        Path | None
    ) = (
        MODELS_ROOT
        / "Qwen2.5-Coder-0.5B-Instruct"
    )

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

        Do NOT use for directories that may legitimately be
        created later, such as the Hub offload directory.
        """

        if value is None:
            raise RuntimeError(
                f"{setting_name} "
                "is not configured."
            )

        path = (
            value.expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        path = path.resolve()

        if not path.exists():
            raise RuntimeError(
                f"{setting_name} "
                "does not exist: "
                f"{path}"
            )

        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()