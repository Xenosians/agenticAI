from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    Central runtime configuration.

    Model locations and runtime choices must not be
    hardcoded inside Hub/worker implementation code.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------
    # AI service
    # ---------------------------------------------------------

    ai_host: str = "127.0.0.1"

    ai_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    # ---------------------------------------------------------
    # Agent definitions
    # ---------------------------------------------------------

    agents_dir: Path = (
        PROJECT_ROOT
        / "subagents"
        / "agents"
    )

    # ---------------------------------------------------------
    # Hub
    # ---------------------------------------------------------

    hub_backend: str = "ministral"

    hub_model_path: Path | None = None

    # RTX 4050 / Ada cannot execute the current
    # fine-grained FP8 kernel directly.
    hub_dequantize_fp8: bool = True

    # ---------------------------------------------------------
    # Account worker
    # ---------------------------------------------------------

    account_backend: str = "qwen-funccall"

    account_model_key: str = (
        "qwen2.5-0.5b-funccall"
    )

    account_model_path: Path | None = None

    # ---------------------------------------------------------
    # Access worker
    # ---------------------------------------------------------

    access_enabled: bool = False

    access_backend: str = "qwen-funccall"

    access_model_key: str = "access-model-tbd"

    access_model_path: Path | None = None

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def require_path(
        self,
        value: Path | None,
        setting_name: str,
    ) -> Path:
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
                f"{setting_name} does not exist: "
                f"{path}"
            )

        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()