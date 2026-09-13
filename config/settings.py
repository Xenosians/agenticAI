from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


class WorkerModelSettings(
    BaseModel
):
    """
    Configuration for one logical specialist-worker model.

    The dictionary key in WORKER_MODELS is the logical model
    name referenced by agent definition files.

    Example:

        "qwen3-0.6b": {
            "backend": "qwen3",
            "model_path": "/models/Qwen3-0.6B",
            "enabled": true
        }
    """

    backend: str

    model_path: Path

    enabled: bool = True

    @field_validator(
        "backend"
    )
    @classmethod
    def validate_backend(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:
            raise ValueError(
                "Worker model backend "
                "must not be empty."
            )

        return normalized


class Settings(
    BaseSettings
):
    """
    Central validated configuration provider for the AI service.

    Deployment-specific values enter application code through
    this object instead of being read directly from os.environ
    throughout the runtime.
    """

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
    # AI SERVICE
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
    # PHOENIX CALLBACK TRANSPORT
    # ============================================================

    phoenix_base_url: (
        str | None
    ) = None

    itsm_internal_job_token: (
        str | None
    ) = None

    phoenix_http_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
    )

    # ============================================================
    # DURABLE COMPLETION OUTBOX
    # ============================================================

    completion_outbox_path: Path = (
        Path(
            ".runtime/"
            "completion_outbox.sqlite3"
        )
    )

    outbox_retry_seconds: float = Field(
        default=5.0,
        gt=0,
    )

    outbox_batch_size: int = Field(
        default=100,
        ge=1,
    )

    # ============================================================
    # DURABLE APPROVAL STORE
    # ============================================================

    approval_store_path: Path = (
        Path(
            ".runtime/"
            "approvals.sqlite3"
        )
    )

    # ============================================================
    # PROCESS EXECUTION WORKSPACE
    # ============================================================

    process_workspace_root: Path = (
        PROJECT_ROOT
    )

    # ============================================================
    # AGENT DEFINITIONS
    # ============================================================

    agents_dir: Path = (
        PROJECT_ROOT
        / "subagents"
        / "agents"
    )

    # ============================================================
    # MAIN HUB
    #
    # The Hub is intentionally separate from worker models
    # because it has Hub-specific loading options.
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
    # SPECIALIST WORKER MODELS
    #
    # Logical model name -> worker model configuration.
    #
    # Agent definitions reference these logical keys through
    # their `model:` front-matter field.
    #
    # Adding a new specialist model no longer requires adding
    # dedicated Settings fields or editing hub.py.
    # ============================================================

    worker_models: dict[
        str,
        WorkerModelSettings,
    ] = Field(
        default_factory=dict
    )

    # ============================================================
    # DIRECTORY PROVIDER
    # ============================================================

    directory_backend: str = (
        "mock"
    )

    # ============================================================
    # LDAP / ACTIVE DIRECTORY
    # ============================================================

    ad_host: (
        str | None
    ) = None

    ad_port: (
        int | None
    ) = Field(
        default=None,
        ge=1,
        le=65535,
    )

    ad_use_ssl: bool = (
        True
    )

    ad_base_dn: (
        str | None
    ) = None

    # ------------------------------------------------------------
    # READ-ONLY BIND
    # ------------------------------------------------------------

    ad_bind_user: (
        str | None
    ) = None

    ad_bind_dn: (
        str | None
    ) = None

    ad_bind_password: (
        str | None
    ) = None

    # ------------------------------------------------------------
    # WRITE / MUTATION BIND
    # ------------------------------------------------------------

    ad_write_bind_user: (
        str | None
    ) = None

    ad_write_bind_dn: (
        str | None
    ) = None

    ad_write_bind_password: (
        str | None
    ) = None

    # ------------------------------------------------------------
    # LOGICAL ACCESS GROUP MAPPING
    # ------------------------------------------------------------

    ad_access_groups: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )

    # ------------------------------------------------------------
    # LOCAL / INTEGRATION TEST IDENTITY
    # ------------------------------------------------------------

    ad_test_user: (
        str | None
    ) = None

    # ============================================================
    # VALIDATION
    # ============================================================

    @field_validator(
        "directory_backend"
    )
    @classmethod
    def validate_directory_backend(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "ldap",
            "mock",
        }

        if normalized not in supported:
            raise ValueError(
                "DIRECTORY_BACKEND must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "worker_models"
    )
    @classmethod
    def validate_worker_models(
        cls,
        value: dict[
            str,
            WorkerModelSettings,
        ],
    ) -> dict[
        str,
        WorkerModelSettings,
    ]:
        normalized: dict[
            str,
            WorkerModelSettings,
        ] = {}

        for model_key, config in (
            value.items()
        ):
            if not isinstance(
                model_key,
                str,
            ):
                raise ValueError(
                    "WORKER_MODELS keys "
                    "must be strings."
                )

            key = (
                model_key
                .strip()
            )

            if not key:
                raise ValueError(
                    "WORKER_MODELS contains "
                    "an empty model key."
                )

            if key in normalized:
                raise ValueError(
                    "WORKER_MODELS contains "
                    f"duplicate model key '{key}'."
                )

            normalized[
                key
            ] = config

        return normalized

    @field_validator(
        "phoenix_base_url"
    )
    @classmethod
    def validate_phoenix_base_url(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = (
            value
            .strip()
            .rstrip("/")
        )

        if not normalized:
            return None

        parsed = urlsplit(
            normalized
        )

        if (
            parsed.scheme
            not in {
                "http",
                "https",
            }
        ):
            raise ValueError(
                "PHOENIX_BASE_URL must use "
                "http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "PHOENIX_BASE_URL must contain "
                "a hostname."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "PHOENIX_BASE_URL must not contain "
                "userinfo."
            )

        if parsed.query:
            raise ValueError(
                "PHOENIX_BASE_URL must not contain "
                "a query string."
            )

        if parsed.fragment:
            raise ValueError(
                "PHOENIX_BASE_URL must not contain "
                "a fragment."
            )

        if parsed.path not in {
            "",
            "/",
        }:
            raise ValueError(
                "PHOENIX_BASE_URL must be an origin "
                "without a path."
            )

        try:
            parsed.port

        except ValueError as exc:
            raise ValueError(
                "PHOENIX_BASE_URL contains "
                "an invalid port."
            ) from exc

        return normalized

    # ============================================================
    # PHOENIX HELPERS
    # ============================================================

    def require_phoenix_base_url(
        self,
    ) -> str:
        value = (
            self.phoenix_base_url
        )

        if not value:
            raise RuntimeError(
                "PHOENIX_BASE_URL "
                "is not configured."
            )

        return value

    def require_internal_job_token(
        self,
    ) -> str:
        value = (
            self.itsm_internal_job_token
        )

        if (
            value is None
            or not value.strip()
        ):
            raise RuntimeError(
                "ITSM_INTERNAL_JOB_TOKEN "
                "is not configured."
            )

        return value.strip()

    # ============================================================
    # WORKER MODEL HELPERS
    # ============================================================

    def worker_model(
        self,
        model_key: str,
    ) -> WorkerModelSettings | None:
        """
        Return worker-model configuration without requiring the
        model to be enabled or its path to exist.

        Useful while deciding which configured agents should be
        included in the runtime.
        """

        return self.worker_models.get(
            model_key
        )

    def require_worker_model(
        self,
        model_key: str,
    ) -> WorkerModelSettings:
        """
        Return an enabled worker-model configuration with a
        resolved existing model path.
        """

        config = self.worker_model(
            model_key
        )

        if config is None:
            raise RuntimeError(
                "Worker model "
                f"'{model_key}' "
                "is not configured in "
                "WORKER_MODELS."
            )

        if not config.enabled:
            raise RuntimeError(
                "Worker model "
                f"'{model_key}' "
                "is disabled."
            )

        resolved_path = (
            self.require_path(
                config.model_path,
                (
                    "WORKER_MODELS"
                    f"[{model_key}]"
                    ".model_path"
                ),
            )
        )

        return config.model_copy(
            update={
                "model_path":
                    resolved_path,
            }
        )

    # ============================================================
    # DIRECTORY HELPERS
    # ============================================================

    @property
    def ad_read_bind_identity(
        self,
    ) -> str | None:
        return (
            self.ad_bind_user
            or self.ad_bind_dn
        )

    @property
    def ad_write_bind_identity(
        self,
    ) -> str | None:
        return (
            self.ad_write_bind_user
            or self.ad_write_bind_dn
        )

    @property
    def resolved_ad_port(
        self,
    ) -> int:
        if self.ad_port is not None:
            return self.ad_port

        if self.ad_use_ssl:
            return 636

        return 389

    # ============================================================
    # PATH HELPERS
    # ============================================================

    def resolve_runtime_path(
        self,
        value: Path,
    ) -> Path:
        """
        Resolve a runtime persistence path.

        Relative runtime paths are anchored to PROJECT_ROOT
        instead of depending on the process working directory.
        """

        path = (
            value
            .expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return path.resolve()

    def resolve_project_path(
        self,
        value: Path,
    ) -> Path:
        """
        Resolve a configured project-relative path.

        Relative paths are anchored to PROJECT_ROOT instead of
        depending on the process working directory.
        """

        path = (
            value
            .expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return path.resolve()

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
            value
            .expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        path = (
            path.resolve()
        )

        if not path.exists():
            raise RuntimeError(
                f"{setting_name} "
                "does not exist: "
                f"{path}"
            )

        return path


@lru_cache
def get_settings() -> Settings:
    """
    Return the process-wide validated configuration instance.

    The process-wide Settings lifecycle will be moved into the
    application runtime container in the next cleanup stage.
    Keeping this function temporarily preserves current callers
    while worker topology is decoupled first.
    """

    return Settings()