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


class ModelProfileSettings(
    BaseModel
):
    """
    Configuration for one logical model profile.

    Roles such as Hub, Account, Access, or Developer refer to
    logical model keys instead of directly depending on model
    brands, filesystem paths, quantization methods, or backend
    implementations.

    Deployment-specific runtime choices belong here.
    """

    backend: str

    model_path: (
        Path | None
    ) = None

    enabled: bool = True

    # ============================================================
    # MODEL RUNTIME PROFILE
    # ============================================================

    # auto:
    #   Detect checkpoint-native quantization where supported.
    #
    # none:
    #   Load without additional quantization.
    #
    # fp8:
    #   Transformers FineGrainedFP8Config.
    #
    # bnb4:
    #   bitsandbytes 4-bit quantization.
    quantization: str = (
        "auto"
    )

    compute_dtype: str = (
        "bfloat16"
    )

    # Model parameter storage/loading dtype.
    #
    # "auto" preserves checkpoint/backend-native dtype behavior.
    model_dtype: str = (
        "auto"
    )

    device_map: (
        str | None
    ) = (
        "auto"
    )

    # ============================================================
    # WORKER PROMPT PROFILE
    #
    # standard:
    #   Full descriptive capability protocol for larger models.
    #
    # compact:
    #   Bounded tool-call protocol intended for small specialist
    #   models. Authority and execution semantics remain unchanged.
    # ============================================================

    worker_prompt_profile: str = (
        "standard"
    )

    # ============================================================
    # FP8 OPTIONS
    # ============================================================

    dequantize_fp8: bool = (
        True
    )

    # ============================================================
    # BITSANDBYTES 4-BIT OPTIONS
    # ============================================================

    bnb_4bit_quant_type: str = (
        "nf4"
    )

    bnb_4bit_use_double_quant: bool = (
        True
    )

    # ============================================================
    # OPTIONAL OFFLOAD
    # ============================================================

    offload_folder: (
        Path | None
    ) = None

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
                "Model backend must not be empty."
            )

        return normalized

    @field_validator(
        "worker_prompt_profile"
    )
    @classmethod
    def validate_worker_prompt_profile(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "standard",
            "compact",
        }

        if normalized not in supported:
            raise ValueError(
                "worker_prompt_profile must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "quantization"
    )
    @classmethod
    def validate_quantization(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "auto",
            "bnb4",
            "fp8",
            "none",
        }

        if normalized not in supported:
            raise ValueError(
                "Model quantization must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "compute_dtype"
    )
    @classmethod
    def validate_compute_dtype(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "bfloat16",
            "float16",
            "float32",
        }

        if normalized not in supported:
            raise ValueError(
                "Model compute_dtype must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "model_dtype"
    )
    @classmethod
    def validate_model_dtype(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "auto",
            "bfloat16",
            "float16",
            "float32",
        }

        if normalized not in supported:

            raise ValueError(
                "Model model_dtype must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "bnb_4bit_quant_type"
    )
    @classmethod
    def validate_bnb_quant_type(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "fp4",
            "nf4",
        }

        if normalized not in supported:
            raise ValueError(
                "bnb_4bit_quant_type must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "device_map"
    )
    @classmethod
    def validate_device_map(
        cls,
        value: (
            str | None
        ),
    ) -> (
        str | None
    ):
        if value is None:
            return None

        normalized = (
            value.strip()
        )

        if not normalized:
            return None

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
    # DURABLE JOB LIVENESS
    # ============================================================

    job_heartbeat_interval_seconds: float = Field(
        default=30.0,
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
    # LEARNING / TRAJECTORY CAPTURE
    #
    # Raw runtime trajectories are evidence only.
    #
    # They are sanitized before persistence and are NOT
    # automatically eligible for model training.
    # ============================================================

    learning_capture_enabled: bool = (
        False
    )

    learning_trajectory_path: Path = (
        Path(
            ".runtime/"
            "learning/"
            "trajectories.jsonl"
        )
    )
    
    learning_correction_path: Path = (
        Path(
            ".runtime/"
            "learning/"
            "corrections.jsonl"
        )
    )

    # ============================================================
    # DEVELOPER WORKSPACE
    # ============================================================

    process_workspace_root: Path = (
        PROJECT_ROOT
    )

    # ============================================================
    # GOVERNED GIT REPOSITORIES
    #
    # Model-facing Git capabilities receive logical repository
    # identifiers only.
    #
    # Physical paths remain trusted configuration.
    #
    # Relative repository paths are resolved beneath
    # PROCESS_WORKSPACE_ROOT.
    # ============================================================

    git_default_repository: str = (
        "ai"
    )

    git_repositories: dict[
        str,
        Path,
    ] = Field(
        default_factory=lambda: {
            "ai":
                Path("."),
        }
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
    # MODEL PROFILES
    # ============================================================

    hub_model_key: str = (
        "hub-main"
    )

    model_profiles: dict[
        str,
        ModelProfileSettings,
    ] = Field(
        default_factory=dict
    )

    # ============================================================
    # MODEL RESIDENCY POLICY
    #
    # GpuScheduler serializes accelerator work. Residency is a
    # separate concern: without an explicit bound, sequential use
    # of multiple specialists can leave every backend cached and
    # eventually exhaust VRAM / WSL memory.
    #
    # The default keeps the Hub resident plus one active specialist.
    # Non-pinned models are evicted using LRU when capacity is full.
    # ============================================================

    model_max_loaded_models: int = Field(
        default=2,
        ge=1,
    )

    # The configured HUB_MODEL_KEY is always pinned by ModelManager.
    # This list is for additional deployment-specific pinned models.
    model_pinned_keys: list[str] = Field(
        default_factory=list
    )

    # ============================================================
    # GENERATION POLICY
    #
    # Generation budgets are deployment/runtime configuration.
    #
    # Orchestration consumes these values instead of owning
    # model-size or machine-specific token assumptions.
    # ============================================================

    hub_router_max_new_tokens: int = Field(
        default=512,
        ge=1,
    )

    specialist_max_new_tokens: int = Field(
        default=256,
        ge=1,
    )

    primary_response_max_new_tokens: int = Field(
        default=384,
        ge=1,
    )

    primary_synthesis_max_new_tokens: int = Field(
        default=512,
        ge=1,
    )

    # ============================================================
    # DIRECTORY PROVIDER
    # ============================================================

    directory_backend: str = (
        "mock"
    )

    # ============================================================
    # CORPORATE ACCOUNT CREATION
    # ============================================================

    account_email_domain: (
        str | None
    ) = None

    ad_account_container_dn: (
        str | None
    ) = None

    account_temporary_password_length: int = Field(
        default=24,
        ge=16,
        le=128,
    )

    # ============================================================
    # TICKETING PROVIDER
    # ============================================================

    ticketing_backend: str = (
        "mock"
    )

    # ============================================================
    # ASSET / DEVICE PROVIDER
    # ============================================================

    asset_backend: str = (
        "mock"
    )

    # ============================================================
    # KNOWLEDGE / RUNBOOK PROVIDER
    # ============================================================

    knowledge_backend: str = (
        "mock"
    )

    # ============================================================
    # JIRA
    # ============================================================

    jira_base_url: (
        str | None
    ) = None

    jira_email: (
        str | None
    ) = None

    jira_api_token: (
        str | None
    ) = None

    jira_http_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
    )

    # ============================================================
    # ATLASSIAN ADMIN FOUNDATION
    # ============================================================

    atlassian_admin_api_key: (
        str | None
    ) = None

    atlassian_org_id: (
        str | None
    ) = None

    atlassian_admin_http_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
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
    # LOGICAL RESOURCE -> AD GROUP MAPPING
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
        "account_email_domain"
    )
    @classmethod
    def validate_account_email_domain(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower().strip(".")
        if not normalized or "." not in normalized:
            raise ValueError(
                "ACCOUNT_EMAIL_DOMAIN must be a DNS-style domain."
            )
        if not all(
            part
            and part.replace("-", "").isalnum()
            for part in normalized.split(".")
        ):
            raise ValueError(
                "ACCOUNT_EMAIL_DOMAIN contains invalid labels."
            )
        return normalized

    @field_validator(
        "ad_account_container_dn"
    )
    @classmethod
    def validate_account_container_dn(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

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
        "ticketing_backend"
    )
    @classmethod
    def validate_ticketing_backend(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "jira",
            "mock",
        }

        if normalized not in supported:
            raise ValueError(
                "TICKETING_BACKEND must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "asset_backend"
    )
    @classmethod
    def validate_asset_backend(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "mock",
        }

        if normalized not in supported:

            raise ValueError(
                "ASSET_BACKEND must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "knowledge_backend"
    )
    @classmethod
    def validate_knowledge_backend(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        supported = {
            "mock",
        }

        if normalized not in supported:

            raise ValueError(
                "KNOWLEDGE_BACKEND must be one of: "
                + ", ".join(
                    sorted(
                        supported
                    )
                )
            )

        return normalized

    @field_validator(
        "hub_model_key"
    )
    @classmethod
    def validate_hub_model_key(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .strip()
        )

        if not normalized:
            raise ValueError(
                "HUB_MODEL_KEY must not be empty."
            )

        return normalized

    @field_validator(
        "model_profiles"
    )
    @classmethod
    def validate_model_profiles(
        cls,
        value: dict[
            str,
            ModelProfileSettings,
        ],
    ) -> dict[
        str,
        ModelProfileSettings,
    ]:
        normalized: dict[
            str,
            ModelProfileSettings,
        ] = {}

        for (
            model_key,
            profile,
        ) in value.items():
            if not isinstance(
                model_key,
                str,
            ):
                raise ValueError(
                    "MODEL_PROFILES keys "
                    "must be strings."
                )

            key = (
                model_key
                .strip()
            )

            if not key:
                raise ValueError(
                    "MODEL_PROFILES contains "
                    "an empty model key."
                )

            if key in normalized:
                raise ValueError(
                    "MODEL_PROFILES contains "
                    f"duplicate model key '{key}'."
                )

            normalized[
                key
            ] = profile

        return normalized

    @field_validator(
        "model_pinned_keys"
    )
    @classmethod
    def validate_model_pinned_keys(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized: list[str] = []

        for model_key in value:
            if not isinstance(
                model_key,
                str,
            ):
                raise ValueError(
                    "MODEL_PINNED_KEYS values must be strings."
                )

            key = (
                model_key
                .strip()
            )

            if not key:
                raise ValueError(
                    "MODEL_PINNED_KEYS contains an empty key."
                )

            if key not in normalized:
                normalized.append(
                    key
                )

        return normalized

    @field_validator(
        "phoenix_base_url"
    )
    @classmethod
    def validate_phoenix_base_url(
        cls,
        value: (
            str | None
        ),
    ) -> (
        str | None
    ):
        if value is None:
            return None

        normalized = (
            value
            .strip()
            .rstrip("/")
        )

        if not normalized:
            return None

        parsed = (
            urlsplit(
                normalized
            )
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

    @field_validator(
        "jira_base_url"
    )
    @classmethod
    def validate_jira_base_url(
        cls,
        value: (
            str | None
        ),
    ) -> (
        str | None
    ):
        if value is None:
            return None

        normalized = (
            value
            .strip()
            .rstrip("/")
        )

        if not normalized:
            return None

        parsed = (
            urlsplit(
                normalized
            )
        )

        if (
            parsed.scheme
            not in {
                "http",
                "https",
            }
        ):
            raise ValueError(
                "JIRA_BASE_URL must use "
                "http or https."
            )

        if not parsed.hostname:
            raise ValueError(
                "JIRA_BASE_URL must contain "
                "a hostname."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "JIRA_BASE_URL must not contain "
                "credentials."
            )

        if parsed.query:
            raise ValueError(
                "JIRA_BASE_URL must not contain "
                "a query string."
            )

        if parsed.fragment:
            raise ValueError(
                "JIRA_BASE_URL must not contain "
                "a fragment."
            )

        if parsed.path not in {
            "",
            "/",
        }:
            raise ValueError(
                "JIRA_BASE_URL must be an origin "
                "without a path."
            )

        try:
            parsed.port

        except ValueError as exc:
            raise ValueError(
                "JIRA_BASE_URL contains "
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

        return (
            value.strip()
        )

    # ============================================================
    # TICKETING HELPERS
    # ============================================================

    def require_jira_base_url(
        self,
    ) -> str:
        value = (
            self.jira_base_url
        )

        if not value:
            raise RuntimeError(
                "JIRA_BASE_URL "
                "is not configured."
            )

        return value

    def require_jira_email(
        self,
    ) -> str:
        value = (
            self.jira_email
        )

        if (
            value is None
            or not value.strip()
        ):
            raise RuntimeError(
                "JIRA_EMAIL "
                "is not configured."
            )

        return (
            value.strip()
        )

    def require_jira_api_token(
        self,
    ) -> str:
        value = (
            self.jira_api_token
        )

        if (
            value is None
            or not value.strip()
        ):
            raise RuntimeError(
                "JIRA_API_TOKEN "
                "is not configured."
            )

        return (
            value.strip()
        )

    # ============================================================
    # MODEL PROFILE HELPERS
    # ============================================================

    def model_profile(
        self,
        model_key: str,
    ) -> (
        ModelProfileSettings
        | None
    ):
        return (
            self.model_profiles
            .get(
                model_key
            )
        )

    def require_model_profile(
        self,
        model_key: str,
    ) -> ModelProfileSettings:
        profile = (
            self.model_profile(
                model_key
            )
        )

        if profile is None:
            raise RuntimeError(
                "Model profile "
                f"'{model_key}' "
                "is not configured in "
                "MODEL_PROFILES."
            )

        if not profile.enabled:
            raise RuntimeError(
                "Model profile "
                f"'{model_key}' "
                "is disabled."
            )

        model_path = (
            profile.model_path
        )

        resolved_model_path = None

        if model_path is not None:
            resolved_model_path = (
                self.require_path(
                    model_path,
                    (
                        "MODEL_PROFILES"
                        f"[{model_key}]"
                        ".model_path"
                    ),
                )
            )

        offload_folder = (
            profile.offload_folder
        )

        resolved_offload_folder = None

        if offload_folder is not None:
            resolved_offload_folder = (
                self.resolve_project_path(
                    offload_folder
                )
            )

        return (
            profile.model_copy(
                update={
                    "model_path":
                        resolved_model_path,

                    "offload_folder":
                        resolved_offload_folder,
                }
            )
        )

    # ============================================================
    # DIRECTORY HELPERS
    # ============================================================

    @property
    def ad_read_bind_identity(
        self,
    ) -> (
        str | None
    ):
        return (
            self.ad_bind_user
            or self.ad_bind_dn
        )

    @property
    def ad_write_bind_identity(
        self,
    ) -> (
        str | None
    ):
        return (
            self.ad_write_bind_user
            or self.ad_write_bind_dn
        )

    @property
    def resolved_ad_port(
        self,
    ) -> int:
        if self.ad_port is not None:
            return (
                self.ad_port
            )

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
        path = (
            value
            .expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return (
            path.resolve()
        )

    def resolve_project_path(
        self,
        value: Path,
    ) -> Path:
        path = (
            value
            .expanduser()
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        return (
            path.resolve()
        )

    def require_path(
        self,
        value: (
            Path | None
        ),
        setting_name: str,
    ) -> Path:
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
def get_settings(
) -> Settings:
    """
    Return the process-cached validated Settings instance.
    """

    return Settings()