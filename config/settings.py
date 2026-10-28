from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from config.path_portability import (
    resolve_portable_path,
)

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from pydantic_settings import (
    BaseSettings,
    NoDecode,
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

    # Optional PEFT/LoRA adapter layered over model_path at runtime.
    #
    # This lets specialist candidates share the base checkpoint
    # instead of requiring a second fully merged model copy.
    adapter_path: (
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
    # compact_grounded:
    #   Opt-in candidate protocol without example target identifiers.
    # ============================================================

    worker_prompt_profile: str = (
        "standard"
    )

    # Hub routing uses its own prompt contract. Compact routing is an explicit
    # serving choice; it does not change specialist prompts or train/promote an
    # adapter, and its evidence is not admitted to the standard Hub curriculum.
    router_prompt_profile: str = "standard"

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
        "worker_prompt_profile", "router_prompt_profile"
    )
    @classmethod
    def validate_prompt_profile(
        cls,
        value: str,
        info: ValidationInfo,
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

        if info.field_name == "worker_prompt_profile":
            supported.add("compact_grounded")

        if normalized not in supported:
            raise ValueError(
                "prompt profile must be one of: "
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

    job_websocket_enabled: bool = True
    job_metrics_enabled: bool = True

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

    # Separate, opt-in local SMTP/IMAP lab. The model cannot choose hosts,
    # container names, commands, ports, or a public mail domain.
    mailbox_lab_enabled: bool = False
    mailbox_lab_domain: str = "itsm.test"
    mailbox_smtp_port: int = Field(default=1587, ge=1024, le=65535)
    mailbox_imap_port: int = Field(default=1143, ge=1024, le=65535)
    mailbox_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    mailbox_provisioning_lock_path: Path = PROJECT_ROOT / ".runtime/mailbox-provisioning.lock"

    @field_validator("mailbox_lab_domain")
    @classmethod
    def validate_mailbox_lab_domain(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+test", value) or len(value) > 253:
            raise ValueError("MAILBOX_LAB_DOMAIN must be an exact lowercase .test domain.")
        return value

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

    learning_context_path: Path = (
        Path(".runtime/learning/continual/context-records.jsonl")
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

    # ============================================================
    # NETWORKING (OpenWrt, read only)
    # ============================================================

    openwrt_base_url: str | None = None
    openwrt_username: str | None = None
    openwrt_password: str | None = None
    openwrt_verify_tls: bool = True
    openwrt_allow_loopback_http: bool = False
    openwrt_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    # ============================================================
    # PLATFORM OPERATIONS (runtime containers, service health)
    # ============================================================

    # Exact container names the operations tools may read; empty disables them.
    operations_containers: Annotated[list[str], NoDecode] = Field(default_factory=list)
    operations_docker_socket: Path = Path("/var/run/docker.sock")
    # name=url pairs. Empty derives the Phoenix and AI health URLs from the
    # PHOENIX_BASE_URL and AI_HOST/AI_PORT settings.
    operations_health_urls: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)
    operations_timeout_seconds: float = Field(default=3.0, gt=0, le=10)

    @field_validator("operations_containers", mode="before")
    @classmethod
    def parse_operations_containers(cls, value) -> list[str]:
        import re
        items = value.split(",") if isinstance(value, str) else list(value or [])
        names: list[str] = []
        for item in items:
            name = str(item).strip()
            if not name:
                continue
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
                raise ValueError("OPERATIONS_CONTAINERS must list exact Docker container names separated by commas.")
            if name not in names:
                names.append(name)
        return names

    @field_validator("operations_health_urls", mode="before")
    @classmethod
    def parse_operations_health_urls(cls, value) -> dict[str, str]:
        import re
        if isinstance(value, str):
            pairs = [item.split("=", 1) for item in value.split(",") if item.strip()]
            if any(len(pair) != 2 for pair in pairs):
                raise ValueError("OPERATIONS_HEALTH_URLS must be name=url pairs separated by commas.")
            value = {name.strip(): url.strip() for name, url in pairs}
        urls: dict[str, str] = {}
        for name, url in dict(value or {}).items():
            parsed = urlsplit(url)
            # Health probes never carry credentials, so a URL that embeds any is refused.
            if (not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", name) or parsed.scheme not in {"http", "https"}
                    or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError(
                    "OPERATIONS_HEALTH_URLS entries need a short lowercase name and an http(s) URL "
                    "without credentials, query or fragment."
                )
            urls[name] = url
        if len(urls) > 10:
            raise ValueError("OPERATIONS_HEALTH_URLS accepts at most 10 endpoints.")
        return urls

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

    # ============================================================
    # MODEL SERVING CACHE / PHYSICAL-BASE REUSE
    #
    # Logical model identities remain unchanged. These settings only
    # control deployment/runtime materialization.
    # ============================================================

    model_shared_base_enabled: bool = (
        False
    )

    model_artifact_cache_enabled: bool = (
        False
    )

    model_artifact_cache_root: Path = (
        PROJECT_ROOT
        .parent
        / "model-cache"
    )

    model_artifact_cache_materialize_quantized: bool = (
        True
    )

    model_artifact_cache_prefetch: bool = (
        False
    )

    # Release unused CUDA allocator blocks after inference while
    # retaining all live model tensors.
    #
    # Useful on constrained accelerators where physical model reuse
    # intentionally keeps the base model resident between requests.
    model_trim_accelerator_cache_after_generation: bool = (
        False
    )

    # Keep the Hub permanently resident when deployment capacity
    # permits it.
    #
    # Constrained deployments may disable this and use the same
    # generic LRU residency mechanism for the Hub as for specialists.
    #
    # This is a deployment/resource policy only. It does not affect
    # routing, semantics, authorization, or model identity.
    model_pin_hub: bool = (
        True
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

    # Within the SYNTHESIS stage ceiling (1024); the prompt asks for a quarter
    # of this in words so formatted answers finish instead of being cut off.
    primary_synthesis_max_new_tokens: int = Field(
        default=768,
        ge=1,
    )

    # ============================================================
    # DIRECTORY PROVIDER
    # ============================================================

    directory_backend: str = (
        "unconfigured"
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
        "unconfigured"
    )

    # ============================================================
    # ASSET / DEVICE PROVIDER
    # ============================================================

    asset_backend: str = (
        "unconfigured"
    )
    asset_sqlite_path: Path | None = None

    # ============================================================
    # KNOWLEDGE / RUNBOOK PROVIDER
    # ============================================================

    knowledge_snapshot_path: str | None = None
    knowledge_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    knowledge_backend: str = (
        "unconfigured"
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

    atlassian_jira_resource_ari: (
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

        normalized = (
            value
            .strip()
            .lower()
            .strip(".")
        )

        # Optional configuration may appear in .env as:
        #
        #   ACCOUNT_EMAIL_DOMAIN=
        #
        # Treat that the same as omission. LDAP-backed account creation
        # may derive the domain from trusted directory configuration.
        if not normalized:
            return None

        if "." not in normalized:
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
            "unconfigured",
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
            "unconfigured",
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
            "sqlite",
            "unconfigured",
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

    @field_validator("asset_sqlite_path", mode="before")
    @classmethod
    def validate_asset_sqlite_path(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if str(value).strip() == ":memory:":
            raise ValueError("ASSET_SQLITE_PATH must identify a persistent database file.")
        return resolve_portable_path(value, base=PROJECT_ROOT)

    @model_validator(mode="after")
    def validate_asset_storage(self):
        if self.asset_backend == "sqlite" and self.asset_sqlite_path is None:
            raise ValueError("ASSET_SQLITE_PATH is required when ASSET_BACKEND=sqlite.")
        return self

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
            "local",
            "unconfigured",
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

        adapter_path = (
            profile.adapter_path
        )

        resolved_adapter_path = None

        if adapter_path is not None:
            resolved_adapter_path = (
                self.require_path(
                    adapter_path,
                    (
                        "MODEL_PROFILES"
                        f"[{model_key}]"
                        ".adapter_path"
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

                    "adapter_path":
                        resolved_adapter_path,

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
        return (
            resolve_portable_path(
                value,
                base=PROJECT_ROOT,
            )
        )

    def resolve_project_path(
        self,
        value: Path,
    ) -> Path:
        return (
            resolve_portable_path(
                value,
                base=PROJECT_ROOT,
            )
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
            resolve_portable_path(
                value,
                base=PROJECT_ROOT,
            )
        )

        if not path.exists():
            raise RuntimeError(
                f"{setting_name} "
                "does not exist: "
                f"{path}"
            )

        return path


# Provider-endpoint policy is read from os.environ by the extension transport
# (it is re-read per request so an operator change applies without a code
# path through Settings). Export these from .env at startup so a value in
# .env behaves the same as one set in the service environment.
PROCESS_ENV_FROM_DOTENV = (
    "EXTENSION_ALLOWED_ORIGINS",
    "EXTENSION_INSECURE_HTTP_ORIGINS",
)


def export_process_env_from_dotenv(
    path: Path = PROJECT_ROOT / ".env",
) -> list[str]:
    """Copy the provider-endpoint policy from .env into os.environ.

    A value already present in the process environment wins. Returns the
    names that were exported (never their values).
    """
    import os

    from dotenv import dotenv_values

    if not path.is_file():
        return []
    values = dotenv_values(path)
    exported = []
    for name in PROCESS_ENV_FROM_DOTENV:
        value = values.get(name)
        if value and name not in os.environ:
            os.environ[name] = value
            exported.append(name)
    return exported


@lru_cache
def get_settings(
) -> Settings:
    """
    Return the process-cached validated Settings instance.
    """

    export_process_env_from_dotenv()
    return Settings()
