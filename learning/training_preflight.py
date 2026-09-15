from __future__ import annotations

import hashlib
import importlib.util
import json

from datetime import (
    datetime,
    timezone,
)

from importlib import (
    metadata,
)

from pathlib import (
    Path,
)

from typing import (
    Iterable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from learning.training_export import (
    TrainingSplitManifest,
)

from learning.types import (
    PreferenceDatasetRecord,
)


# ============================================================
# CONSTANTS
# ============================================================


DEFAULT_TRAINING_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "datasets",
)


TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


# ============================================================
# HELPERS
# ============================================================


def _utc_now(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _canonical_json(
    value: dict,
) -> str:

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def _sha256_bytes(
    value: bytes,
) -> str:

    return (
        hashlib
        .sha256(
            value
        )
        .hexdigest()
    )


def _sha256_file(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = (
                handle.read(
                    1024
                    * 1024
                )
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


def _load_jsonl_records(
    path: Path,
) -> list[
    PreferenceDatasetRecord
]:

    records: list[
        PreferenceDatasetRecord
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            if not line.strip():

                continue

            try:

                payload = (
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Invalid JSON in training split "
                    f"{path} at line {line_number}."
                ) from exc

            try:

                record = (
                    PreferenceDatasetRecord
                    .model_validate(
                        payload
                    )
                )

            except Exception as exc:

                raise ValueError(
                    "Invalid preference dataset record "
                    f"in {path} at line {line_number}: "
                    f"{exc}"
                ) from exc

            records.append(
                record
            )

    return records


# ============================================================
# RECIPE
# ============================================================


class DpoQloraRecipe(
    BaseModel
):
    """
    Declarative Phase-4 training recipe.

    This object describes intended training only.

    Creating or validating a recipe never loads model weights and
    never starts optimization.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "dpo-qlora-recipe.v1"
        ),
        alias="schema",
    )

    objective: str = (
        "dpo"
    )

    base_model_path: Path

    output_root: Path

    quantization: str = (
        "bnb4"
    )

    compute_dtype: str = (
        "bfloat16"
    )

    bnb_4bit_quant_type: str = (
        "nf4"
    )

    bnb_4bit_use_double_quant: bool = (
        True
    )

    lora_r: int = Field(
        default=16,
        ge=1,
    )

    lora_alpha: int = Field(
        default=32,
        ge=1,
    )

    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
    )

    beta: float = Field(
        default=0.1,
        gt=0.0,
    )

    learning_rate: float = Field(
        default=5e-6,
        gt=0.0,
    )

    num_train_epochs: float = Field(
        default=1.0,
        gt=0.0,
    )

    per_device_train_batch_size: int = Field(
        default=1,
        ge=1,
    )

    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
    )

    max_length: int = Field(
        default=2048,
        ge=128,
    )

    max_prompt_length: int = Field(
        default=1024,
        ge=64,
    )

    gradient_checkpointing: bool = (
        True
    )

    require_cuda: bool = (
        True
    )

    require_bfloat16: bool = (
        True
    )

    minimum_cuda_memory_gib: float = Field(
        default=4.0,
        ge=0.0,
    )

    @field_validator(
        "objective"
    )
    @classmethod
    def validate_objective(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if (
            normalized
            != "dpo"
        ):

            raise ValueError(
                "Phase 4A currently supports "
                "objective='dpo' only."
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

        if (
            normalized
            != "bnb4"
        ):

            raise ValueError(
                "Phase 4A currently supports "
                "quantization='bnb4' only."
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

        if normalized not in {
            "bfloat16",
            "float16",
        }:

            raise ValueError(
                "compute_dtype must be "
                "bfloat16 or float16."
            )

        return normalized

    @field_validator(
        "bnb_4bit_quant_type"
    )
    @classmethod
    def validate_quant_type(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized not in {
            "nf4",
            "fp4",
        }:

            raise ValueError(
                "bnb_4bit_quant_type must be "
                "nf4 or fp4."
            )

        return normalized


# ============================================================
# PREFLIGHT TYPES
# ============================================================


class TrainingPreflightCheck(
    BaseModel
):
    name: str

    passed: bool

    detail: str


class TrainingCudaInfo(
    BaseModel
):
    available: bool = False

    device_count: int = 0

    device_name: (
        str | None
    ) = None

    total_memory_gib: float = 0.0

    bfloat16_supported: (
        bool | None
    ) = None


class TrainingPreflightReport(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "training-preflight-report.v1"
        ),
        alias="schema",
    )

    generated_at: str

    split_id: (
        str | None
    ) = None

    source_dataset_version: (
        str | None
    ) = None

    train_record_count: int = 0

    validation_record_count: int = 0

    package_versions: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )

    cuda: TrainingCudaInfo = Field(
        default_factory=(
            TrainingCudaInfo
        )
    )

    checks: list[
        TrainingPreflightCheck
    ] = Field(
        default_factory=list
    )

    failed_checks: list[
        str
    ] = Field(
        default_factory=list
    )

    ready: bool = False


class TrainingDryRunManifest(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "training-dry-run-manifest.v1"
        ),
        alias="schema",
    )

    created_at: str

    split_id: str

    source_dataset_id: str

    source_dataset_version: str

    source_content_sha256: str

    train_sha256: str

    validation_sha256: str

    train_record_count: int

    validation_record_count: int

    recipe: DpoQloraRecipe

    package_versions: dict[
        str,
        str,
    ]

    cuda: TrainingCudaInfo

    preflight_sha256: str


# ============================================================
# PREFLIGHT
# ============================================================


class DpoQloraPreflight:
    """
    Phase-4A training readiness verifier.

    This class deliberately does NOT:

    - load model weights
    - construct a Trainer
    - allocate training tensors
    - modify adapters
    - start optimization

    It verifies immutable inputs and environment readiness only.
    """

    def __init__(
        self,
        *,
        split_dir: Path,
        recipe: DpoQloraRecipe,
        required_packages: Iterable[
            str
        ] = DEFAULT_TRAINING_PACKAGES,
        inspect_cuda: bool = True,
    ) -> None:

        self.split_dir = (
            split_dir
            .expanduser()
            .resolve()
        )

        self.recipe = (
            recipe.model_copy(
                update={
                    "base_model_path":
                        recipe
                        .base_model_path
                        .expanduser()
                        .resolve(),

                    "output_root":
                        recipe
                        .output_root
                        .expanduser()
                        .resolve(),
                }
            )
        )

        self.required_packages = (
            tuple(
                required_packages
            )
        )

        self.inspect_cuda = (
            inspect_cuda
        )

    # ========================================================
    # CHECK HELPERS
    # ========================================================

    def _check(
        self,
        *,
        name: str,
        passed: bool,
        detail: str,
    ) -> TrainingPreflightCheck:

        return (
            TrainingPreflightCheck(
                name=(
                    name
                ),

                passed=(
                    passed
                ),

                detail=(
                    detail
                ),
            )
        )

    # ========================================================
    # SPLIT VERIFICATION
    # ========================================================

    def _verify_split(
        self,
    ) -> tuple[
        TrainingSplitManifest,
        list[
            PreferenceDatasetRecord
        ],
        list[
            PreferenceDatasetRecord
        ],
        list[
            TrainingPreflightCheck
        ],
    ]:

        checks: list[
            TrainingPreflightCheck
        ] = []

        manifest_path = (
            self.split_dir
            / "manifest.json"
        )

        train_path = (
            self.split_dir
            / "train.jsonl"
        )

        validation_path = (
            self.split_dir
            / "validation.jsonl"
        )

        required_paths = [
            manifest_path,
            train_path,
            validation_path,
        ]

        missing = [
            str(
                path
            )

            for path
            in required_paths

            if not path.is_file()
        ]

        if missing:

            raise ValueError(
                "Training split is incomplete. "
                "Missing files: "
                + ", ".join(
                    missing
                )
            )

        try:

            manifest = (
                TrainingSplitManifest
                .model_validate_json(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        except Exception as exc:

            raise ValueError(
                "Training split manifest is invalid: "
                f"{exc}"
            ) from exc

        train_hash = (
            _sha256_file(
                train_path
            )
        )

        validation_hash = (
            _sha256_file(
                validation_path
            )
        )

        train_hash_valid = (
            train_hash
            == manifest.train_sha256
        )

        validation_hash_valid = (
            validation_hash
            == manifest.validation_sha256
        )

        checks.append(
            self._check(
                name=(
                    "train_sha256"
                ),

                passed=(
                    train_hash_valid
                ),

                detail=(
                    "train.jsonl matches manifest."
                    if train_hash_valid
                    else (
                        "train.jsonl SHA-256 does not "
                        "match manifest."
                    )
                ),
            )
        )

        checks.append(
            self._check(
                name=(
                    "validation_sha256"
                ),

                passed=(
                    validation_hash_valid
                ),

                detail=(
                    "validation.jsonl matches manifest."
                    if validation_hash_valid
                    else (
                        "validation.jsonl SHA-256 does "
                        "not match manifest."
                    )
                ),
            )
        )

        if (
            not train_hash_valid
            or not validation_hash_valid
        ):

            return (
                manifest,
                [],
                [],
                checks,
            )

        train_records = (
            _load_jsonl_records(
                train_path
            )
        )

        validation_records = (
            _load_jsonl_records(
                validation_path
            )
        )

        train_count_valid = (
            len(
                train_records
            )
            == manifest.train_record_count
        )

        validation_count_valid = (
            len(
                validation_records
            )
            == manifest.validation_record_count
        )

        checks.append(
            self._check(
                name=(
                    "train_record_count"
                ),

                passed=(
                    train_count_valid
                ),

                detail=(
                    "Training record count matches manifest."
                    if train_count_valid
                    else (
                        "Training record count does not "
                        "match manifest."
                    )
                ),
            )
        )

        checks.append(
            self._check(
                name=(
                    "validation_record_count"
                ),

                passed=(
                    validation_count_valid
                ),

                detail=(
                    "Validation record count matches manifest."
                    if validation_count_valid
                    else (
                        "Validation record count does not "
                        "match manifest."
                    )
                ),
            )
        )

        train_ids = [
            record.record_id

            for record
            in train_records
        ]

        validation_ids = [
            record.record_id

            for record
            in validation_records
        ]

        train_id_set = (
            set(
                train_ids
            )
        )

        validation_id_set = (
            set(
                validation_ids
            )
        )

        train_unique = (
            len(
                train_ids
            )
            == len(
                train_id_set
            )
        )

        validation_unique = (
            len(
                validation_ids
            )
            == len(
                validation_id_set
            )
        )

        checks.append(
            self._check(
                name=(
                    "unique_train_record_ids"
                ),

                passed=(
                    train_unique
                ),

                detail=(
                    "Training record IDs are unique."
                    if train_unique
                    else (
                        "Training split contains duplicate "
                        "record IDs."
                    )
                ),
            )
        )

        checks.append(
            self._check(
                name=(
                    "unique_validation_record_ids"
                ),

                passed=(
                    validation_unique
                ),

                detail=(
                    "Validation record IDs are unique."
                    if validation_unique
                    else (
                        "Validation split contains duplicate "
                        "record IDs."
                    )
                ),
            )
        )

        disjoint = (
            not (
                train_id_set
                & validation_id_set
            )
        )

        checks.append(
            self._check(
                name=(
                    "split_record_id_disjointness"
                ),

                passed=(
                    disjoint
                ),

                detail=(
                    "Train and validation record IDs are disjoint."
                    if disjoint
                    else (
                        "Record IDs overlap between train "
                        "and validation."
                    )
                ),
            )
        )

        train_manifest_ids_match = (
            train_id_set
            == set(
                manifest.train_record_ids
            )
        )

        validation_manifest_ids_match = (
            validation_id_set
            == set(
                manifest.validation_record_ids
            )
        )

        checks.append(
            self._check(
                name=(
                    "train_manifest_membership"
                ),

                passed=(
                    train_manifest_ids_match
                ),

                detail=(
                    "Training membership matches manifest."
                    if train_manifest_ids_match
                    else (
                        "Training record membership differs "
                        "from manifest."
                    )
                ),
            )
        )

        checks.append(
            self._check(
                name=(
                    "validation_manifest_membership"
                ),

                passed=(
                    validation_manifest_ids_match
                ),

                detail=(
                    "Validation membership matches manifest."
                    if validation_manifest_ids_match
                    else (
                        "Validation record membership differs "
                        "from manifest."
                    )
                ),
            )
        )

        return (
            manifest,
            train_records,
            validation_records,
            checks,
        )

    # ========================================================
    # MODEL DIRECTORY
    # ========================================================

    def _model_checks(
        self,
    ) -> list[
        TrainingPreflightCheck
    ]:

        model_path = (
            self.recipe
            .base_model_path
        )

        exists = (
            model_path.is_dir()
        )

        checks = [
            self._check(
                name=(
                    "base_model_directory"
                ),

                passed=(
                    exists
                ),

                detail=(
                    f"Base model directory: {model_path}"
                    if exists
                    else (
                        "Base model directory does not exist: "
                        f"{model_path}"
                    )
                ),
            )
        ]

        if not exists:

            return checks

        config_path = (
            model_path
            / "config.json"
        )

        has_config = (
            config_path.is_file()
        )

        checks.append(
            self._check(
                name=(
                    "base_model_config"
                ),

                passed=(
                    has_config
                ),

                detail=(
                    "config.json found."
                    if has_config
                    else (
                        "Base model directory is missing "
                        "config.json."
                    )
                ),
            )
        )

        tokenizer_found = (
            any(
                (
                    model_path
                    / filename
                ).is_file()

                for filename
                in TOKENIZER_ARTIFACTS
            )
        )

        checks.append(
            self._check(
                name=(
                    "base_model_tokenizer"
                ),

                passed=(
                    tokenizer_found
                ),

                detail=(
                    "Tokenizer artifact found."
                    if tokenizer_found
                    else (
                        "No supported tokenizer artifact "
                        "was found in the base model directory."
                    )
                ),
            )
        )

        return checks

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    def _dependency_checks(
        self,
    ) -> tuple[
        list[
            TrainingPreflightCheck
        ],
        dict[
            str,
            str,
        ],
    ]:

        checks: list[
            TrainingPreflightCheck
        ] = []

        versions: dict[
            str,
            str,
        ] = {}

        for package_name in (
            self.required_packages
        ):

            module_name = (
                package_name
                .replace(
                    "-",
                    "_",
                )
            )

            available = (
                importlib.util.find_spec(
                    module_name
                )
                is not None
            )

            version = (
                "unknown"
            )

            if available:

                try:

                    version = (
                        metadata.version(
                            package_name
                        )
                    )

                except (
                    metadata.PackageNotFoundError
                ):

                    version = (
                        "unknown"
                    )

                versions[
                    package_name
                ] = (
                    version
                )

            checks.append(
                self._check(
                    name=(
                        f"dependency_{package_name}"
                    ),

                    passed=(
                        available
                    ),

                    detail=(
                        (
                            f"{package_name} "
                            f"{version} available."
                        )
                        if available
                        else (
                            f"{package_name} is not installed."
                        )
                    ),
                )
            )

        return (
            checks,
            versions,
        )

    # ========================================================
    # CUDA
    # ========================================================

    def _cuda_checks(
        self,
    ) -> tuple[
        list[
            TrainingPreflightCheck
        ],
        TrainingCudaInfo,
    ]:

        if not self.inspect_cuda:

            return (
                [],
                TrainingCudaInfo(),
            )

        try:

            import torch

        except Exception as exc:

            return (
                [
                    self._check(
                        name=(
                            "cuda_runtime"
                        ),

                        passed=(
                            not self.recipe.require_cuda
                        ),

                        detail=(
                            "Torch could not be imported for "
                            f"CUDA inspection: {exc}"
                        ),
                    )
                ],
                TrainingCudaInfo(),
            )

        available = (
            bool(
                torch.cuda.is_available()
            )
        )

        device_count = (
            int(
                torch.cuda.device_count()
            )
            if available
            else 0
        )

        device_name = None

        total_memory_gib = 0.0

        bfloat16_supported: (
            bool
            | None
        ) = None

        if available:

            properties = (
                torch.cuda
                .get_device_properties(
                    0
                )
            )

            device_name = (
                properties.name
            )

            total_memory_gib = (
                float(
                    properties.total_memory
                )
                / (
                    1024
                    ** 3
                )
            )

            is_bf16_supported = getattr(
                torch.cuda,
                "is_bf16_supported",
                None,
            )

            if callable(
                is_bf16_supported
            ):

                bfloat16_supported = (
                    bool(
                        is_bf16_supported()
                    )
                )

        cuda_info = (
            TrainingCudaInfo(
                available=(
                    available
                ),

                device_count=(
                    device_count
                ),

                device_name=(
                    device_name
                ),

                total_memory_gib=(
                    total_memory_gib
                ),

                bfloat16_supported=(
                    bfloat16_supported
                ),
            )
        )

        cuda_passed = (
            available
            or not self.recipe.require_cuda
        )

        memory_passed = (
            not self.recipe.require_cuda
            or (
                available
                and total_memory_gib
                >= self.recipe.minimum_cuda_memory_gib
            )
        )

        checks = [
            self._check(
                name=(
                    "cuda_available"
                ),

                passed=(
                    cuda_passed
                ),

                detail=(
                    (
                        f"CUDA available with "
                        f"{device_count} device(s)."
                    )
                    if available
                    else (
                        "CUDA is not available."
                    )
                ),
            ),

            self._check(
                name=(
                    "cuda_memory"
                ),

                passed=(
                    memory_passed
                ),

                detail=(
                    (
                        f"Primary device has "
                        f"{total_memory_gib:.2f} GiB VRAM; "
                        f"minimum is "
                        f"{self.recipe.minimum_cuda_memory_gib:.2f} GiB."
                    )
                    if available
                    else (
                        "CUDA VRAM could not be inspected."
                    )
                ),
            ),
        ]

        if (
            self.recipe.compute_dtype
            == "bfloat16"
            and self.recipe.require_bfloat16
        ):

            bf16_passed = (
                available
                and bfloat16_supported
                is True
            )

            checks.append(
                self._check(
                    name=(
                        "cuda_bfloat16"
                    ),

                    passed=(
                        bf16_passed
                    ),

                    detail=(
                        "CUDA device reports BF16 support."
                        if bf16_passed
                        else (
                            "BF16 support is required but "
                            "was not confirmed."
                        )
                    ),
                )
            )

        return (
            checks,
            cuda_info,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
    ) -> TrainingPreflightReport:

        checks: list[
            TrainingPreflightCheck
        ] = []

        split_manifest: (
            TrainingSplitManifest
            | None
        ) = None

        train_records: list[
            PreferenceDatasetRecord
        ] = []

        validation_records: list[
            PreferenceDatasetRecord
        ] = []

        try:

            (
                split_manifest,
                train_records,
                validation_records,
                split_checks,
            ) = (
                self._verify_split()
            )

            checks.extend(
                split_checks
            )

        except Exception as exc:

            checks.append(
                self._check(
                    name=(
                        "training_split"
                    ),

                    passed=False,

                    detail=(
                        str(
                            exc
                        )
                    ),
                )
            )

        checks.extend(
            self._model_checks()
        )

        (
            dependency_checks,
            package_versions,
        ) = (
            self._dependency_checks()
        )

        checks.extend(
            dependency_checks
        )

        (
            cuda_checks,
            cuda_info,
        ) = (
            self._cuda_checks()
        )

        checks.extend(
            cuda_checks
        )

        failed_checks = [
            check.name

            for check
            in checks

            if not check.passed
        ]

        ready = (
            split_manifest
            is not None
            and not failed_checks
        )

        return (
            TrainingPreflightReport(
                generated_at=(
                    _utc_now()
                ),

                split_id=(
                    split_manifest.split_id
                    if split_manifest
                    is not None
                    else None
                ),

                source_dataset_version=(
                    split_manifest
                    .source_dataset_version
                    if split_manifest
                    is not None
                    else None
                ),

                train_record_count=(
                    len(
                        train_records
                    )
                ),

                validation_record_count=(
                    len(
                        validation_records
                    )
                ),

                package_versions=(
                    package_versions
                ),

                cuda=(
                    cuda_info
                ),

                checks=(
                    checks
                ),

                failed_checks=(
                    failed_checks
                ),

                ready=(
                    ready
                ),
            )
        )

    # ========================================================
    # DRY-RUN MANIFEST
    # ========================================================

    def write_dry_run_manifest(
        self,
        report: TrainingPreflightReport,
    ) -> Path:

        if not report.ready:

            raise ValueError(
                "Training preflight is not ready. "
                "Dry-run manifest will not be written."
            )

        manifest_path = (
            self.split_dir
            / "manifest.json"
        )

        split_manifest = (
            TrainingSplitManifest
            .model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        report_payload = (
            report.model_dump(
                mode="json",
                by_alias=True,
            )
        )

        preflight_sha256 = (
            _sha256_bytes(
                _canonical_json(
                    report_payload
                )
                .encode(
                    "utf-8"
                )
            )
        )

        dry_run = (
            TrainingDryRunManifest(
                created_at=(
                    _utc_now()
                ),

                split_id=(
                    split_manifest.split_id
                ),

                source_dataset_id=(
                    split_manifest
                    .source_dataset_id
                ),

                source_dataset_version=(
                    split_manifest
                    .source_dataset_version
                ),

                source_content_sha256=(
                    split_manifest
                    .source_content_sha256
                ),

                train_sha256=(
                    split_manifest
                    .train_sha256
                ),

                validation_sha256=(
                    split_manifest
                    .validation_sha256
                ),

                train_record_count=(
                    split_manifest
                    .train_record_count
                ),

                validation_record_count=(
                    split_manifest
                    .validation_record_count
                ),

                recipe=(
                    self.recipe
                ),

                package_versions=(
                    report.package_versions
                ),

                cuda=(
                    report.cuda
                ),

                preflight_sha256=(
                    preflight_sha256
                ),
            )
        )

        target_directory = (
            self.recipe
            .output_root
            / "dry-runs"
            / split_manifest.split_id
        )

        target_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = (
            target_directory
            / "training-manifest.json"
        )

        if target.exists():

            raise ValueError(
                "Dry-run training manifest already exists: "
                f"{target}"
            )

        target.write_text(
            json.dumps(
                dry_run.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return (
            target
        )