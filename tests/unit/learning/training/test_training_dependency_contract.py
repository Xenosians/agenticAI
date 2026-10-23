from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from learning.training.dpo_qlora import EXPECTED_TRAINING_VERSIONS


def test_declared_dependencies_admit_the_validated_training_stack():
    root = Path(__file__).resolve().parents[4]
    for filename in ("requirements.txt", "requirements-training.txt"):
        for line in (root / filename).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r ")):
                continue
            requirement = Requirement(line)
            locked = EXPECTED_TRAINING_VERSIONS.get(canonicalize_name(requirement.name))
            if locked is not None:
                assert locked in requirement.specifier, (
                    f"{filename}: {requirement} excludes validated {locked}"
                )


def test_training_datasets_dependency_uses_the_validated_version():
    root = Path(__file__).resolve().parents[4]
    requirements = [Requirement(line) for line in
                    (root / "requirements-training.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith(("#", "-r "))]
    dataset = next(item for item in requirements if canonicalize_name(item.name) == "datasets")
    assert str(dataset.specifier) == "==" + EXPECTED_TRAINING_VERSIONS["datasets"]
