from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from jira_manifest import (
    EXPECTED_HEAD,
    FINAL_HASHES,
    LEGACY_GENERATED_HASHES,
    ORIGINAL_HASHES,
    RETIRED_HASHES,
)

ROOT = Path.cwd()
BUNDLE = Path(__file__).resolve().parent
PAYLOAD = BUNDLE / "jira_payload"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> tuple[int, str, str]:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def fail(message: str) -> None:
    raise SystemExit(message)


def validate_root() -> None:
    if not (ROOT / ".git").exists() or not (ROOT / "subagents").is_dir():
        fail("Run this from /mnt/c/project/agenticaiPersonal/itsm-agent")
    rc, out, _ = git("rev-parse", "HEAD")
    if rc != 0:
        fail("Unable to read git HEAD.")
    if out != EXPECTED_HEAD:
        # Permit rerun only when every production payload target already matches.
        if not all(sha256(ROOT / rel) == expected for rel, expected in FINAL_HASHES.items()):
            fail(f"Expected repository HEAD {EXPECTED_HEAD}, found {out}. Refusing to overwrite a different revision.")


def validate_overwrites() -> None:
    problems: list[str] = []
    for rel, final_hash in FINAL_HASHES.items():
        target = ROOT / rel
        actual = sha256(target)
        if actual is None:
            continue
        if actual == final_hash:
            continue
        if rel in ORIGINAL_HASHES and actual == ORIGINAL_HASHES[rel]:
            continue
        if rel in LEGACY_GENERATED_HASHES and actual == LEGACY_GENERATED_HASHES[rel]:
            continue
        problems.append(rel)

    for rel, expected in RETIRED_HASHES.items():
        target = ROOT / rel
        if not target.is_file():
            continue
        if sha256(target) != expected:
            problems.append(rel)

    if problems:
        print("Refusing to overwrite/delete files that changed after the supplied repository snapshot:")
        for rel in sorted(set(problems)):
            print(f"  {rel}")
        fail("Preserve those edits first, then rerun the Jira lifecycle installer.")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def clean_legacy_artifacts() -> None:
    # Old extracted milestone bundles and archives are installation debris, not source.
    for pattern in (
        "agenticAI_J3_complete_jira_lifecycle*",
        "agenticAI_J3C_governed_jira_comment_patch*",
        ".j3_upgrade_backup",
    ):
        for path in ROOT.glob(pattern):
            remove_path(path)

    # Retire milestone-named source/tests after validating their exact snapshot hashes.
    for rel in RETIRED_HASHES:
        remove_path(ROOT / rel)

    # Disposable milestone approval stores are intentionally not migrated.
    runtime = ROOT / ".runtime"
    if runtime.is_dir():
        for path in runtime.iterdir():
            name = path.name.lower()
            if path.is_file() and (name.startswith("j2") or name.startswith("j3")) and ".sqlite3" in name:
                path.unlink()

    # Generated caches are not repository source and keep stale milestone filenames alive.
    for cache in list(ROOT.rglob("__pycache__")):
        if ".git" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)
    for pyc in list(ROOT.rglob("*.pyc")):
        if ".git" not in pyc.parts:
            pyc.unlink(missing_ok=True)


def install_payload() -> None:
    for rel, expected_hash in FINAL_HASHES.items():
        src = PAYLOAD / rel
        if sha256(src) != expected_hash:
            fail(f"Bundle integrity failure: {rel}")
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def verify_names() -> None:
    bad: list[str] = []
    roots = [
        ROOT / "scripts",
        ROOT / "services" / "jira",
        ROOT / "services" / "ticketing",
        ROOT / "tools" / "jira",
        ROOT / "tools" / "ticketing",
        ROOT / "tests" / "unit" / "services" / "jira",
        ROOT / "tests" / "unit" / "tools" / "jira",
        ROOT / "tests" / "unit" / "tools" / "ticketing",
        ROOT / "tests" / "unit" / "subagents" / "llm" / "runtime",
    ]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower.startswith(("j2", "j3", "test_j2", "test_j3")):
                bad.append(str(path.relative_to(ROOT)))
    if bad:
        print("Unexpected milestone-named Jira files remain:")
        for rel in bad:
            print(f"  {rel}")
        fail("Jira cleanup did not converge.")


def main() -> None:
    validate_root()
    validate_overwrites()
    clean_legacy_artifacts()
    install_payload()
    verify_names()
    print("Jira lifecycle patch installed and legacy milestone artifacts cleaned.")
    print("Run:")
    print("  bash /mnt/c/project/agenticaiPersonal/itsm-agent/jira_lifecycle_patch/jira_verify.sh")


if __name__ == "__main__":
    main()
