from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys

from pathlib import Path


# ============================================================
# MIGRATION IDENTITY
# ============================================================


EXPECTED_HEAD = (
    "8799a69ae2e7ffa89bddbb36b5cdda07f779c839"
)


SCRIPT_RELATIVE_PATH = (
    Path(
        "scripts/"
        "refactor_subagents_layout.py"
    )
)


# ============================================================
# PRODUCTION SOURCE MOVES
# ============================================================


SOURCE_MOVES: dict[
    str,
    str,
] = {
    # ========================================================
    # CORE — DEFINITIONS
    # ========================================================

    "subagents/core/types.py":
        "subagents/core/definitions/types.py",

    "subagents/core/loader.py":
        "subagents/core/definitions/loader.py",

    "subagents/core/registry.py":
        "subagents/core/definitions/registry.py",

    # ========================================================
    # CORE — ORCHESTRATION
    # ========================================================

    "subagents/core/llm_router.py":
        "subagents/core/orchestration/router.py",

    "subagents/core/orchestrator.py":
        "subagents/core/orchestration/orchestrator.py",

    "subagents/core/primary_assistant.py":
        "subagents/core/orchestration/primary_assistant.py",

    "subagents/core/runtime.py":
        "subagents/core/orchestration/runtime.py",

    # ========================================================
    # CORE — TOOLING
    # ========================================================

    "subagents/core/capabilities.py":
        "subagents/core/tooling/capabilities.py",

    "subagents/core/tool_gateway.py":
        "subagents/core/tooling/gateway.py",

    "subagents/core/tool_parser.py":
        "subagents/core/tooling/parser.py",

    "subagents/core/tool_prompt.py":
        "subagents/core/tooling/prompt.py",

    # ========================================================
    # LLM — RUNTIME
    # ========================================================

    "subagents/llm/base.py":
        "subagents/llm/runtime/base.py",

    "subagents/llm/factory.py":
        "subagents/llm/runtime/factory.py",

    "subagents/llm/inference.py":
        "subagents/llm/runtime/inference.py",

    "subagents/llm/model_manager.py":
        "subagents/llm/runtime/model_manager.py",

    "subagents/llm/observability.py":
        "subagents/llm/runtime/observability.py",

    "subagents/llm/registry.py":
        "subagents/llm/runtime/registry.py",

    "subagents/llm/scheduler.py":
        "subagents/llm/runtime/scheduler.py",

    # ========================================================
    # LLM — BACKENDS
    # ========================================================

    "subagents/llm/ministral_hub.py":
        "subagents/llm/backends/ministral_hub.py",

    "subagents/llm/qwen3_worker.py":
        "subagents/llm/backends/qwen3_worker.py",

    "subagents/llm/qwen_coder_worker.py":
        "subagents/llm/backends/qwen_coder_worker.py",

    "subagents/llm/qwen_funcall.py":
        "subagents/llm/backends/qwen_funcall.py",
}


# ============================================================
# TEST MOVES
# ============================================================


TEST_MOVES: dict[
    str,
    str,
] = {
    # ========================================================
    # CORE — DEFINITIONS
    # ========================================================

    "subagents/test/test_loader.py":
        (
            "tests/unit/subagents/core/definitions/"
            "test_loader.py"
        ),

    "subagents/test/test_registry.py":
        (
            "tests/unit/subagents/core/definitions/"
            "test_registry.py"
        ),

    # ========================================================
    # CORE — ORCHESTRATION
    # ========================================================

    "subagents/test/test_llm_router.py":
        (
            "tests/unit/subagents/core/orchestration/"
            "test_router.py"
        ),

    "subagents/test/test_orchestrator.py":
        (
            "tests/unit/subagents/core/orchestration/"
            "test_orchestrator.py"
        ),

    "subagents/test/test_runtime.py":
        (
            "tests/unit/subagents/core/orchestration/"
            "test_runtime.py"
        ),

    "subagents/test/test_task_context_capture.py":
        (
            "tests/unit/subagents/core/orchestration/"
            "test_task_context_capture.py"
        ),

    "subagents/test/test_structured_tool_results.py":
        (
            "tests/unit/subagents/core/orchestration/"
            "test_structured_tool_results.py"
        ),

    # ========================================================
    # CORE — TOOLING
    # ========================================================

    "subagents/test/test_capability_catalog.py":
        (
            "tests/unit/subagents/core/tooling/"
            "test_capability_catalog.py"
        ),

    "subagents/test/test_capability_driven_agents.py":
        (
            "tests/unit/subagents/core/tooling/"
            "test_capability_driven_agents.py"
        ),

    "subagents/test/test_tool_gateway.py":
        (
            "tests/unit/subagents/core/tooling/"
            "test_gateway.py"
        ),

    "subagents/test/test_tool_parser.py":
        (
            "tests/unit/subagents/core/tooling/"
            "test_parser.py"
        ),

    # ========================================================
    # LLM — RUNTIME
    # ========================================================

    "subagents/test/test_gpu_scheduler.py":
        (
            "tests/unit/subagents/llm/runtime/"
            "test_scheduler.py"
        ),

    "subagents/test/test_inference_observability.py":
        (
            "tests/unit/subagents/llm/runtime/"
            "test_inference_observability.py"
        ),

    "subagents/test/test_model_registry.py":
        (
            "tests/unit/subagents/llm/runtime/"
            "test_model_registry.py"
        ),

    # ========================================================
    # LLM — BACKENDS
    # ========================================================

    "subagents/test/test_ministral_hub.py":
        (
            "tests/unit/subagents/llm/backends/"
            "test_ministral_hub.py"
        ),
}


# ============================================================
# PACKAGE INITIALIZERS
# ============================================================


PACKAGE_FILES = (
    "subagents/core/definitions/__init__.py",
    "subagents/core/orchestration/__init__.py",
    "subagents/core/tooling/__init__.py",

    "subagents/llm/runtime/__init__.py",
    "subagents/llm/backends/__init__.py",

    "tests/unit/subagents/__init__.py",
    "tests/unit/subagents/core/__init__.py",
    "tests/unit/subagents/core/definitions/__init__.py",
    "tests/unit/subagents/core/orchestration/__init__.py",
    "tests/unit/subagents/core/tooling/__init__.py",

    "tests/unit/subagents/llm/__init__.py",
    "tests/unit/subagents/llm/runtime/__init__.py",
    "tests/unit/subagents/llm/backends/__init__.py",
)


# ============================================================
# MODULE HELPERS
# ============================================================


def module_from_python_path(
    path: str,
) -> str:

    candidate = (
        Path(
            path
        )
    )

    if (
        candidate.suffix
        != ".py"
    ):

        raise ValueError(
            "Python module path must end in .py: "
            f"{path}"
        )

    if (
        candidate.name
        == "__init__.py"
    ):

        return (
            ".".join(
                candidate.parent.parts
            )
        )

    return (
        ".".join(
            candidate
            .with_suffix("")
            .parts
        )
    )


SOURCE_MODULE_MOVES: dict[
    str,
    str,
] = {
    module_from_python_path(
        source
    ):
        module_from_python_path(
            destination
        )

    for (
        source,
        destination,
    )
    in SOURCE_MOVES.items()
}


TEST_MODULE_MOVES: dict[
    str,
    str,
] = {
    module_from_python_path(
        source
    ):
        module_from_python_path(
            destination
        )

    for (
        source,
        destination,
    )
    in TEST_MOVES.items()
}


MODULE_MOVES: dict[
    str,
    str,
] = {
    **SOURCE_MODULE_MOVES,
    **TEST_MODULE_MOVES,
}


ALL_FILE_MOVES: dict[
    str,
    str,
] = {
    **SOURCE_MOVES,
    **TEST_MOVES,
}


# ============================================================
# SUBPROCESS HELPERS
# ============================================================


def run(
    *args: str,
    root: (
        Path
        | None
    ) = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:

    return subprocess.run(
        list(
            args
        ),
        cwd=(
            root
        ),
        check=(
            check
        ),
        text=True,
        capture_output=(
            capture
        ),
    )


def repository_root(
) -> Path:

    result = (
        run(
            "git",
            "rev-parse",
            "--show-toplevel",
        )
    )

    return (
        Path(
            result.stdout.strip()
        )
        .resolve()
    )


def current_head(
    root: Path,
) -> str:

    return (
        run(
            "git",
            "rev-parse",
            "HEAD",
            root=(
                root
            ),
        )
        .stdout
        .strip()
    )


def read_head_file(
    *,
    root: Path,
    path: str,
) -> str:

    return (
        run(
            "git",
            "show",
            f"HEAD:{path}",
            root=(
                root
            ),
        )
        .stdout
    )


# ============================================================
# SAFETY
# ============================================================


def assert_expected_head(
    root: Path,
) -> None:

    observed = (
        current_head(
            root
        )
    )

    if (
        observed
        != EXPECTED_HEAD
    ):

        raise SystemExit(
            "REFUSING: unexpected repository HEAD.\n\n"
            f"Expected: {EXPECTED_HEAD}\n"
            f"Observed: {observed}\n"
        )


def assert_clean_tree_except_script(
    root: Path,
) -> None:

    result = (
        run(
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            root=(
                root
            ),
        )
    )

    allowed = {
        (
            "?? "
            + str(
                SCRIPT_RELATIVE_PATH
            )
        ),
    }

    unexpected = [
        line

        for line
        in result.stdout.splitlines()

        if (
            line.strip()
            and line
            not in allowed
        )
    ]

    if unexpected:

        print(
            "REFUSING: working tree is not clean."
        )

        print()

        for line in unexpected:

            print(
                f"  {line}"
            )

        print()

        print(
            "Commit/stash unrelated changes first."
        )

        raise SystemExit(
            2
        )


def assert_move_layout(
    root: Path,
) -> None:

    failures: list[
        str
    ] = []

    for (
        source,
        destination,
    ) in (
        ALL_FILE_MOVES.items()
    ):

        source_path = (
            root
            / source
        )

        destination_path = (
            root
            / destination
        )

        if not (
            source_path
            .is_file()
        ):

            failures.append(
                (
                    "missing source: "
                    f"{source}"
                )
            )

        if (
            destination_path
            .exists()
        ):

            failures.append(
                (
                    "destination already exists: "
                    f"{destination}"
                )
            )

    if failures:

        print(
            "REFUSING: migration layout preflight failed."
        )

        for failure in failures:

            print(
                f"  {failure}"
            )

        raise SystemExit(
            2
        )


# ============================================================
# RELATIVE IMPORT REWRITE
# ============================================================


RELATIVE_FROM_PATTERN = (
    re.compile(
        (
            r"^(?P<indent>\s*)"
            r"from\s+"
            r"(?P<dots>\.+)"
            r"(?P<module>[A-Za-z_][A-Za-z0-9_\.]*)"
            r"\s+import\b"
        ),
        flags=(
            re.MULTILINE
        ),
    )
)


PACKAGE_RELATIVE_PATTERN = (
    re.compile(
        (
            r"^\s*"
            r"from\s+\.+\s+import\b"
        ),
        flags=(
            re.MULTILINE
        ),
    )
)


def resolve_relative_module(
    *,
    old_module: str,
    dots: str,
    relative_module: str,
) -> str:

    package_parts = (
        old_module
        .split(".")[
            :-1
        ]
    )

    level = (
        len(
            dots
        )
    )

    parent_steps = (
        level
        - 1
    )

    if (
        parent_steps
        > len(
            package_parts
        )
    ):

        raise ValueError(
            "Relative import escapes package: "
            f"{old_module} "
            f"{dots}{relative_module}"
        )

    if parent_steps:

        base_parts = (
            package_parts[
                :-parent_steps
            ]
        )

    else:

        base_parts = (
            package_parts
        )

    old_target = (
        ".".join(
            [
                *base_parts,
                *relative_module.split("."),
            ]
        )
    )

    return (
        MODULE_MOVES.get(
            old_target,
            old_target,
        )
    )


def rewrite_relative_imports(
    *,
    text: str,
    old_module: str,
) -> str:

    if (
        PACKAGE_RELATIVE_PATTERN
        .search(
            text
        )
    ):

        raise ValueError(
            "Unsupported relative package import "
            f"in moved module {old_module}: "
            "'from . import name'."
        )

    def replace(
        match: re.Match,
    ) -> str:

        resolved = (
            resolve_relative_module(
                old_module=(
                    old_module
                ),

                dots=(
                    match.group(
                        "dots"
                    )
                ),

                relative_module=(
                    match.group(
                        "module"
                    )
                ),
            )
        )

        return (
            f"{match.group('indent')}"
            f"from {resolved} import"
        )

    return (
        RELATIVE_FROM_PATTERN
        .sub(
            replace,
            text,
        )
    )


# ============================================================
# __FILE__ DEPTH PRESERVATION
# ============================================================


PARENTS_PATTERN = (
    re.compile(
        (
            r"("
            r"Path\(__file__\)"
            r"(?:\.resolve\(\))?"
            r"\.parents\["
            r")"
            r"(?P<depth>\d+)"
            r"(\])"
        )
    )
)


def directory_depth(
    path: str,
) -> int:

    return (
        len(
            Path(
                path
            )
            .parent
            .parts
        )
    )


def rewrite_file_parent_depth(
    *,
    text: str,
    source: str,
    destination: str,
) -> str:

    delta = (
        directory_depth(
            destination
        )
        - directory_depth(
            source
        )
    )

    if (
        delta
        == 0
    ):

        return (
            text
        )

    def replace(
        match: re.Match,
    ) -> str:

        old_value = (
            int(
                match.group(
                    "depth"
                )
            )
        )

        new_value = (
            old_value
            + delta
        )

        if (
            new_value
            < 0
        ):

            raise ValueError(
                "Invalid Path(__file__).parents "
                "depth after migration."
            )

        return (
            f"{match.group(1)}"
            f"{new_value}"
            f"{match.group(3)}"
        )

    return (
        PARENTS_PATTERN
        .sub(
            replace,
            text,
        )
    )


# ============================================================
# EXACT MODULE REWRITE
# ============================================================


def rewrite_module_references(
    text: str,
) -> str:

    for (
        old_module,
        new_module,
    ) in sorted(
        MODULE_MOVES.items(),
        key=lambda item: (
            len(
                item[
                    0
                ]
            )
        ),
        reverse=True,
    ):

        pattern = (
            re.compile(
                (
                    r"(?<![A-Za-z0-9_\.])"
                    + re.escape(
                        old_module
                    )
                    + r"(?![A-Za-z0-9_\.])"
                )
            )
        )

        text = (
            pattern.sub(
                new_module,
                text,
            )
        )

    return (
        text
    )


# ============================================================
# FILE PATH REWRITE
# ============================================================


def rewrite_path_references(
    text: str,
) -> str:

    for (
        old_path,
        new_path,
    ) in sorted(
        ALL_FILE_MOVES.items(),
        key=lambda item: (
            len(
                item[
                    0
                ]
            )
        ),
        reverse=True,
    ):

        text = (
            text.replace(
                old_path,
                new_path,
            )
        )

    return (
        text
    )


def rewrite_all_references(
    text: str,
) -> str:

    text = (
        rewrite_module_references(
            text
        )
    )

    text = (
        rewrite_path_references(
            text
        )
    )

    return (
        text
    )


# ============================================================
# GIT MOVE
# ============================================================


def git_move(
    *,
    root: Path,
    source: str,
    destination: str,
) -> None:

    destination_path = (
        root
        / destination
    )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        "git",
        "mv",
        source,
        destination,
        root=(
            root
        ),
    )


def apply_moves(
    root: Path,
) -> None:

    for (
        source,
        destination,
    ) in (
        SOURCE_MOVES.items()
    ):

        git_move(
            root=(
                root
            ),

            source=(
                source
            ),

            destination=(
                destination
            ),
        )

    for (
        source,
        destination,
    ) in (
        TEST_MOVES.items()
    ):

        git_move(
            root=(
                root
            ),

            source=(
                source
            ),

            destination=(
                destination
            ),
        )


# ============================================================
# PACKAGE FILES
# ============================================================


def create_package_files(
    root: Path,
) -> None:

    for relative in (
        PACKAGE_FILES
    ):

        path = (
            root
            / relative
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not (
            path.exists()
        ):

            path.write_text(
                "",
                encoding="utf-8",
            )


# ============================================================
# REBUILD MOVED FILES FROM HEAD
# ============================================================


def rebuild_moved_file(
    *,
    root: Path,
    source: str,
    destination: str,
) -> None:

    original = (
        read_head_file(
            root=(
                root
            ),

            path=(
                source
            ),
        )
    )

    old_module = (
        module_from_python_path(
            source
        )
    )

    rewritten = (
        rewrite_relative_imports(
            text=(
                original
            ),

            old_module=(
                old_module
            ),
        )
    )

    rewritten = (
        rewrite_file_parent_depth(
            text=(
                rewritten
            ),

            source=(
                source
            ),

            destination=(
                destination
            ),
        )
    )

    rewritten = (
        rewrite_all_references(
            rewritten
        )
    )

    destination_path = (
        root
        / destination
    )

    destination_path.write_text(
        rewritten,
        encoding="utf-8",
    )


def rebuild_all_moved_files(
    root: Path,
) -> None:

    for (
        source,
        destination,
    ) in (
        ALL_FILE_MOVES.items()
    ):

        rebuild_moved_file(
            root=(
                root
            ),

            source=(
                source
            ),

            destination=(
                destination
            ),
        )


# ============================================================
# REPOSITORY TEXT DISCOVERY
# ============================================================


EXCLUDED_PARTS = {
    ".git",
    ".runtime",
    "__pycache__",
}


def repository_text_files(
    root: Path,
) -> list[
    Path
]:

    result: list[
        Path
    ] = []

    for path in (
        root.rglob(
            "*"
        )
    ):

        if not (
            path.is_file()
        ):

            continue

        relative = (
            path.relative_to(
                root
            )
        )

        if any(
            part
            in EXCLUDED_PARTS

            for part
            in relative.parts
        ):

            continue

        if (
            relative
            == SCRIPT_RELATIVE_PATH
        ):

            continue

        try:

            payload = (
                path.read_bytes()
            )

        except OSError:

            continue

        if (
            b"\x00"
            in payload
        ):

            continue

        try:

            payload.decode(
                "utf-8"
            )

        except UnicodeDecodeError:

            continue

        result.append(
            path
        )

    return (
        sorted(
            result
        )
    )


# ============================================================
# REPOSITORY-WIDE REWRITE
# ============================================================


def rewrite_repository(
    root: Path,
) -> int:

    changed = (
        0
    )

    for path in (
        repository_text_files(
            root
        )
    ):

        original = (
            path.read_text(
                encoding="utf-8"
            )
        )

        rewritten = (
            rewrite_all_references(
                original
            )
        )

        if (
            rewritten
            == original
        ):

            continue

        path.write_text(
            rewritten,
            encoding="utf-8",
        )

        changed += (
            1
        )

    return (
        changed
    )


# ============================================================
# STALE IMPORT AUDIT
# ============================================================


def find_stale_python_imports(
    root: Path,
) -> list[
    str
]:

    old_modules = (
        set(
            MODULE_MOVES
        )
    )

    failures: list[
        str
    ] = []

    for path in (
        root.rglob(
            "*.py"
        )
    ):

        relative = (
            path.relative_to(
                root
            )
        )

        if any(
            part
            in EXCLUDED_PARTS

            for part
            in relative.parts
        ):

            continue

        if (
            relative
            == SCRIPT_RELATIVE_PATH
        ):

            continue

        try:

            source = (
                path.read_text(
                    encoding="utf-8"
                )
            )

            tree = (
                ast.parse(
                    source,
                    filename=(
                        str(
                            relative
                        )
                    ),
                )
            )

        except Exception as exc:

            failures.append(
                (
                    f"{relative}: parse failure: "
                    f"{exc}"
                )
            )

            continue

        for node in (
            ast.walk(
                tree
            )
        ):

            if isinstance(
                node,
                ast.Import,
            ):

                for alias in (
                    node.names
                ):

                    if (
                        alias.name
                        in old_modules
                    ):

                        failures.append(
                            (
                                f"{relative}:"
                                f"{node.lineno}: "
                                f"stale import "
                                f"{alias.name}"
                            )
                        )

                continue

            if not isinstance(
                node,
                ast.ImportFrom,
            ):

                continue

            # Other packages may validly use relative imports.
            # Only absolute imports are relevant to this
            # repository-wide module move audit.
            if (
                node.level
                != 0
            ):

                continue

            if (
                node.module
                in old_modules
            ):

                failures.append(
                    (
                        f"{relative}:"
                        f"{node.lineno}: "
                        "stale from-import "
                        f"{node.module}"
                    )
                )

    return (
        failures
    )


# ============================================================
# STALE RAW MODULE/PATH AUDIT
# ============================================================


def find_stale_references(
    root: Path,
) -> list[
    str
]:

    failures: list[
        str
    ] = []

    script_path = (
        root
        / SCRIPT_RELATIVE_PATH
    ).resolve()

    for path in (
        repository_text_files(
            root
        )
    ):

        if (
            path.resolve()
            == script_path
        ):

            continue

        text = (
            path.read_text(
                encoding="utf-8"
            )
        )

        relative = (
            path.relative_to(
                root
            )
        )

        for old_path in (
            ALL_FILE_MOVES
        ):

            if (
                old_path
                in text
            ):

                failures.append(
                    (
                        f"{relative}: stale path "
                        f"{old_path}"
                    )
                )

        for old_module in (
            MODULE_MOVES
        ):

            pattern = (
                re.compile(
                    (
                        r"(?<![A-Za-z0-9_\.])"
                        + re.escape(
                            old_module
                        )
                        + r"(?![A-Za-z0-9_\.])"
                    )
                )
            )

            if (
                pattern.search(
                    text
                )
            ):

                failures.append(
                    (
                        f"{relative}: stale module "
                        f"{old_module}"
                    )
                )

    return (
        failures
    )


# ============================================================
# COMPILE CHECK
# ============================================================


def compile_repository(
    root: Path,
) -> None:

    result = (
        subprocess.run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "subagents",
                "learning",
                "tests",
                "api",
                "agent",
                "services",
                "tools",
                "config",
                "scripts",
            ],
            cwd=(
                root
            ),
        )
    )

    if (
        result.returncode
        != 0
    ):

        raise SystemExit(
            "Compile check failed."
        )


# ============================================================
# IMPORT SMOKE
# ============================================================


def import_smoke(
    root: Path,
) -> None:

    modules = (
        sorted(
            SOURCE_MODULE_MOVES.values()
        )
    )

    code = (
        "import importlib\n"
        f"modules = {modules!r}\n"
        "for name in modules:\n"
        "    importlib.import_module(name)\n"
        "print("
        "'Imported', "
        "len(modules), "
        "'moved production modules.'"
        ")\n"
    )

    result = (
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
            ],
            cwd=(
                root
            ),
        )
    )

    if (
        result.returncode
        != 0
    ):

        raise SystemExit(
            "Moved-module import smoke failed."
        )


# ============================================================
# DRY RUN
# ============================================================


def dry_run(
    root: Path,
) -> None:

    assert_expected_head(
        root
    )

    assert_clean_tree_except_script(
        root
    )

    assert_move_layout(
        root
    )

    print(
        "Subagents Layout Refactor — DRY RUN"
    )

    print(
        "==================================="
    )

    print()

    print(
        f"HEAD: {current_head(root)}"
    )

    print()

    print(
        "Production moves:"
    )

    for (
        source,
        destination,
    ) in (
        SOURCE_MOVES.items()
    ):

        print(
            f"  {source}"
        )

        print(
            f"    -> {destination}"
        )

    print()

    print(
        "Test moves:"
    )

    for (
        source,
        destination,
    ) in (
        TEST_MOVES.items()
    ):

        print(
            f"  {source}"
        )

        print(
            f"    -> {destination}"
        )

    print()

    print(
        "Module mappings:"
    )

    for (
        source,
        destination,
    ) in sorted(
        MODULE_MOVES.items()
    ):

        print(
            f"  {source}"
        )

        print(
            f"    -> {destination}"
        )

    print()

    print(
        "DRY RUN: PASS"
    )

    print()

    print(
        "No files were modified."
    )


# ============================================================
# APPLY
# ============================================================


def apply_refactor(
    root: Path,
) -> None:

    assert_expected_head(
        root
    )

    assert_clean_tree_except_script(
        root
    )

    assert_move_layout(
        root
    )

    print(
        "Subagents Layout Refactor — APPLY"
    )

    print(
        "================================="
    )

    print()

    print(
        "[1/8] Moving production and test modules..."
    )

    apply_moves(
        root
    )

    print(
        "[2/8] Creating package initializers..."
    )

    create_package_files(
        root
    )

    print(
        "[3/8] Rebuilding moved files from HEAD..."
    )

    rebuild_all_moved_files(
        root
    )

    print(
        "[4/8] Rewriting repository-wide references..."
    )

    changed = (
        rewrite_repository(
            root
        )
    )

    print(
        f"      rewritten files: {changed}"
    )

    print(
        "[5/8] Auditing stale Python imports..."
    )

    stale_imports = (
        find_stale_python_imports(
            root
        )
    )

    if stale_imports:

        print()

        print(
            "STALE PYTHON IMPORTS:"
        )

        for failure in (
            stale_imports
        ):

            print(
                f"  {failure}"
            )

        raise SystemExit(
            3
        )

    print(
        "[6/8] Auditing stale paths/module names..."
    )

    stale_references = (
        find_stale_references(
            root
        )
    )

    if stale_references:

        print()

        print(
            "STALE REFERENCES:"
        )

        for failure in (
            stale_references
        ):

            print(
                f"  {failure}"
            )

        raise SystemExit(
            3
        )

    print(
        "[7/8] Compiling repository..."
    )

    compile_repository(
        root
    )

    print(
        "[8/8] Importing moved production modules..."
    )

    import_smoke(
        root
    )

    print()

    print(
        "SUBAGENTS LAYOUT REFACTOR: PASS"
    )

    print()

    subprocess.run(
        [
            "git",
            "diff",
            "HEAD",
            "--stat",
        ],
        cwd=(
            root
        ),
    )

    print()

    subprocess.run(
        [
            "git",
            "status",
            "--short",
        ],
        cwd=(
            root
        ),
    )


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Behavior-preserving organization of "
                "subagents/core, subagents/llm, and their "
                "unit tests."
            )
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the refactor. "
            "Without this flag only preflight is run."
        ),
    )

    return (
        parser
    )


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    root = (
        repository_root()
    )

    if args.apply:

        apply_refactor(
            root
        )

    else:

        dry_run(
            root
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )