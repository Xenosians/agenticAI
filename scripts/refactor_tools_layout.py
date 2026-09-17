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
    "0f306efa5d04c300c552bb36c073c9fe9776541c"
)


SCRIPT_RELATIVE_PATH = (
    Path(
        "scripts/"
        "refactor_tools_layout.py"
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
    # DEVELOPER
    # ========================================================

    "tools/developer_catalog.py":
        "tools/developer/catalog.py",

    "tools/developer_execution.py":
        "tools/developer/execution.py",

    "tools/developer_mcp.py":
        "tools/developer/mcp.py",

    "tools/developer_presentation.py":
        "tools/developer/presentation.py",

    "tools/developer_runtime.py":
        "tools/developer/runtime.py",

    "tools/developer_runtime_presentation.py":
        "tools/developer/runtime_presentation.py",

    # ========================================================
    # GIT
    #
    # Moving git.py to git/__init__.py intentionally preserves:
    #
    #     import tools.git
    #
    # ========================================================

    "tools/git.py":
        "tools/git/__init__.py",

    "tools/git_catalog.py":
        "tools/git/catalog.py",

    "tools/git_mcp.py":
        "tools/git/mcp.py",

    "tools/git_presentation.py":
        "tools/git/presentation.py",

    # ========================================================
    # TICKETING
    # ========================================================

    "tools/ticketing_catalog.py":
        "tools/ticketing/catalog.py",

    "tools/ticketing_mcp.py":
        "tools/ticketing/mcp.py",

    "tools/ticketing_presentation.py":
        "tools/ticketing/presentation.py",

    # ========================================================
    # WORKSPACE
    #
    # workspace.py becomes the package initializer so:
    #
    #     from tools.workspace import ...
    #
    # remains valid.
    # ========================================================

    "tools/workspace.py":
        "tools/workspace/__init__.py",

    "tools/workspace_discovery.py":
        "tools/workspace/discovery.py",

    "tools/workspace_policy.py":
        "tools/workspace/policy.py",

    "tools/workspace_presentation.py":
        "tools/workspace/presentation.py",

    # ========================================================
    # PRESENTATION
    #
    # presentation.py becomes the package initializer so:
    #
    #     from tools.presentation import ...
    #
    # remains valid.
    # ========================================================

    "tools/presentation.py":
        "tools/presentation/__init__.py",

    "tools/result_cards.py":
        "tools/presentation/result_cards.py",

    "tools/result_presentation_registry.py":
        "tools/presentation/registry.py",
}


# ============================================================
# TEST MOVES
# ============================================================


TEST_MOVES: dict[
    str,
    str,
] = {
    # ========================================================
    # DEVELOPER
    # ========================================================

    "subagents/test/test_developer_execution_tools.py":
        (
            "tests/unit/tools/developer/"
            "test_execution.py"
        ),

    "subagents/test/test_developer_runtime_tools.py":
        (
            "tests/unit/tools/developer/"
            "test_runtime.py"
        ),

    # ========================================================
    # GIT
    # ========================================================

    "subagents/test/test_git_capability_metadata.py":
        (
            "tests/unit/tools/git/"
            "test_capability_metadata.py"
        ),

    "subagents/test/test_git_stack.py":
        (
            "tests/unit/tools/git/"
            "test_stack.py"
        ),

    "subagents/test/test_git_tools.py":
        (
            "tests/unit/tools/git/"
            "test_tools.py"
        ),

    # ========================================================
    # TICKETING
    # ========================================================

    "subagents/test/test_ticketing_mcp_registration.py":
        (
            "tests/unit/tools/ticketing/"
            "test_mcp_registration.py"
        ),

    "subagents/test/test_ticketing_tools.py":
        (
            "tests/unit/tools/ticketing/"
            "test_tools.py"
        ),

    "subagents/test/test_ticket_read_details.py":
        (
            "tests/unit/tools/ticketing/"
            "test_read_details.py"
        ),

    "subagents/test/test_ticket_search.py":
        (
            "tests/unit/tools/ticketing/"
            "test_search.py"
        ),

    # ========================================================
    # WORKSPACE
    # ========================================================

    "subagents/test/test_workspace_discovery_tools.py":
        (
            "tests/unit/tools/workspace/"
            "test_discovery.py"
        ),
}


# ============================================================
# PACKAGE INITIALIZERS
#
# Do NOT create:
#
#     tools/git/__init__.py
#     tools/workspace/__init__.py
#     tools/presentation/__init__.py
#
# because real modules are moved into those files.
# ============================================================


PACKAGE_FILES = (
    "tools/developer/__init__.py",
    "tools/ticketing/__init__.py",

    "tests/unit/tools/__init__.py",

    "tests/unit/tools/developer/__init__.py",
    "tests/unit/tools/git/__init__.py",
    "tests/unit/tools/ticketing/__init__.py",
    "tests/unit/tools/workspace/__init__.py",
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
                candidate
                .parent
                .parts
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


ALL_MODULE_MOVES: dict[
    str,
    str,
] = {
    **SOURCE_MODULE_MOVES,
    **TEST_MODULE_MOVES,
}


# Only entries whose module name actually changes need import
# rewriting.
MODULE_RENAMES: dict[
    str,
    str,
] = {
    source:
        destination

    for (
        source,
        destination,
    )
    in ALL_MODULE_MOVES.items()

    if (
        source
        != destination
    )
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

    return (
        subprocess.run(
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


def status_lines(
    root: Path,
) -> list[
    str
]:

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

    return [
        line

        for line
        in result.stdout.splitlines()

        if line.strip()
    ]


def assert_clean_tree_except_script(
    root: Path,
) -> None:

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
        in status_lines(
            root
        )

        if (
            line
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
            "Commit/stash unrelated work first."
        )

        raise SystemExit(
            2
        )


# ============================================================
# MOVE STATE
# ============================================================


class MoveState:
    PENDING = (
        "pending"
    )

    COMPLETE = (
        "complete"
    )

    CONFLICT = (
        "conflict"
    )

    MISSING = (
        "missing"
    )


def classify_move(
    *,
    root: Path,
    source: str,
    destination: str,
) -> str:

    source_exists = (
        (
            root
            / source
        )
        .exists()
    )

    destination_exists = (
        (
            root
            / destination
        )
        .exists()
    )

    if (
        source_exists
        and destination_exists
    ):

        return (
            MoveState.CONFLICT
        )

    if source_exists:

        return (
            MoveState.PENDING
        )

    if destination_exists:

        return (
            MoveState.COMPLETE
        )

    return (
        MoveState.MISSING
    )


def inspect_move_state(
    root: Path,
) -> dict[
    str,
    int,
]:

    counts = {
        MoveState.PENDING:
            0,

        MoveState.COMPLETE:
            0,

        MoveState.CONFLICT:
            0,

        MoveState.MISSING:
            0,
    }

    failures: list[
        str
    ] = []

    for (
        source,
        destination,
    ) in (
        ALL_FILE_MOVES.items()
    ):

        state = (
            classify_move(
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
        )

        counts[
            state
        ] += (
            1
        )

        if (
            state
            in {
                MoveState.CONFLICT,
                MoveState.MISSING,
            }
        ):

            failures.append(
                (
                    f"{state}: "
                    f"{source} -> {destination}"
                )
            )

    if failures:

        print(
            "Invalid migration filesystem state:"
        )

        for failure in failures:

            print(
                f"  {failure}"
            )

        raise SystemExit(
            2
        )

    return (
        counts
    )


# ============================================================
# RELATIVE IMPORT HANDLING
# ============================================================


RELATIVE_FROM_PATTERN = (
    re.compile(
        (
            r"^(?P<indent>\s*)"
            r"from\s+"
            r"(?P<dots>\.+)"
            r"(?P<module>"
            r"[A-Za-z_][A-Za-z0-9_\.]*"
            r")"
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

    target = (
        ".".join(
            [
                *base_parts,
                *relative_module.split("."),
            ]
        )
    )

    return (
        MODULE_RENAMES.get(
            target,
            target,
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
            "Unsupported package-style relative import "
            f"in moved file {old_module}: "
            "'from . import name'."
        )

    def replace(
        match: re.Match,
    ) -> str:

        target = (
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
            f"from {target} import"
        )

    return (
        RELATIVE_FROM_PATTERN
        .sub(
            replace,
            text,
        )
    )


# ============================================================
# __FILE__ PARENT DEPTH
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
                "Invalid Path(__file__).parents depth."
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
        MODULE_RENAMES.items(),
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
# PATH REWRITE
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


def ensure_git_move(
    *,
    root: Path,
    source: str,
    destination: str,
) -> None:

    state = (
        classify_move(
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
    )

    if (
        state
        == MoveState.COMPLETE
    ):

        return

    if (
        state
        != MoveState.PENDING
    ):

        raise RuntimeError(
            "Cannot move path in state "
            f"{state}: "
            f"{source} -> {destination}"
        )

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


def ensure_all_moves(
    root: Path,
) -> None:

    for (
        source,
        destination,
    ) in (
        SOURCE_MOVES.items()
    ):

        ensure_git_move(
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

        ensure_git_move(
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
# PACKAGE INITIALIZERS
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
# DETERMINISTIC MOVED-FILE REBUILD
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

    (
        root
        / destination
    ).write_text(
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
# TEXT FILE DISCOVERY
# ============================================================


EXCLUDED_PARTS = {
    ".git",
    ".runtime",
    "__pycache__",
    ".pytest_cache",
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

            raw = (
                path.read_bytes()
            )

        except OSError:

            continue

        if (
            b"\x00"
            in raw
        ):

            continue

        try:

            raw.decode(
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


def rewrite_repository_references(
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
# STALE PYTHON IMPORT AUDIT
# ============================================================


def find_stale_python_imports(
    root: Path,
) -> list[
    str
]:

    old_modules = (
        set(
            MODULE_RENAMES
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
                    f"{relative}: "
                    f"parse failure: {exc}"
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
                                "stale import "
                                f"{alias.name}"
                            )
                        )

                continue

            if not isinstance(
                node,
                ast.ImportFrom,
            ):

                continue

            # Valid relative imports elsewhere are not our concern.
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
# STALE RAW REFERENCE AUDIT
# ============================================================


def find_stale_references(
    root: Path,
) -> list[
    str
]:

    failures: list[
        str
    ] = []

    for path in (
        repository_text_files(
            root
        )
    ):

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

        # ----------------------------------------------------
        # OLD FILE PATHS
        # ----------------------------------------------------

        for old_path in (
            ALL_FILE_MOVES
        ):

            if (
                old_path
                in text
            ):

                failures.append(
                    (
                        f"{relative}: "
                        "stale path "
                        f"{old_path}"
                    )
                )

        # ----------------------------------------------------
        # OLD MODULE NAMES
        # ----------------------------------------------------

        for old_module in (
            MODULE_RENAMES
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
                        f"{relative}: "
                        "stale module "
                        f"{old_module}"
                    )
                )

    return (
        failures
    )


# ============================================================
# COMPILE VALIDATION
# ============================================================


def compile_repository(
    root: Path,
) -> None:

    targets = (
        "tools",
        "subagents",
        "learning",
        "tests",
        "agent",
        "api",
        "config",
        "services",
        "scripts",
    )

    result = (
        subprocess.run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                *targets,
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
            "Python compile validation failed."
        )


# ============================================================
# PRODUCTION IMPORT SMOKE
# ============================================================


def import_smoke(
    root: Path,
) -> None:

    modules = (
        sorted(
            set(
                SOURCE_MODULE_MOVES.values()
            )
        )
    )

    script = (
        "import importlib\n"
        f"modules = {modules!r}\n"
        "for module in modules:\n"
        "    importlib.import_module(module)\n"
        "print("
        "'Imported', "
        "len(modules), "
        "'moved tool modules.'"
        ")\n"
    )

    result = (
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
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
            "Moved tool module import smoke failed."
        )


# ============================================================
# COMPATIBILITY IMPORT SMOKE
# ============================================================


def compatibility_import_smoke(
    root: Path,
) -> None:
    """
    Verify package conversions retain their historical module
    names:

        tools.git
        tools.workspace
        tools.presentation
    """

    modules = (
        "tools.git",
        "tools.workspace",
        "tools.presentation",
    )

    script = (
        "import importlib\n"
        f"modules = {modules!r}\n"
        "for module in modules:\n"
        "    importlib.import_module(module)\n"
        "print("
        "'Compatibility package imports:', "
        "len(modules)"
        ")\n"
    )

    result = (
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
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
            "Compatibility package import smoke failed."
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

    state = (
        inspect_move_state(
            root
        )
    )

    if (
        state[
            MoveState.COMPLETE
        ]
        != 0
    ):

        raise SystemExit(
            "Migration appears partially applied. "
            "Use --resume instead."
        )

    print(
        "Tools Layout Refactor — DRY RUN"
    )

    print(
        "==============================="
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

        old_module = (
            SOURCE_MODULE_MOVES[
                module_from_python_path(
                    source
                )
            ]
        )

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
        "Compatibility-preserved module conversions:"
    )

    for module in (
        "tools.git",
        "tools.workspace",
        "tools.presentation",
    ):

        print(
            f"  {module}"
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
# APPLY / RESUME
# ============================================================


def apply_refactor(
    *,
    root: Path,
    resume: bool,
) -> None:

    assert_expected_head(
        root
    )

    state_before = (
        inspect_move_state(
            root
        )
    )

    if resume:

        if (
            state_before[
                MoveState.COMPLETE
            ]
            == 0
        ):

            raise SystemExit(
                "--resume requested but migration "
                "has not started."
            )

        print(
            "Tools Layout Refactor — RESUME"
        )

    else:

        if (
            state_before[
                MoveState.COMPLETE
            ]
            != 0
        ):

            raise SystemExit(
                "Migration is partially applied. "
                "Use --resume."
            )

        assert_clean_tree_except_script(
            root
        )

        print(
            "Tools Layout Refactor — APPLY"
        )

    print(
        "=============================="
    )

    print()

    # ========================================================
    # 1. MOVES
    # ========================================================

    print(
        "[1/9] Ensuring git moves..."
    )

    ensure_all_moves(
        root
    )

    # ========================================================
    # 2. PACKAGE INITIALIZERS
    # ========================================================

    print(
        "[2/9] Creating package initializers..."
    )

    create_package_files(
        root
    )

    # ========================================================
    # 3. DETERMINISTIC REBUILD
    # ========================================================

    print(
        "[3/9] Rebuilding moved Python files "
        "from committed HEAD..."
    )

    rebuild_all_moved_files(
        root
    )

    # ========================================================
    # 4. GLOBAL REWRITE
    # ========================================================

    print(
        "[4/9] Rewriting repository references..."
    )

    changed = (
        rewrite_repository_references(
            root
        )
    )

    print(
        f"      rewritten files: {changed}"
    )

    # ========================================================
    # 5. AST AUDIT
    # ========================================================

    print(
        "[5/9] Auditing stale Python imports..."
    )

    stale_imports = (
        find_stale_python_imports(
            root
        )
    )

    if stale_imports:

        print()

        print(
            "STALE PYTHON IMPORTS FOUND:"
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

    # ========================================================
    # 6. RAW REFERENCE AUDIT
    # ========================================================

    print(
        "[6/9] Auditing stale paths/module names..."
    )

    stale_references = (
        find_stale_references(
            root
        )
    )

    if stale_references:

        print()

        print(
            "STALE REFERENCES FOUND:"
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

    # ========================================================
    # 7. COMPILE
    # ========================================================

    print(
        "[7/9] Compiling repository..."
    )

    compile_repository(
        root
    )

    # ========================================================
    # 8. IMPORT SMOKE
    # ========================================================

    print(
        "[8/9] Importing moved production modules..."
    )

    import_smoke(
        root
    )

    # ========================================================
    # 9. COMPATIBILITY IMPORTS
    # ========================================================

    print(
        "[9/9] Verifying compatibility package imports..."
    )

    compatibility_import_smoke(
        root
    )

    state_after = (
        inspect_move_state(
            root
        )
    )

    expected_count = (
        len(
            ALL_FILE_MOVES
        )
    )

    if (
        state_after[
            MoveState.PENDING
        ]
        != 0
        or state_after[
            MoveState.COMPLETE
        ]
        != expected_count
    ):

        raise SystemExit(
            "Migration did not reach a complete "
            "filesystem state."
        )

    print()

    print(
        "TOOLS LAYOUT REFACTOR: PASS"
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
                "tools/ and tool unit tests."
            )
        )
    )

    mode = (
        parser.add_mutually_exclusive_group()
    )

    mode.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply migration from a clean tree."
        ),
    )

    mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a partially applied migration."
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
            root=(
                root
            ),

            resume=False,
        )

        return 0

    if args.resume:

        apply_refactor(
            root=(
                root
            ),

            resume=True,
        )

        return 0

    dry_run(
        root
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )