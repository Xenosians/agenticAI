import tools.workspace.discovery as discovery


def configure_workspace(
    monkeypatch,
    root,
):
    monkeypatch.setattr(
        discovery,
        "workspace_root",
        lambda: root,
    )


def test_workspace_list_returns_safe_entries(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "src"
    ).mkdir()

    (
        tmp_path
        / "README.md"
    ).write_text(
        "hello",
        encoding="utf-8",
    )

    (
        tmp_path
        / ".env"
    ).write_text(
        "PASSWORD=secret",
        encoding="utf-8",
    )

    (
        tmp_path
        / "credentials.txt"
    ).write_text(
        "hidden",
        encoding="utf-8",
    )

    (
        tmp_path
        / ".git"
    ).mkdir()

    result = (
        discovery.workspace_list()
    )

    assert result[
        "ok"
    ] is True

    names = {
        entry[
            "name"
        ]
        for entry
        in result[
            "entries"
        ]
    }

    assert "src" in names
    assert "README.md" in names

    assert ".env" not in names
    assert ".git" not in names
    assert "credentials.txt" not in names


def test_workspace_search_finds_source_matches(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    src = (
        tmp_path
        / "src"
    )

    src.mkdir()

    (
        src
        / "jobs.py"
    ).write_text(
        (
            "def send_heartbeat():\n"
            "    return 'heartbeat callback'\n"
        ),
        encoding="utf-8",
    )

    (
        src
        / "other.py"
    ).write_text(
        "print('nothing here')\n",
        encoding="utf-8",
    )

    result = (
        discovery.workspace_search(
            "heartbeat"
        )
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "files_scanned"
    ] == 2

    assert len(
        result[
            "matches"
        ]
    ) == 2

    assert result[
        "matches"
    ][
        0
    ][
        "path"
    ] == "src/jobs.py"


def test_workspace_search_skips_sensitive_files(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "normal.py"
    ).write_text(
        "needle\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / "secret_token.py"
    ).write_text(
        "needle\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / ".env"
    ).write_text(
        "needle\n",
        encoding="utf-8",
    )

    result = (
        discovery.workspace_search(
            "needle"
        )
    )

    paths = {
        match[
            "path"
        ]
        for match
        in result[
            "matches"
        ]
    }

    assert paths == {
        "normal.py"
    }


def test_workspace_search_skips_runtime_directories(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / "app.py"
    ).write_text(
        "needle\n",
        encoding="utf-8",
    )

    runtime = (
        tmp_path
        / ".runtime"
    )

    runtime.mkdir()

    (
        runtime
        / "state.py"
    ).write_text(
        "needle\n",
        encoding="utf-8",
    )

    result = (
        discovery.workspace_search(
            "needle"
        )
    )

    paths = [
        match[
            "path"
        ]
        for match
        in result[
            "matches"
        ]
    ]

    assert paths == [
        "app.py"
    ]


def test_workspace_list_blocks_path_escape(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    result = (
        discovery.workspace_list(
            "../outside"
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_workspace_search_blocks_path_escape(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    result = (
        discovery.workspace_search(
            query="anything",
            relative_path=(
                "../outside"
            ),
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_workspace_file_info_returns_metadata(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    src = (
        tmp_path
        / "src"
    )

    src.mkdir()

    target = (
        src
        / "main.py"
    )

    target.write_text(
        "print('hello')\n",
        encoding="utf-8",
    )

    result = (
        discovery.workspace_file_info(
            "src/main.py"
        )
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "path"
    ] == "src/main.py"

    assert result[
        "type"
    ] == "file"

    assert result[
        "readable_text"
    ] is True

    assert result[
        "size_bytes"
    ] > 0


def test_workspace_file_info_blocks_sensitive_path(
    tmp_path,
    monkeypatch,
):
    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    (
        tmp_path
        / ".env"
    ).write_text(
        "SECRET=value",
        encoding="utf-8",
    )

    result = (
        discovery.workspace_file_info(
            ".env"
        )
    )

    assert result[
        "ok"
    ] is False

    assert result[
        "status"
    ] == "denied"


def test_workspace_list_uses_logical_repository(
    tmp_path,
    monkeypatch,
):

    backend = (
        tmp_path
        / "backend"
    )

    backend.mkdir()

    (
        backend
        / "mix.exs"
    ).write_text(
        "defmodule Demo.MixProject do\nend\n",
        encoding="utf-8",
    )

    class Target:
        name = "backend"
        path = backend

    def fake_resolve_repository(
        repository,
    ):
        assert (
            repository
            == "backend"
        )

        return Target()

    monkeypatch.setattr(
        discovery,
        "resolve_workspace_repository",
        fake_resolve_repository,
    )

    result = (
        discovery.workspace_list(
            repository="backend"
        )
    )

    assert result[
        "ok"
    ] is True

    assert result[
        "repository"
    ] == "backend"

    assert result[
        "path"
    ] == "."

    names = {
        item[
            "name"
        ]

        for item
        in result[
            "entries"
        ]
    }

    assert (
        "mix.exs"
        in names
    )


def test_workspace_tools_expose_logical_repository_contract():

    from tools.registry import (
        get_tool,
    )

    for tool_name in [
        "workspace_read_text",
        "workspace_list",
        "workspace_search",
        "workspace_file_info",
    ]:

        tool = (
            get_tool(
                tool_name
            )
        )

        assert tool is not None

        assert (
            "repository"
            in tool[
                "parameters"
            ]
        )

        assert (
            "repository"
            in tool[
                "grounded_arguments"
            ]
        )

        assert callable(
            tool.get(
                "argument_values_resolver"
            )
        )


def test_workspace_search_grounds_query_and_scope():

    from tools.registry import (
        get_tool,
    )

    tool = (
        get_tool(
            "workspace_search"
        )
    )

    assert tool is not None

    assert set(
        tool[
            "grounded_arguments"
        ]
    ) == {
        "repository",
        "query",
        "relative_path",
    }


def test_workspace_listing_hides_generated_and_local_env_paths(
    tmp_path,
    monkeypatch,
):

    configure_workspace(
        monkeypatch,
        tmp_path,
    )

    for directory_name in [
        ".elixir_ls",
        "_build",
        "deps",
        "surreal_data",
    ]:
        (
            tmp_path
            / directory_name
        ).mkdir()

    (
        tmp_path
        / ".envrc"
    ).write_text(
        "SECRET=value\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / "erl_crash.dump"
    ).write_text(
        "runtime state\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / "README.md"
    ).write_text(
        "safe\n",
        encoding="utf-8",
    )

    result = (
        discovery.workspace_list()
    )

    assert result[
        "ok"
    ] is True

    names = {
        item[
            "name"
        ]

        for item
        in result[
            "entries"
        ]
    }

    assert "README.md" in names

    assert ".elixir_ls" not in names
    assert "_build" not in names
    assert "deps" not in names
    assert "surreal_data" not in names
    assert ".envrc" not in names
    assert "erl_crash.dump" not in names
