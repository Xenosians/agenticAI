# Developer project tools

`workspace_project_info`, `workspace_run_tests`, and `workspace_run_build`
accept an optional `repository` argument. It uses the same configured logical
aliases as workspace discovery and governed Git. Omitting it, or supplying
JSON `null`, selects `GIT_DEFAULT_REPOSITORY`.

Use `repository` for an alias and `relative_path` for a directory inside that
repository. For example, `{"repository": "backend", "relative_path": "."}`
selects the backend repository root; an alias in `relative_path` does not
select another repository. Omitted or null MCP paths mean the selected
repository root. Absolute paths, traversal, and links outside the selected
repository are denied.

Model-facing alias values come from the canonical repository registry.
Repository arguments remain grounded in the request, and test/build tools
still require the existing medium-risk approval before executing repository
code. Results, result cards, and approval text identify the repository.

Project detection and fixed commands continue to support Python (`pytest`
and `compileall`), Elixir (`mix test` and `mix compile`), and Nim (`nimble test`
and `nimble build`). Selecting a repository does not add another project
strategy or change the process policy. No new tool identifiers are introduced.
