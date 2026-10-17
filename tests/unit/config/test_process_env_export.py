"""Provider-endpoint policy in .env reaches the extension transport."""

from config.settings import PROCESS_ENV_FROM_DOTENV, export_process_env_from_dotenv


def test_dotenv_policy_is_exported_without_overriding_process_env(tmp_path, monkeypatch):
    for name in PROCESS_ENV_FROM_DOTENV:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "EXTENSION_ALLOWED_ORIGINS=http://198.51.100.7:20128\n"
        "EXTENSION_INSECURE_HTTP_ORIGINS=http://198.51.100.7:20128\n"
        "JIRA_API_TOKEN=not-exported\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXTENSION_ALLOWED_ORIGINS", "https://kept.example.test")

    exported = export_process_env_from_dotenv(env)

    import os
    assert exported == ["EXTENSION_INSECURE_HTTP_ORIGINS"]
    assert os.environ["EXTENSION_ALLOWED_ORIGINS"] == "https://kept.example.test"
    assert os.environ["EXTENSION_INSECURE_HTTP_ORIGINS"] == "http://198.51.100.7:20128"
    assert "JIRA_API_TOKEN" not in exported


def test_missing_dotenv_is_a_no_op(tmp_path):
    assert export_process_env_from_dotenv(tmp_path / "absent.env") == []
