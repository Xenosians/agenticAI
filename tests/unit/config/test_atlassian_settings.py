from config.settings import Settings


def test_atlassian_admin_settings_are_optional_by_default():
    settings = Settings(_env_file=None)

    assert settings.atlassian_admin_api_key is None
    assert settings.atlassian_org_id is None
    assert settings.atlassian_admin_http_timeout_seconds > 0
