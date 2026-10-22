"""Missing deployment configuration cannot silently choose simulated providers."""
import pytest

from config import Settings
from services.assets import MockAssetService, build_asset_service
from services.directory import MockDirectoryService, build_directory_service
from services.knowledge import MockKnowledgeService, build_knowledge_service
from services.ticketing import MockTicketService, build_ticket_service


PROVIDERS = [
    ("directory_backend", build_directory_service, MockDirectoryService),
    ("ticketing_backend", build_ticket_service, MockTicketService),
    ("asset_backend", build_asset_service, MockAssetService),
    ("knowledge_backend", build_knowledge_service, MockKnowledgeService),
]


@pytest.fixture
def isolated_settings(monkeypatch):
    for field, _, _ in PROVIDERS:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)
    return Settings(_env_file=None)


@pytest.mark.parametrize("field,factory,mock_type", PROVIDERS)
def test_omission_stays_unconfigured_and_factory_refuses(isolated_settings, field, factory, mock_type):
    assert getattr(isolated_settings, field) == "unconfigured"
    with pytest.raises(RuntimeError, match=f"{field.upper()} is not configured"):
        factory(isolated_settings)


@pytest.mark.parametrize("field,factory,mock_type", PROVIDERS)
def test_mock_requires_explicit_selection(isolated_settings, field, factory, mock_type):
    settings = isolated_settings.model_copy(update={field: "mock"})
    assert isinstance(factory(settings), mock_type)


def test_failed_real_jira_configuration_does_not_fall_back_to_mock(isolated_settings):
    settings = isolated_settings.model_copy(update={
        "ticketing_backend": "jira", "jira_base_url": None,
        "jira_email": None, "jira_api_token": None,
    })
    with pytest.raises(RuntimeError, match="JIRA_BASE_URL"):
        build_ticket_service(settings)
