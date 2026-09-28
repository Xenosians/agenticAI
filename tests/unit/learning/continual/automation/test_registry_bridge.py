from learning.continual.automation.config import (
    ContinualAutomationSettings,
    load_declared_corpus_sources,
)

from learning.paths import (
    REPOSITORY_ROOT,
)


def test_real_registry_loads_without_legacy_list_contract():
    settings = ContinualAutomationSettings(
        corpus_registry_path=(
            REPOSITORY_ROOT
            / "config"
            / "continual_corpus_registry.json"
        )
    )

    sources = load_declared_corpus_sources(
        settings
    )

    # Remote/runtime registry declarations do not enter the
    # legacy optimizer corpus reader directly.
    assert sources == []
