from pathlib import Path

from learning.curriculum.catalog import get_curriculum
from learning.curriculum.mastery import CurriculumState
from learning.continual.storage import read_json_model
from learning.training.membership import ensure_curriculum_state


def test_missing_developer_state_is_initialized(tmp_path: Path):
    curriculum = get_curriculum("developer")
    state_path = tmp_path / "developer-state.json"

    assert not state_path.exists()

    state = ensure_curriculum_state(
        curriculum=curriculum,
        state_path=state_path,
    )

    assert state_path.is_file()
    assert state.current_chapter_id == "dev-01-workspace-reads"

    stored = read_json_model(
        state_path,
        CurriculumState,
    )

    assert stored.curriculum_id == curriculum.curriculum_id
    assert stored.current_chapter_id == "dev-01-workspace-reads"


def test_existing_state_is_reused(tmp_path: Path):
    curriculum = get_curriculum("developer")
    state_path = tmp_path / "developer-state.json"

    first = ensure_curriculum_state(
        curriculum=curriculum,
        state_path=state_path,
    )

    second = ensure_curriculum_state(
        curriculum=curriculum,
        state_path=state_path,
    )

    assert second.model_dump() == first.model_dump()
