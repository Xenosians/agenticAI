"""The README's request-flow animation stays a safe, self-contained image that
matches the stages it names.

GitHub shows an SVG in a README as an image: script and external resources
are not run, so the animation must be CSS and SMIL only.
"""
# The parsed text is this repository's own generator output, never external
# input, so the standard library parser is safe here.
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.render_request_flow import STAGES, render

SVG = "{http://www.w3.org/2000/svg}"
COMMITTED = Path(__file__).resolve().parents[3] / "docs" / "assets" / "request-flow.svg"


def test_the_committed_animation_is_the_generator_output():
    assert COMMITTED.read_text(encoding="utf-8") == render()


def test_the_animation_has_no_script_or_external_resource():
    text = render()
    root = ET.fromstring(text)

    assert root.findall(f".//{SVG}script") == []
    assert "href=" not in text and "url(http" not in text and "@import" not in text
    assert "on" + "load" not in text.lower() and "javascript:" not in text.lower()


def test_every_stage_is_drawn_lit_in_turn_and_described_for_screen_readers():
    text = render()
    root = ET.fromstring(text)
    titles = [node.text for node in root.iter(f"{SVG}text") if node.get("class") == "title"]
    description = root.findtext(f"{SVG}desc", default="")

    assert titles == [stage.title for stage in STAGES]
    for stage in STAGES:
        assert f"@keyframes lit-{stage.key}" in text
        assert stage.note in description
