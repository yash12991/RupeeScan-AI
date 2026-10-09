from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ElementCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_frontend_ids_are_unique_and_toggle_state_is_accessible():
    parser = ElementCollector()
    parser.feed((PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8"))

    ids = [attrs["id"] for _, attrs in parser.elements if "id" in attrs]
    assert len(ids) == len(set(ids))

    by_id = {attrs.get("id"): attrs for _, attrs in parser.elements if attrs.get("id")}
    assert by_id["toggle-contrast"]["aria-pressed"] == "false"
    assert by_id["toggle-speech"]["aria-pressed"] == "true"
    assert by_id["detection-display"]["aria-live"] == "polite"
    assert by_id["image-upload"]["accept"] == "image/*"


def test_safety_disclaimer_is_visible_in_initial_html():
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8").lower()
    assert "authenticity estimate is experimental" in html
    assert "must not be used as proof" in html
