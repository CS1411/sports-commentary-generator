"""Milestone 3: the Streamlit interface and generation helpers. No model is downloaded: loaders are replaced."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import model as lm

PAGE = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


def test_clean_output_keeps_the_first_line_only():
    assert lm.clean_output("  Goal for Riverside!\nTone: calm\nEvent: x") == "Goal for Riverside!"
    assert lm.clean_output("Great strike. Tone: calm") == "Great strike."
    assert lm.clean_output("") == ""


def test_page_renders_with_its_controls():
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Sports Commentary Generator"
    assert at.radio[0].options == ["excited", "calm"]
    assert len(at.selectbox[0].options) > 5  # held-out examples to try


def test_generate_button_shows_the_commentary(monkeypatch):
    monkeypatch.setattr(lm, "load_tokenizer", lambda: "tok")
    monkeypatch.setattr(lm, "load_adapted", lambda: "adapted")
    monkeypatch.setattr(lm, "load_base", lambda: "base")
    monkeypatch.setattr(lm, "generate_commentary",
                        lambda model, tok, event, tone, n, t: f"[{model}/{tone}] {event[:12]}")
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.text_area[0].set_value("34' GOAL! Test scores.").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.success[0].value == "[adapted/excited] 34' GOAL! Te"


def test_empty_event_asks_for_input():
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.button[0].click().run()
    assert at.warning[0].value == "Write an event first."
