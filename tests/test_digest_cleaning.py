import pytest
import os
from digest_helper import clean_search_digest

def test_clean_search_digest_timestamps():
    assert clean_search_digest("Concert at 19:00 today") == "Concert at today"
    assert clean_search_digest("Start 9:00 end 21:00") == "Start end"
    assert clean_search_digest("12:00 meeting") == "meeting"
    assert clean_search_digest("No time here") == "No time here"

def test_clean_search_digest_normalization():
    assert clean_search_digest("  Too   many    spaces  ") == "Too many spaces"
    assert clean_search_digest("12:00") is None
    assert clean_search_digest("") is None
    assert clean_search_digest(None) is None
    assert clean_search_digest("   12:00   ") is None

def test_clean_search_digest_complex():
    input_str = "Лекция о  искусстве  начало в 18:30, вход свободный"
    expected = "Лекция о искусстве начало в , вход свободный"
    assert clean_search_digest(input_str) == expected

def test_prompts_file_content():
    """Smoke test to ensure PROMPTS.md will contain new rules"""
    prompts_path = os.path.join(os.path.dirname(__file__), "..", "docs", "llm", "prompts.md")
    with open(prompts_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "extract 1-2 highlights like \"musical warm-up\"" in content or "musical warm-up" in content
    assert "по отзывам —" in content
    assert "Poster OCR" in content
