import os
from digest_helper import (
    clean_search_digest,
    enforce_digest_word_limit,
    is_short_description_acceptable,
)

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

def test_clean_search_digest_rejects_truncated():
    assert clean_search_digest("Something…") is None
    assert clean_search_digest("Something...") is None

def test_clean_search_digest_complex():
    input_str = "Лекция о  искусстве  начало в 18:30, вход свободный"
    expected = "Лекция о искусстве начало в , вход свободный"
    assert clean_search_digest(input_str) == expected


def test_enforce_digest_word_limit_trims_long_text():
    text = "раз два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать"
    out = enforce_digest_word_limit(text, max_words=16)
    assert out is not None
    assert len(out.split()) <= 16
    assert out.endswith("…")


def test_is_short_description_acceptable_requires_complete_sentence_and_length():
    good = "Музыкальный вечер знакомит с солдатскими балладами и народными песнями о мужестве России."
    bad_ellipsis = "Музыкальный вечер знакомит с солдатскими балладами и народными песнями о мужестве…"
    bad_too_long = (
        "Музыкальный вечер знакомит с солдатскими балладами и народными песнями, "
        "а также с авторскими композициями о памяти, доблести и стойкости народа."
    )
    assert is_short_description_acceptable(good, min_words=12, max_words=16) is True
    assert is_short_description_acceptable(bad_ellipsis, min_words=12, max_words=16) is False
    assert is_short_description_acceptable(bad_too_long, min_words=12, max_words=16) is False

def test_prompts_file_content():
    """Smoke test to ensure PROMPTS.md will contain new rules"""
    prompts_path = os.path.join(os.path.dirname(__file__), "..", "docs", "PROMPTS.md")
    with open(prompts_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "extract 1-2 highlights like \"musical warm-up\"" in content or "musical warm-up" in content
    assert "по отзывам —" in content
    assert "Poster OCR" in content
