import pytest

from video_announce.about import normalize_about_text, normalize_about_with_fallback


def test_normalize_about_llm_preserves_words_and_casing():
    about = "  Amazing Show 🎉\nTONIGHT only  "

    normalized = normalize_about_text(about)

    assert normalized == "Amazing Show TONIGHT only"


def test_normalize_about_llm_can_keep_emoji():
    about = "  Amazing Show 🎉\nTONIGHT only  "

    normalized = normalize_about_text(about, strip_emojis=False)

    assert normalized == "Amazing Show 🎉 TONIGHT only"


def test_normalize_about_with_fallback_shortens_when_missing():
    normalized = normalize_about_with_fallback(
        None,
        title="Jazz Night",
        ocr_text="Jazz Night Club",
        fallback_parts=("Jazz Night. Bring friends",),
        word_limit=3,
    )

    assert normalized == "Jazz Night Bring"
