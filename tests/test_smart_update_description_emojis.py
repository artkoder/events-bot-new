from smart_event_update import _limit_emoji_sequences


def test_limit_emoji_sequences_keeps_first_n() -> None:
    text = "### Заголовок 🎭\n\nАбзац 🎨 Текст 🎵 Ещё 🎬"
    out = _limit_emoji_sequences(text, max_keep=3)
    assert "🎭" in out
    assert "🎨" in out
    assert "🎵" in out
    assert "🎬" not in out


def test_limit_emoji_sequences_handles_zwj_sequences() -> None:
    family = "👨\u200d👩\u200d👧\u200d👦"  # ZWJ sequence
    text = f"Вместе {family} и ещё 😀 потом 😃 и 😄"
    out = _limit_emoji_sequences(text, max_keep=2)
    assert family in out
    assert "😀" in out
    assert "😃" not in out
    assert "😄" not in out


def test_limit_emoji_sequences_zero_removes_all() -> None:
    text = "Текст 🎭 🎨 🎵"
    out = _limit_emoji_sequences(text, max_keep=0)
    assert "🎭" not in out
    assert "🎨" not in out
    assert "🎵" not in out

