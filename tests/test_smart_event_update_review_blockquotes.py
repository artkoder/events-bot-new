from smart_event_update import _promote_review_bullets_to_blockquotes


def test_promote_review_bullets_to_blockquotes_keeps_lists_and_quotes_reviews() -> None:
    src = (
        "### Актёрский состав\n\n"
        "- Татьяна Васильева\n"
        "- Игорь Письменный\n\n"
        "### Особенности постановки и отзывы\n\n"
        "Зрители в восторге:\n"
        "- Лариса: Превосходный спектакль!!! Все актеры великолепно играли!\n"
        "- Мария: Отличный спектакль!!! Посмеялись от души!\n"
        "- Михаил: Великолепный спектакль!!! Остались очень довольны.\n"
    )

    out = _promote_review_bullets_to_blockquotes(src) or ""

    assert "- Татьяна Васильева" in out
    assert "- Игорь Письменный" in out

    assert (
        "> «Превосходный спектакль!!! Все актеры великолепно играли!»\n> — Лариса" in out
    )
    assert "> «Отличный спектакль!!! Посмеялись от души!»\n> — Мария" in out
    assert "> «Великолепный спектакль!!! Остались очень довольны.»\n> — Михаил" in out

    assert "- Лариса:" not in out
    assert "- Мария:" not in out
    assert "- Михаил:" not in out
