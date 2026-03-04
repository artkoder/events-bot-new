from __future__ import annotations

from datetime import date

import vk_intake


def test_maybe_rollover_llm_iso_date_keeps_recent_past_mentions() -> None:
    # Recap-style post: publish date shortly after the event date, no explicit year in text.
    anchor = date(2026, 2, 14)
    assert (
        vk_intake._maybe_rollover_llm_iso_date(
            "2026-02-12",
            anchor_date=anchor,
            has_explicit_year_in_text=False,
        )
        == "2026-02-12"
    )


def test_maybe_rollover_llm_iso_date_rolls_far_past_to_next_year() -> None:
    # Implicit-year mention far in the past should be treated as next year's date.
    anchor = date(2026, 12, 15)
    assert (
        vk_intake._maybe_rollover_llm_iso_date(
            "2026-01-01",
            anchor_date=anchor,
            has_explicit_year_in_text=False,
        )
        == "2027-01-01"
    )


def test_recap_context_rejects_vague_future_teaser_title() -> None:
    text = (
        "Мы знаем, что вы любите Миядзаки!\n"
        "Поэтому 12 февраля «Молодежная академия искусств» вновь исполнила программу.\n"
        "...\n"
        "19 марта музыканты исполнят тематический концерт. Подробности и билеты по ссылке.\n"
    )
    reason = vk_intake._looks_like_recent_recap_with_past_date(
        source_text=text,
        anchor_date=date(2026, 2, 14),
    )
    assert reason
    assert vk_intake._looks_like_vague_teaser_title("🎶 Тематический концерт")


def test_non_recap_text_does_not_trigger_recap_guard() -> None:
    text = "19 марта состоится тематический концерт. Подробности и билеты по ссылке."
    assert (
        vk_intake._looks_like_recent_recap_with_past_date(
            source_text=text,
            anchor_date=date(2026, 2, 14),
        )
        is None
    )

