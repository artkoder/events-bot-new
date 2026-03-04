from __future__ import annotations

import main


def test_normalize_known_venue_mentions_replaces_noisy_prefixes() -> None:
    text = "Место проведения: кинотеатр «Сигнал». Вход свободный."
    out = main._normalize_known_venue_mentions(
        text,
        location_name="Сигнал (арт-пространство), Леонова 22, Калининград",
    )
    assert out is not None
    assert "кинотеатр" not in out.lower()
    assert "Сигнал (арт-пространство)" in out

