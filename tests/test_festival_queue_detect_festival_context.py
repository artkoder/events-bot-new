from festival_queue import detect_festival_context


def test_detect_festival_context_boolean_festival_flag_uses_title_as_name() -> None:
    decision = detect_festival_context(
        parsed_events=[
            {
                "title": "Восточный Новый год",
                "date": "2026-02-17",
                "end_date": "2026-03-02",
                "time": "16:00",
                "location_name": "Морской выставочный центр",
                "event_type": "festival",
                "festival": True,
            }
        ],
        festival_payload=None,
        source_text="Восточный Новый год уже начался! В 2026 его празднуют с 17 февраля по 2 марта.",
    )

    assert decision.context == "festival_post"
    assert decision.festival == "Восточный Новый год"
    assert decision.signals["range_signal"] == 1
    assert decision.signals["multi_signal"] == 1


def test_detect_festival_context_event_with_festival_keeps_string_name() -> None:
    decision = detect_festival_context(
        parsed_events=[
            {
                "title": "Музейные натюрморты",
                "date": "2026-02-15",
                "time": None,
                "end_date": None,
                "location_name": "Музей изобразительных искусств",
                "event_type": "exhibition",
                "festival": "Гофман – наш современник",
            }
        ],
        festival_payload=None,
        source_text="«Музейные натюрморты» ждут вас в зале Фестиваля книжной графики и иллюстрации «Гофман – наш современник».",
    )

    assert decision.context == "event_with_festival"
    assert decision.festival == "Гофман – наш современник"


def test_detect_festival_context_regrounds_wrong_day_series_name_from_source_text() -> None:
    decision = detect_festival_context(
        parsed_events=[
            {
                "title": "День рождения зеленоградского кота",
                "date": "2026-03-01",
                "time": "12:00",
                "event_type": "festival",
                "festival": "День города Зеленоградск",
            }
        ],
        festival_payload=None,
        source_text=(
            "1 марта в Зеленоградске пройдет праздник-фестиваль "
            "«День рождения зеленоградского кота»."
        ),
    )

    assert decision.context == "festival_post"
    assert decision.festival == "День рождения зеленоградского кота"
