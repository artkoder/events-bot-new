from source_parsing.handlers import AddedEventInfo, SourceParsingResult, format_parsing_report


def test_parse_report_includes_source_and_fact_stats_in_smart_update_block():
    result = SourceParsingResult(
        total_events=1,
        processing_duration=1.2,
        added_events=[
            AddedEventInfo(
                event_id=1,
                title="Фигаро",
                telegraph_url="https://telegra.ph/Figaro-02-10",
                ics_url="https://example.test/figaro.ics",
                log_cmd="/log 1",
                date="2026-02-12",
                time="19:00",
                source="dramteatr",
                source_url="https://dramteatr39.ru/spektakli/figaro",
                fact_stats={"added": 3, "duplicate": 2, "conflict": 1, "note": 4},
            )
        ],
    )

    text = format_parsing_report(result, bot_username="eventsbotTestBot")
    assert "Smart Update (детали событий)" in text
    assert "Источник:" in text
    assert "dramteatr39" in text
    assert "Факты: ✅3 ↩️2 ⚠️1 ℹ️4" in text
    assert "start=log_1" in text
