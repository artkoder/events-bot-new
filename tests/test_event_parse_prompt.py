import main


def test_event_parse_prompt_mentions_completed_event_reports() -> None:
    prompt = main._read_base_prompt()
    low = prompt.lower()
    assert "post-event report / recap" in low
    assert "ignore the recap part and extract" in low
    assert "concrete attendable future anchor" in low
