import re

from smart_event_update import _dedupe_description, _sanitize_description_output


def test_dedupe_description_preserves_numbered_lists() -> None:
    text = (
        "### Что в программе\n\n"
        "1. Чиж & Co — «О любви»\n"
        "2. Анна Герман — «Город влюблённых»\n"
        "3. Lady Gaga & Bradley Cooper — Shallow\n"
        "4. Frank Sinatra — I Love You Baby\n\n"
        "Участникам не нужны профессиональные вокальные навыки."
    )
    out = _dedupe_description(text)
    assert out is not None
    assert "### Что в программе" in out
    assert re.search(r"(?m)^1\.\s+Чиж", out)
    assert re.search(r"(?m)^2\.\s+Анна", out)
    assert re.search(r"(?m)^3\.\s+Lady", out)
    assert re.search(r"(?m)^4\.\s+Frank", out)


def test_sanitize_description_does_not_flatten_lists() -> None:
    text = (
        "### Репертуар\n\n"
        "1. Песня первая\n"
        "2. Песня вторая\n"
        "3. Песня третья\n"
    )
    out = _sanitize_description_output(text, source_text="В репертуаре 3 песни.")
    assert out is not None
    assert "### Репертуар" in out
    assert re.search(r"(?m)^1\.\s+Песня первая$", out)
    assert re.search(r"(?m)^2\.\s+Песня вторая$", out)
    assert re.search(r"(?m)^3\.\s+Песня третья$", out)
