from __future__ import annotations

import smart_event_update as su


def test_fact_first_bucket_allows_yandex_music_playlist_url() -> None:
    fact = "Есть плейлист на Я.Музыке: https://music.yandex.ru/users/u/playlists/1030"
    assert su._fact_first_bucket(fact) == "text_clean"


def test_fact_first_bucket_blocks_other_urls() -> None:
    fact = "Ссылка: https://example.com/foo"
    assert su._fact_first_bucket(fact) == "infoblock"


def test_fact_first_forbidden_reasons_allows_playlist_url_in_description() -> None:
    desc = "Полный список песен — в плейлисте: https://music.yandex.ru/users/u/playlists/1030"
    reasons = su._fact_first_forbidden_reasons(desc, anchors=[])
    assert "url" not in reasons


def test_fact_first_forbidden_reasons_flags_disallowed_url_in_description() -> None:
    desc = "Подробности тут: https://example.com/foo"
    reasons = su._fact_first_forbidden_reasons(desc, anchors=[])
    assert "url" in reasons


def test_fact_first_text_clean_strips_chat_url_but_keeps_meaning() -> None:
    facts = [
        "Есть чат для знакомства и общения участников: https://t.me/+abcdef",
        "Есть плейлист на Я.Музыке: https://music.yandex.ru/users/u/playlists/1030",
    ]
    out = su._facts_text_clean_from_facts(facts, anchors=[])
    assert any("Есть чат" in f and "http" not in f for f in out)
    assert any("music.yandex.ru" in f for f in out)

