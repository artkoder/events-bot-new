from __future__ import annotations

from poster_media import PosterMedia
from vk_intake import (
    EventDraft,
    _collapse_same_post_exact_drafts,
    _vk_wall_source_ids_from_url,
)


def _poster(label: str, payload: bytes) -> PosterMedia:
    return PosterMedia(
        data=payload,
        name=f"{label}.jpg",
        ocr_text=f"ocr {label}",
    )


def test_vk_wall_source_ids_from_url_parses_wall_owner_and_post_id() -> None:
    assert _vk_wall_source_ids_from_url("https://vk.com/wall-212233232_1680") == (
        212233232,
        1680,
    )
    assert _vk_wall_source_ids_from_url("https://vk.com/wall212233232_1680") == (
        212233232,
        1680,
    )
    assert _vk_wall_source_ids_from_url("https://example.com/post/1") == (None, None)


def test_collapse_same_post_exact_drafts_merges_identical_child_slot() -> None:
    pitch_short = EventDraft(
        title="Питчинг идей и клубов в «ТеплоСети»",
        date="2026-03-14",
        time="18:00",
        venue="ОЦК ТеплоСеть",
        description="Короткое описание.",
        location_address="Ленина 23",
        city="Советск",
        links=["https://vk.cc/pitch"],
        poster_media=[_poster("pitch-a", b"a")],
        source_text="Питчинг в Теплосети.",
    )
    pitch_long = EventDraft(
        title="Питчинг идей и клубов в ТеплоСети",
        date="2026-03-14",
        time="18:00",
        venue="ОЦК ТеплоСеть",
        description="Более длинное описание питчинга идей и клубов в Теплосети.",
        location_address="Ленина 23",
        city="Советск",
        links=["https://vk.cc/pitch/"],
        poster_media=[_poster("pitch-b", b"b")],
        search_digest="Питчинг в Теплосети.",
        source_text="Развёрнутый исходный текст о питчинге идей и клубов.",
    )
    knitting = EventDraft(
        title="Мастер-класс: вязание изделий из пакетов",
        date="2026-03-10",
        time="18:00",
        venue="ОЦК ТеплоСеть",
        description="Отдельный мастер-класс.",
    )

    result = _collapse_same_post_exact_drafts([pitch_short, pitch_long, knitting])

    assert len(result) == 2
    merged_pitch = next(d for d in result if "Питчинг" in d.title)
    assert merged_pitch.description == pitch_long.description
    assert merged_pitch.location_address == "Ленина 23"
    assert merged_pitch.city == "Советск"
    assert merged_pitch.search_digest == "Питчинг в Теплосети."
    assert merged_pitch.links == ["https://vk.cc/pitch/"]
    assert len(merged_pitch.poster_media) == 2
    assert any(d.title == knitting.title for d in result)
