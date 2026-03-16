from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_module():
    module_path = Path("kaggle/GuideExcursionsMonitor/guide_excursions_monitor.py")
    spec = importlib.util.spec_from_file_location("guide_nb_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_extract_post_uses_block_first_for_announce_multi(monkeypatch):
    module = _load_module()

    async def _unexpected_ask_gemma(*args, **kwargs):
        raise AssertionError("whole-post ask_gemma should not run for announce_multi block-first extraction")

    async def _fake_extract_occurrence_block(source_payload, *, post, flags, screen, block):
        return {
            "source_block_id": block["id"],
            "canonical_title": f"Тест {block['id']}",
            "title_normalized": f"тест {block['id'].lower()}",
            "date": "2026-03-22" if block["id"] == "B1" else "2026-04-05",
            "time": "09:00" if block["id"] == "B1" else "10:00",
            "status": "scheduled",
            "availability_mode": "scheduled_public",
            "channel_url": post.source_url,
            "source_fingerprint": f"fp-{block['id']}",
            "fact_pack": {},
            "fact_claims": [],
            "guide_names": [],
            "organizer_names": [],
            "audience_fit": [],
        }

    monkeypatch.setattr(module, "ask_gemma", _unexpected_ask_gemma)
    monkeypatch.setattr(module, "_extract_occurrence_block", _fake_extract_occurrence_block)

    post = module.ScannedPost(
        message_id=5806,
        grouped_id=None,
        post_date=datetime(2026, 3, 11, 12, 2, 52, tzinfo=timezone.utc),
        source_url="https://t.me/amber_fringilla/5806",
        text=(
            "22 марта, Экопрогулка в Южном парке. Начало в 9:00.\n\n"
            "5 апреля - знакомство с историей растительного мира на острове Канта. Начало в 10:00."
        ),
        views=100,
        forwards=1,
        reactions_total=10,
        reactions_json={"❤": 10},
        media_refs=[],
    )
    out = await module.extract_post(
        {"username": "amber_fringilla", "source_kind": "guide_personal", "base_region": "Калининградская область"},
        post,
        {"has_date_signal": True, "has_time_signal": True, "has_price_signal": False, "has_booking_signal": False},
        {"post_kind": "announce_multi", "extract_mode": "announce", "base_region_fit": "inside"},
    )

    assert [item["source_block_id"] for item in out] == ["B1", "B2"]
    assert [item["canonical_title"] for item in out] == ["Тест B1", "Тест B2"]


@pytest.mark.asyncio
async def test_extract_post_routes_single_announce_through_tier1_and_enrich(monkeypatch):
    module = _load_module()

    calls: list[str] = []

    async def _fake_announce(source_payload, *, post, flags, screen):
        calls.append("announce")
        return [
            {
                "source_block_id": "B1",
                "canonical_title": "Экопрогулка в Южном парке",
                "title_normalized": "экопрогулка в южном парке",
                "date": "2026-03-22",
                "time": "09:00",
                "status": "scheduled",
                "availability_mode": "scheduled_public",
                "channel_url": post.source_url,
                "guide_names": ["Юлия Гришанова"],
                "fact_pack": {},
            }
        ]

    async def _fake_enrich(source_payload, *, post, flags, screen, occurrence_seed, focus_excerpt):
        calls.append("enrich")
        assert occurrence_seed["canonical_title"] == "Экопрогулка в Южном парке"
        return {
            "route_summary": "Южный парк и весенние птицы",
            "summary_one_liner": "Прогулка по Южному парку с акцентом на птиц.",
            "fact_pack": {"main_hook": "Птицы и первые приметы весны"},
        }

    monkeypatch.setattr(module, "_extract_announce_post_tier1", _fake_announce)
    monkeypatch.setattr(module, "_extract_occurrence_semantics", _fake_enrich)

    post = module.ScannedPost(
        message_id=5806,
        grouped_id=None,
        post_date=datetime(2026, 3, 11, 12, 2, 52, tzinfo=timezone.utc),
        source_url="https://t.me/amber_fringilla/5806",
        text="22 марта. Экопрогулка в Южном парке. Начало в 9:00.",
        views=100,
        forwards=1,
        reactions_total=10,
        reactions_json={"❤": 10},
        media_refs=[],
    )
    out = await module.extract_post(
        {"username": "amber_fringilla", "source_kind": "guide_personal", "base_region": "Калининградская область"},
        post,
        {"has_date_signal": True, "has_time_signal": True},
        {"post_kind": "announce_single", "extract_mode": "announce", "base_region_fit": "inside"},
    )

    assert calls == ["announce", "enrich"]
    assert out[0]["route_summary"] == "Южный парк и весенние птицы"
    assert out[0]["fact_pack"]["main_hook"] == "Птицы и первые приметы весны"


@pytest.mark.asyncio
async def test_extract_post_routes_status_and_template_modes_to_narrow_prompts(monkeypatch):
    module = _load_module()

    calls: list[str] = []

    async def _fake_status(source_payload, *, post, flags, screen):
        calls.append("status")
        return [
            {
                "canonical_title": "Южный Амалиенау",
                "title_normalized": "южный амалиенау",
                "date": "2026-03-22",
                "status": "scheduled",
                "availability_mode": "scheduled_public",
                "fact_pack": {},
                "fact_claims": [{"claim_role": "status_delta", "fact_type": "seats", "fact_value": "осталось 3 места"}],
            }
        ]

    async def _fake_template(source_payload, *, post, flags, screen):
        calls.append("template")
        return [
            {
                "canonical_title": "Роминтенская прогулка",
                "title_normalized": "роминтенская прогулка",
                "availability_mode": "on_demand",
                "digest_eligible": False,
                "digest_eligibility_reason": "missing_date",
                "template_hint": {"route_topics": ["пуща", "Краснолесье"]},
                "fact_pack": {},
            }
        ]

    monkeypatch.setattr(module, "_extract_status_post", _fake_status)
    monkeypatch.setattr(module, "_extract_template_post", _fake_template)

    post = module.ScannedPost(
        message_id=1,
        grouped_id=None,
        post_date=datetime(2026, 3, 11, 12, 2, 52, tzinfo=timezone.utc),
        source_url="https://t.me/example/1",
        text="Тест",
        views=0,
        forwards=0,
        reactions_total=0,
        reactions_json={},
        media_refs=[],
    )

    status_out = await module.extract_post({}, post, {}, {"post_kind": "status_update", "extract_mode": "status", "base_region_fit": "inside"})
    template_out = await module.extract_post({}, post, {}, {"post_kind": "template_signal", "extract_mode": "template", "base_region_fit": "inside"})

    assert calls == ["status", "template"]
    assert status_out[0]["fact_claims"][0]["claim_role"] == "status_delta"
    assert template_out[0]["digest_eligible"] is False
