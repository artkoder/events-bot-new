from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from db import Database
import kaggle_registry
from guide_excursions import enrich as guide_enrich
from guide_excursions import service as guide_service
from ops_run import finish_ops_run, start_ops_run


def _load_guide_kaggle_monitor_module():
    path = Path(__file__).resolve().parent.parent / "kaggle" / "GuideExcursionsMonitor" / "guide_excursions_monitor.py"
    spec = importlib.util.spec_from_file_location("tests_guide_kaggle_monitor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kaggle_monitor_parses_bare_array_occurrence_payload():
    mod = _load_guide_kaggle_monitor_module()
    data = mod._extract_json(
        """
        ```json
        [
          {
            "source_block_id": "B1",
            "canonical_title": "Южный Амалиенау. История района в судьбах людей"
          }
        ]
        ```
        """
    )
    items = mod._coerce_occurrence_items(data)
    assert len(items) == 1
    assert items[0]["source_block_id"] == "B1"
    assert items[0]["canonical_title"] == "Южный Амалиенау. История района в судьбах людей"


def test_audience_percent_scores_accept_compact_ten_scale():
    assert guide_enrich._normalize_percent_score(7) == 70
    assert guide_enrich._normalize_percent_score("9") == 90
    assert guide_enrich._normalize_percent_score(72) == 72
    assert guide_enrich._normalize_percent_score(0) == 0


def test_enrich_retry_after_parser_reads_provider_hint():
    assert guide_enrich._retry_after_seconds("Rate limit exceeded: tpm (retry after 12000ms)") == 12.0
    assert guide_enrich._retry_after_seconds("plain failure") is None


def test_normalize_digest_eligibility_promotes_limited_fact_rich_occurrence():
    eligible, reason = guide_service._normalize_digest_eligibility(
        date_iso="2026-03-26",
        availability_mode="limited",
        status="available",
        time_text="10:30",
        city="Калининград",
        meeting_point="Школа",
        route_summary="Остров Канта и Третьяковская галерея",
        price_text=None,
        booking_text="Звоните",
        booking_url=None,
        summary_one_liner="Групповая экскурсия в Третьяковскую галерею с прогулкой по городу.",
        digest_blurb="Групповая экскурсия в Третьяковскую галерею с прогулкой по городу.",
        digest_eligible=False,
        digest_reason="Specific date, price, and group excursion details are present.",
    )

    assert eligible is True
    assert reason == "Specific date, price, and group excursion details are present."


@pytest.mark.asyncio
async def test_resolve_scan_window_uses_bootstrap_horizon_for_first_full_run(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    limit, days_back = await guide_service._resolve_scan_window(db, "full")

    assert limit == guide_service.GUIDE_SCAN_LIMIT_FULL
    assert days_back == guide_service.GUIDE_DAYS_BACK_BOOTSTRAP

    run_id = await start_ops_run(
        db,
        kind="guide_monitoring",
        trigger="manual",
        chat_id=1,
        operator_id=1,
        details={"mode": "full"},
    )
    await finish_ops_run(db, run_id=run_id, status="success", metrics={}, details={"mode": "full"})

    limit, days_back = await guide_service._resolve_scan_window(db, "full")

    assert limit == guide_service.GUIDE_SCAN_LIMIT_FULL
    assert days_back == guide_service.GUIDE_DAYS_BACK_FULL


@pytest.mark.asyncio
async def test_run_guide_monitor_uses_bootstrap_horizon_on_first_full_scan(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    captured: dict[str, int] = {}
    results_path = tmp_path / "guide_excursions_results.json"
    results_path.write_text("{}", encoding="utf-8")

    async def _fake_run_guide_monitor_kaggle(db_obj, *, run_id, mode, limit, days_back, chat_id, status_callback):
        captured["limit"] = int(limit)
        captured["days_back"] = int(days_back)
        return results_path, {"kernel_ref": "zigomaro/guide-excursions-monitor", "status": "complete"}

    async def _fake_import_results_file(db_obj, *, results_path):
        return (
            {
                "sources_scanned": 0,
                "posts_scanned": 0,
                "posts_prefiltered": 0,
                "occurrences_created": 0,
                "occurrences_updated": 0,
                "templates_touched": 0,
                "profiles_touched": 0,
                "past_occurrences_skipped": 0,
                "llm_ok": 0,
                "llm_deferred": 0,
                "llm_error": 0,
                "errors": 0,
                "duration_sec": 0,
            },
            [],
            {"partial": False, "sources": [], "occurrence_changes": []},
        )

    monkeypatch.setattr(guide_service, "run_guide_monitor_kaggle", _fake_run_guide_monitor_kaggle)
    monkeypatch.setattr(guide_service, "_import_results_file", _fake_import_results_file)

    result = await guide_service.run_guide_monitor(
        db,
        bot=None,
        chat_id=None,
        operator_id=1,
        trigger="test",
        mode="full",
        send_progress=False,
    )

    assert result.ops_run_id is not None
    assert captured["limit"] == guide_service.GUIDE_SCAN_LIMIT_FULL
    assert captured["days_back"] == guide_service.GUIDE_DAYS_BACK_BOOTSTRAP


@pytest.mark.asyncio
async def test_import_results_materializes_fact_pack_and_claims(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _fake_enrich(rows):
        assert rows
        occurrence_id = int(rows[0]["occurrence_id"])
        return {
            occurrence_id: {
                "main_hook": "городские сюжеты старого Закхайма",
                "main_hook_confidence": 86,
                "main_hook_evidence": ["старый район", "городские сюжеты"],
                "audience_region_fit_label": "mixed",
                "audience_region_local_score": 72,
                "audience_region_tourist_score": 68,
                "audience_region_confidence": 74,
                "audience_region_evidence": ["старый район", "городские сюжеты"],
                "audience_region_ambiguity": "Подходит и тем, кто уже знает город, и тем, кто только знакомится.",
            }
        }

    async def _fake_profile_enrich(rows):
        assert rows
        profile_id = int(rows[0]["profile_id"])
        return {
            profile_id: {
                "display_name": "Татьяна Удовенко",
                "guide_line": "Татьяна Удовенко, автор прогулок по Калининграду",
                "summary_short": "Авторские прогулки по Калининграду и его старым районам.",
                "credentials": ["аккредитованный гид"],
                "expertise_tags": ["городская история", "районы Калининграда"],
                "confidence": 84,
            }
        }

    monkeypatch.setattr(guide_service, "apply_occurrence_enrichment", _fake_enrich)
    monkeypatch.setattr(guide_service, "apply_profile_enrichment", _fake_profile_enrich)

    results_path = tmp_path / "guide_excursions_results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "guide-test-1",
                "scan_mode": "full",
                "started_at": "2026-03-15T09:00:00+00:00",
                "finished_at": "2026-03-15T09:05:00+00:00",
                "partial": False,
                "sources": [
                    {
                        "username": "tanja_from_koenigsberg",
                        "source_title": "Татьяна из Кёнигсберга",
                        "source_kind": "guide_personal",
                        "source_status": "ok",
                        "about_text": "Запись: https://t.me/tanja",
                        "about_links": ["https://t.me/tanja"],
                        "posts": [
                            {
                                "message_id": 3895,
                                "post_date": "2026-03-15T09:01:00+00:00",
                                "source_url": "https://t.me/tanja_from_koenigsberg/3895",
                                "text": '16 марта в 11:00 "У Тани на районе: Закхайм и окрестности"',
                                "views": 1200,
                                "reactions_total": 87,
                                "media_refs": [{"message_id": 3895, "kind": "photo"}],
                                "prefilter_passed": True,
                                "llm_status": "ok",
                                "screen": {
                                    "decision": "announce",
                                    "post_kind": "announce_single",
                                    "extract_mode": "announce",
                                },
                                "occurrences": [
                                    {
                                        "canonical_title": "У Тани на районе: Закхайм и окрестности",
                                        "title_normalized": "у тани на районе закхайм и окрестности",
                                        "date": "2026-03-16",
                                        "time": "11:00",
                                        "city": "Калининград",
                                        "meeting_point": "у Закхаймских ворот",
                                        "audience_fit": ["местным", "туристам"],
                                        "price_text": "2000 ₽",
                                        "booking_text": "@tanja",
                                        "booking_url": "https://t.me/tanja",
                                        "channel_url": "https://t.me/tanja_from_koenigsberg/3895",
                                        "status": "scheduled",
                                        "summary_one_liner": "Прогулка по старому району с городскими сюжетами.",
                                        "digest_blurb": "Прогулка по старому району с городскими сюжетами.",
                                        "digest_eligible": True,
                                        "is_last_call": False,
                                        "post_kind": "announce_single",
                                        "availability_mode": "scheduled_public",
                                        "guide_names": ["Татьяна Удовенко"],
                                        "organizer_names": ["Татьяна из Кёнигсберга"],
                                        "fact_pack": {
                                            "canonical_title": "У Тани на районе: Закхайм и окрестности",
                                            "date": "2026-03-16",
                                            "time": "11:00",
                                            "meeting_point": "у Закхаймских ворот",
                                            "booking_url": "https://t.me/tanja",
                                        },
                                        "fact_claims": [
                                            {
                                                "fact_key": "date",
                                                "fact_value": "2026-03-16",
                                                "claim_role": "anchor",
                                                "confidence": 0.95,
                                                "fact_refs": ["T1"],
                                            },
                                            {
                                                "fact_key": "meeting_point",
                                                "fact_value": "у Закхаймских ворот",
                                                "claim_role": "support",
                                                "confidence": 0.8,
                                                "fact_refs": ["T1"],
                                            },
                                        ],
                                        "template_hint": {
                                            "canonical_title": "У Тани на районе: Закхайм и окрестности",
                                            "aliases": ["Закхайм и окрестности"],
                                            "summary_short": "Городской маршрут по старому району.",
                                            "facts_rollup": {"route_anchor": "Закхайм"},
                                        },
                                        "profile_hint": {
                                            "summary_short": "Авторские прогулки по Калининграду.",
                                            "audience_strengths": ["местные", "туристы"],
                                            "source_links": ["https://t.me/tanja"],
                                            "facts_rollup": {"specialty": "urban history"},
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics, errors, summary = await guide_service._import_results_file(db, results_path=str(results_path))

    assert errors == []
    assert metrics["sources_scanned"] == 1
    assert metrics["occurrences_created"] == 1
    assert metrics["profiles_touched"] >= 1
    assert metrics["templates_touched"] == 1
    assert summary["run_id"] == "guide-test-1"

    async with db.raw_conn() as conn:
        await conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = None
        cur = await conn.execute(
            "SELECT fact_pack_json, booking_url FROM guide_occurrence WHERE canonical_title=?",
            ("У Тани на районе: Закхайм и окрестности",),
        )
        row = await cur.fetchone()
        assert row is not None
        fact_pack = json.loads(row[0])
        assert fact_pack["meeting_point"] == "у Закхаймских ворот"
        assert fact_pack["main_hook"] == "городские сюжеты старого Закхайма"
        assert fact_pack["audience_region_fit_label"] == "mixed"
        assert row[1] == "https://t.me/tanja"

        cur = await conn.execute(
            "SELECT fact_key, claim_role FROM guide_fact_claim WHERE entity_kind='occurrence' ORDER BY fact_key"
        )
        occurrence_claims = await cur.fetchall()
        assert ("audience_region_fit_label", "audience_region_fit") in occurrence_claims
        assert ("date", "anchor") in occurrence_claims
        assert ("main_hook", "enrich_hook") in occurrence_claims
        assert ("meeting_point", "support") in occurrence_claims

        cur = await conn.execute(
            "SELECT display_name, summary_short, facts_rollup_json FROM guide_profile WHERE slug='tatyana-udovenko'"
        )
        profile_row = await cur.fetchone()
        assert profile_row is not None
        assert profile_row[0] == "Татьяна Удовенко"
        assert "Авторские прогулки" in profile_row[1]
        profile_facts = json.loads(profile_row[2])
        assert profile_facts["guide_line"] == "Татьяна Удовенко, автор прогулок по Калининграду"
        assert "аккредитованный гид" in profile_facts["credentials"]

    facts_text = await guide_service.render_guide_occurrence_facts(db, 1)
    assert "Materialized fact pack" in facts_text
    assert "meeting_point" in facts_text
    assert "[anchor] date = 2026-03-16" in facts_text
    assert "/guide_log 1" in facts_text
    assert "@tanja_from_koenigsberg/3895" in facts_text

    log_text = await guide_service.render_guide_occurrence_log(db, 1)
    assert "Guide source log #1" in log_text
    assert "@tanja_from_koenigsberg/3895" in log_text
    assert "date = 2026-03-16" in log_text

    ops_run_id = await start_ops_run(
        db,
        kind="guide_monitoring",
        trigger="manual",
        details={
            "mode": "full",
            "run_id": "guide-test-1",
            "transport": "kaggle",
            "source_reports": summary["sources"],
            "occurrence_changes": summary["occurrence_changes"],
        },
    )
    await finish_ops_run(
        db,
        run_id=ops_run_id,
        status="success",
        metrics=metrics,
        details={
            "mode": "full",
            "run_id": "guide-test-1",
            "transport": "kaggle",
            "source_reports": summary["sources"],
            "occurrence_changes": summary["occurrence_changes"],
        },
    )

    report_chunks = await guide_service.render_guide_run_report(db, ops_run_id)
    report_text = "\n".join(report_chunks)
    assert f"ops_run_id={ops_run_id}" in report_text
    assert "@tanja_from_koenigsberg" in report_text
    assert "#1" in report_text

    runs_chunks = await guide_service.render_guide_runs_summary(db, hours=48)
    runs_text = "\n".join(runs_chunks)
    assert f"/guide_report {ops_run_id}" in runs_text
    assert "kaggle/full" in runs_text

    future_text, future_markup = await guide_service.build_guide_future_occurrences_message(db, page=1)
    assert "Будущие экскурсии гидов" in future_text
    assert "1. Пн, 16 марта, 11:00" in future_text
    assert future_markup is not None
    assert any(
        button.callback_data == "guide:occdel:1:1"
        for row in future_markup.inline_keyboard
        for button in row
    )

    templates_text, templates_markup = await guide_service.build_guide_templates_message(db, page=1)
    assert "Типовые экскурсии" in templates_text
    assert "1. У Тани на районе: Закхайм и окрестности" in templates_text
    assert templates_markup is not None
    assert any(
        button.callback_data == "guide:tplshow:1"
        for row in templates_markup.inline_keyboard
        for button in row
    )
    assert any(
        button.callback_data == "guide:tpldel:1:1"
        for row in templates_markup.inline_keyboard
        for button in row
    )

    template_detail = await guide_service.render_guide_template_detail(db, template_id=1)
    assert "Guide template #1" in template_detail
    assert "Route anchor: Закхайм" in template_detail or "Типовые маршруты:" in template_detail
    assert "Главные фишки: городские сюжеты старого Закхайма" in template_detail
    assert "Locals/tourists/mixed: mixed=1" in template_detail

    now_utc = datetime.now(timezone.utc)
    created_recent = (now_utc - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    updated_recent = (now_utc - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    old_seen = (now_utc - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")

    async with db.raw_conn() as conn:
        await conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = None
        await conn.execute(
            "UPDATE guide_occurrence SET first_seen_at=?, updated_at=?, last_seen_post_at=? WHERE id=1",
            (created_recent, created_recent, created_recent),
        )
        cur = await conn.execute(
            "SELECT template_id, primary_source_id FROM guide_occurrence WHERE id=1"
        )
        tpl_row = await cur.fetchone()
        assert tpl_row is not None
        cur = await conn.execute("SELECT id FROM guide_monitor_post WHERE message_id=3895")
        post_row = await cur.fetchone()
        assert post_row is not None
        await conn.execute(
            """
            INSERT INTO guide_occurrence(
                id,
                template_id,
                primary_source_id,
                primary_message_id,
                source_fingerprint,
                canonical_title,
                title_normalized,
                digest_eligible,
                date,
                time,
                    summary_one_liner,
                    first_seen_at,
                    updated_at,
                    last_seen_post_at
                )
                VALUES(2, ?, ?, 3895, 'fp-guide-updated', 'Повторная экскурсия по Закхайму', 'повторная экскурсия по закхайму', 1, '2026-03-18', '12:00', 'Обновлённая карточка', ?, ?, ?)
                """,
                (tpl_row[0], tpl_row[1], old_seen, updated_recent, updated_recent),
            )
        await conn.execute(
            "INSERT INTO guide_occurrence_source(occurrence_id, post_id, role) VALUES(2, ?, 'primary')",
            (post_row[0],),
        )
        await conn.commit()

    recent_changes = "\n".join(await guide_service.render_guide_recent_changes(db, hours=120))
    assert "Новые:" in recent_changes
    assert "Обновлённые:" in recent_changes
    assert "#1 Пн, 16 марта, 11:00" in recent_changes
    assert "#2 Ср, 18 марта, 12:00" in recent_changes

    delete_occurrence = await guide_service.delete_guide_occurrence(db, 2)
    assert delete_occurrence["deleted"] is True
    assert delete_occurrence["occurrence_id"] == 2

    delete_template = await guide_service.delete_guide_template(db, 1)
    assert delete_template["deleted"] is True
    assert delete_template["template_id"] == 1

    templates_after_delete, _ = await guide_service.build_guide_templates_message(db, page=1)
    assert "Шаблонов пока нет." in templates_after_delete


@pytest.mark.asyncio
async def test_import_results_skips_past_occurrence(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    results_path = tmp_path / "guide_excursions_results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "guide-test-past",
                "scan_mode": "full",
                "started_at": "2026-03-15T09:00:00+00:00",
                "finished_at": "2026-03-15T09:05:00+00:00",
                "partial": False,
                "sources": [
                    {
                        "username": "gid_zelenogradsk",
                        "source_title": "Гид Зеленоградск",
                        "source_kind": "guide_personal",
                        "source_status": "ok",
                        "posts": [
                            {
                                "message_id": 2508,
                                "post_date": "2026-03-15T09:01:00+00:00",
                                "source_url": "https://t.me/gid_zelenogradsk/2508",
                                "text": "Старая прогулка",
                                "prefilter_passed": True,
                                "llm_status": "ok",
                                "screen": {"decision": "announce", "post_kind": "announce_single", "extract_mode": "announce"},
                                "occurrences": [
                                    {
                                        "canonical_title": "Старая прогулка",
                                        "title_normalized": "старая прогулка",
                                        "date": "2020-01-01",
                                        "time": "10:00",
                                        "digest_eligible": False,
                                        "post_kind": "announce_single",
                                        "availability_mode": "scheduled_public",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics, errors, _summary = await guide_service._import_results_file(db, results_path=str(results_path))

    assert errors == []
    assert metrics["past_occurrences_skipped"] == 1
    assert metrics["occurrences_created"] == 0

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM guide_occurrence")
        row = await cur.fetchone()
        assert row[0] == 0


@pytest.mark.asyncio
async def test_import_results_normalizes_live_fact_claim_shapes(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    results_path = tmp_path / "guide_excursions_results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "guide-test-live-claims",
                "scan_mode": "light",
                "started_at": "2026-03-15T09:00:00+00:00",
                "finished_at": "2026-03-15T09:05:00+00:00",
                "partial": False,
                "sources": [
                    {
                        "username": "amber_fringilla",
                        "source_title": "Amber Fringilla",
                        "source_kind": "guide_personal",
                        "source_status": "ok",
                        "posts": [
                            {
                                "message_id": 5806,
                                "post_date": "2026-03-15T09:01:00+00:00",
                                "source_url": "https://t.me/amber_fringilla/5806",
                                "text": "22 марта в 9:00 экскурсия в Южном парке",
                                "prefilter_passed": True,
                                "llm_status": "ok",
                                "screen": {
                                    "decision": "announce",
                                    "post_kind": "announce_single",
                                    "extract_mode": "announce",
                                },
                                "occurrences": [
                                    {
                                        "canonical_title": "Экопрогулка в Южном парке",
                                        "title_normalized": "экопрогулка в южном парке",
                                        "date": "2026-03-22",
                                        "time": "09:00",
                                        "city": "Калининград",
                                        "meeting_point": "Главный вход в Южный парк",
                                        "digest_eligible": True,
                                        "post_kind": "announce_single",
                                        "availability_mode": "scheduled_public",
                                        "fact_pack": {
                                            "canonical_title": "Экопрогулка в Южном парке",
                                            "date": "2026-03-22",
                                            "time": "09:00",
                                            "meeting_point": "Главный вход в Южный парк",
                                        },
                                        "fact_claims": [
                                            {
                                                "claim_role": "anchor",
                                                "fact_type": "date",
                                                "fact_value": "2026-03-22",
                                            },
                                            {
                                                "claim_role": "support",
                                                "claim_text": "Главный вход в Южный парк",
                                            },
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics, errors, _summary = await guide_service._import_results_file(db, results_path=str(results_path))

    assert errors == []
    assert metrics["occurrences_created"] == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT fact_key, fact_value, claim_role
            FROM guide_fact_claim
            WHERE entity_kind='occurrence'
            ORDER BY fact_key
            """
        )
        rows = [tuple(row) for row in await cur.fetchall()]
    assert ("claim_text", "Главный вход в Южный парк", "support") in rows
    assert ("date", "2026-03-22", "anchor") in rows


@pytest.mark.asyncio
async def test_import_results_normalizes_bare_fact_claim_text_shape(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    results_path = tmp_path / "guide_excursions_results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "guide-test-bare-fact",
                "scan_mode": "light",
                "started_at": "2026-03-15T09:00:00+00:00",
                "finished_at": "2026-03-15T09:05:00+00:00",
                "partial": False,
                "sources": [
                    {
                        "username": "ruin_keepers",
                        "source_title": "Ruin Keepers",
                        "source_kind": "organization_with_tours",
                        "source_status": "ok",
                        "posts": [
                            {
                                "message_id": 5065,
                                "post_date": "2026-03-15T09:01:00+00:00",
                                "source_url": "https://t.me/ruin_keepers/5065",
                                "text": "25 марта в 10:00 экскурсия по Амалиенау",
                                "prefilter_passed": True,
                                "llm_status": "ok",
                                "screen": {
                                    "decision": "announce",
                                    "post_kind": "announce_single",
                                    "extract_mode": "announce",
                                },
                                "occurrences": [
                                    {
                                        "canonical_title": "Южный Амалиенау. История района в судьбах людей",
                                        "title_normalized": "южный амалиенау история района в судьбах людей",
                                        "date": "2026-03-25",
                                        "time": "10:00",
                                        "digest_eligible": True,
                                        "post_kind": "announce_single",
                                        "availability_mode": "scheduled_public",
                                        "fact_claims": [
                                            {
                                                "claim_role": "anchor",
                                                "fact": "Экскурсия состоится 25 марта.",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics, errors, _summary = await guide_service._import_results_file(db, results_path=str(results_path))

    assert errors == []
    assert metrics["occurrences_created"] == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT fact_key, fact_value, claim_role
            FROM guide_fact_claim
            WHERE entity_kind='occurrence'
            ORDER BY fact_key
            """
        )
        rows = [tuple(row) for row in await cur.fetchall()]
    assert ("claim_text", "Экскурсия состоится 25 марта.", "anchor") in rows


@pytest.mark.asyncio
async def test_resume_guide_monitor_jobs_imports_completed_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_registry, "_REGISTRY_PATH", tmp_path / "kaggle_jobs.json")
    await kaggle_registry.register_job(
        "guide_monitoring",
        "owner/kernel",
        meta={"run_id": "guide-run-123", "pid": 999999, "mode": "full", "chat_id": 12345},
    )

    imported: dict[str, object] = {}

    class _DummyClient:
        def get_kernel_status(self, kernel_ref):
            assert kernel_ref == "owner/kernel"
            return {"status": "complete"}

    async def fake_download(_client, kernel_ref, run_id):
        imported["downloaded"] = (kernel_ref, run_id)
        path = tmp_path / "guide_excursions_results.json"
        path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        return path

    async def fake_import_from_results(*_args, **kwargs):
        imported["import"] = kwargs
        return guide_service.GuideMonitorResult(
            run_id=str(kwargs.get("run_id") or ""),
            ops_run_id=321,
            trigger=str(kwargs.get("trigger") or ""),
            mode=str(kwargs.get("mode") or ""),
            metrics={},
            errors=[],
        )

    monkeypatch.setattr(guide_service, "KaggleClient", lambda: _DummyClient())
    monkeypatch.setattr(guide_service, "download_guide_results", fake_download)
    monkeypatch.setattr(guide_service, "run_guide_import_from_results", fake_import_from_results)

    recovered = await guide_service.resume_guide_monitor_jobs(db=None, bot=None, chat_id=12345)

    assert recovered == 1
    assert imported["downloaded"] == ("owner/kernel", "guide-run-123")
    assert imported["import"]["trigger"] == "recovery_import"
    assert imported["import"]["transport"] == "kaggle_recovery"
    assert imported["import"]["chat_id"] == 12345
    assert await kaggle_registry.list_jobs("guide_monitoring") == []
