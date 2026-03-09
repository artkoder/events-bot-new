from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from db import Database
from general_stats import collect_general_stats, format_general_stats_message


def _utc_sql(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@pytest.mark.asyncio
async def test_collect_general_stats_aggregates_metrics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    tz = ZoneInfo("Europe/Kaliningrad")
    now_local = datetime(2026, 2, 16, 7, 30, tzinfo=tz)
    start_utc = (now_local - timedelta(hours=24)).astimezone(timezone.utc)
    end_utc = now_local.astimezone(timezone.utc)

    in_window_1 = _utc_sql(start_utc + timedelta(minutes=30))
    in_window_2 = _utc_sql(start_utc + timedelta(hours=20))
    out_before = _utc_sql(start_utc - timedelta(minutes=10))

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO telegram_source(id, username, enabled) VALUES(1001, 'src1_test', 1)"
        )
        await conn.execute(
            "INSERT INTO telegram_source(id, username, enabled) VALUES(1002, 'src2_test', 1)"
        )

        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, has_date, status, created_at) "
            "VALUES(1, 1, 11, 0, 'a', 1, 'pending', ?)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, has_date, status, created_at) "
            "VALUES(2, 1, 12, 0, 'b', 1, 'pending', ?)",
            (in_window_2,),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, has_date, status, created_at) "
            "VALUES(3, 1, 13, 0, 'c', 1, 'imported', ?)",
            (out_before,),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, has_date, status, created_at) "
            "VALUES(4, 1, 14, 0, 'd', 1, 'rejected', ?)",
            (out_before,),
        )

        await conn.execute(
            "INSERT INTO vk_inbox_import_event(inbox_id, event_id, created_at) VALUES(1, 101, ?)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO vk_inbox_import_event(inbox_id, event_id, created_at) VALUES(1, 102, ?)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO vk_inbox_import_event(inbox_id, event_id, created_at) VALUES(2, 103, ?)",
            (in_window_2,),
        )
        await conn.execute(
            "INSERT INTO vk_inbox_import_event(inbox_id, event_id, created_at) VALUES(3, 104, ?)",
            (out_before,),
        )

        await conn.execute(
            "INSERT INTO telegram_scanned_message(source_id, message_id, processed_at, status, events_extracted, events_imported) "
            "VALUES(1001, 10, ?, 'done', 1, 0)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO telegram_scanned_message(source_id, message_id, processed_at, status, events_extracted, events_imported) "
            "VALUES(1001, 11, ?, 'done', 0, 0)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO telegram_scanned_message(source_id, message_id, processed_at, status, events_extracted, events_imported) "
            "VALUES(1002, 20, ?, 'done', 0, 2)",
            (in_window_2,),
        )

        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, source_text, added_at) "
            "VALUES(1, 'Event1', 'd', '2026-02-20', '18:00', 'loc', 'src', ?)",
            (out_before,),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, source_text, added_at) "
            "VALUES(2, 'Event2', 'd', '2026-02-20', '19:00', 'loc', 'src', ?)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, source_text, added_at) "
            "VALUES(3, 'Event3', 'd', '2026-02-21', '20:00', 'loc', 'src', ?)",
            (in_window_2,),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, source_text, added_at) "
            "VALUES(4, 'Event4', 'd', '2026-02-22', '21:00', 'loc', 'src', ?)",
            (out_before,),
        )

        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(1, 1, 'vk', 'u1', ?)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(2, 1, 'vk', 'u2', ?)",
            (out_before,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(3, 4, 'tg', 'u3', ?)",
            (in_window_2,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(4, 4, 'tg', 'u4', ?)",
            (out_before,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(5, 4, 'tg', 'u5', ?)",
            (out_before,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(6, 2, 'parser:dramteatr', 'p1', ?)",
            (in_window_1,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(7, 3, 'parser:qtickets', 'p2', ?)",
            (in_window_2,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(8, 3, 'telegram', 'tg-created', ?)",
            (in_window_2,),
        )

        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, source_text, added_at) "
            "VALUES(5, 'PastEvent', 'd', '2026-02-10', '18:00', 'loc', 'src', ?)",
            (out_before,),
        )
        await conn.execute(
            "INSERT INTO event_source(id, event_id, source_type, source_url, imported_at) "
            "VALUES(9, 5, 'vk', 'past-vk', ?)",
            (out_before,),
        )

        await conn.execute(
            "INSERT INTO geo_city_region_cache(city_norm, details, created_at, updated_at) VALUES('newcity', ?, ?, ?)",
            (json.dumps({"raw": "Новый город"}), in_window_1, in_window_1),
        )
        await conn.execute(
            "INSERT INTO geo_city_region_cache(city_norm, details, created_at, updated_at) VALUES('oldcity', ?, ?, ?)",
            (json.dumps({"raw": "Старый"}), out_before, out_before),
        )

        await conn.execute(
            "INSERT INTO festival_queue(id, status, source_kind, source_url, created_at, updated_at, next_run_at) "
            "VALUES(1, 'pending', 'vk', 'https://vk.com/a', ?, ?, ?)",
            (in_window_1, in_window_1, in_window_1),
        )
        await conn.execute(
            "INSERT INTO festival_queue(id, status, source_kind, source_url, created_at, updated_at, next_run_at) "
            "VALUES(2, 'pending', 'vk', 'https://vk.com/b', ?, ?, ?)",
            (out_before, out_before, out_before),
        )
        await conn.execute(
            "INSERT INTO festival_queue(id, status, source_kind, source_url, created_at, updated_at, next_run_at) "
            "VALUES(3, 'running', 'tg', 'https://t.me/c/1/2', ?, ?, ?)",
            (out_before, out_before, out_before),
        )
        await conn.execute(
            "INSERT INTO festival_queue(id, status, source_kind, source_url, created_at, updated_at, next_run_at) "
            "VALUES(4, 'pending', 'url', 'https://fest.example.com/program', ?, ?, ?)",
            (out_before, out_before, out_before),
        )
        await conn.execute(
            "INSERT INTO festival_queue(id, status, source_kind, source_url, created_at, updated_at, next_run_at) "
            "VALUES(5, 'error', 'tg', 'https://t.me/c/1/99', ?, ?, ?)",
            (out_before, out_before, out_before),
        )
        await conn.execute(
            "INSERT INTO festival_queue(id, status, source_kind, source_url, created_at, updated_at, next_run_at) "
            "VALUES(6, 'done', 'url', 'https://done.example.com', ?, ?, ?)",
            (out_before, out_before, out_before),
        )

        await conn.execute(
            "INSERT INTO festival(id, name, created_at, updated_at) VALUES(1, 'Fest1', ?, ?)",
            (in_window_1, in_window_1),
        )
        await conn.execute(
            "INSERT INTO festival(id, name, created_at, updated_at) VALUES(2, 'Fest2', ?, ?)",
            (out_before, in_window_2),
        )

        await conn.execute(
            "INSERT INTO ops_run(kind, trigger, started_at, finished_at, status, metrics_json, details_json) "
            "VALUES('vk_auto_import', 'scheduled', ?, ?, 'success', ?, '{}')",
            (
                in_window_1,
                in_window_1,
                json.dumps(
                    {"inbox_processed": 0, "inbox_imported": 0, "inbox_rejected": 0},
                    ensure_ascii=False,
                ),
            ),
        )
        await conn.execute(
            "INSERT INTO ops_run(kind, trigger, started_at, finished_at, status, metrics_json, details_json) "
            "VALUES('parse', 'manual', ?, ?, 'success', ?, ?)",
            (
                in_window_1,
                in_window_1,
                json.dumps({"total_events": 2, "events_created": 0, "events_updated": 0}, ensure_ascii=False),
                json.dumps(
                    {
                        "sources": {
                            "dramteatr": {"processed": 4, "new_events": 2, "updated_events": 1},
                            "qtickets": {"processed": 1, "new_events": 0, "updated_events": 2},
                        }
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        await conn.execute(
            "INSERT INTO ops_run(kind, trigger, started_at, finished_at, status, metrics_json, details_json) "
            "VALUES('3di', 'scheduled', ?, ?, 'success', ?, '{}')",
            (
                in_window_2,
                in_window_2,
                json.dumps(
                    {"events_considered": 5, "previews_rendered": 2, "preview_errors": 1, "preview_skipped": 2},
                    ensure_ascii=False,
                ),
            ),
        )
        await conn.execute(
            "INSERT INTO ops_run(kind, trigger, started_at, finished_at, status, metrics_json, details_json) "
            "VALUES('festival_queue', 'manual', ?, ?, 'success', ?, '{}')",
            (
                in_window_2,
                in_window_2,
                json.dumps({"processed": 1, "success": 1, "failed": 0}, ensure_ascii=False),
            ),
        )
        await conn.execute(
            "INSERT INTO ops_run(kind, trigger, started_at, finished_at, status, metrics_json, details_json) "
            "VALUES('tg_monitoring', 'scheduled', ?, ?, 'success', ?, '{}')",
            (
                in_window_2,
                in_window_2,
                json.dumps(
                    {
                        "sources_scanned": 2,
                        "messages_processed": 3,
                        "messages_with_events": 2,
                        "events_created": 1,
                        "events_merged": 2,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        await conn.commit()

    snapshot = await collect_general_stats(
        db,
        tz_name="Europe/Kaliningrad",
        now=now_local,
    )
    metrics = snapshot.metrics

    assert metrics["vk"]["vk_posts_added"] == 2
    assert metrics["vk"]["vk_posts_auto_imported"] == 2
    assert metrics["vk"]["vk_events_from_auto_import"] == 3
    assert metrics["vk"]["vk_queue_added_period"] == 2
    assert metrics["vk"]["vk_queue_parsed_period"] == 2
    assert metrics["vk"]["vk_queue_unresolved_now"] == 2
    assert len(metrics["vk"]["vk_auto_import_runs"]) == 1

    assert metrics["telegram"]["sources_scanned"] == 2
    assert metrics["telegram"]["messages_with_events"] == 2
    assert metrics["telegram"]["sources_with_events"] == 2
    assert metrics["telegram"]["events_created"] == 1
    assert metrics["telegram"]["events_updated"] == 2
    assert len(metrics["telegram"]["tg_monitoring_runs"]) == 1

    assert metrics["events"]["events_created"] == 2
    assert metrics["events"]["events_updated"] == 2
    assert metrics["events"]["updated_sources_distribution"] == {2: 1, 3: 1}
    source_share = metrics["events"]["source_share"]
    assert source_share["period_total_events"] == 4
    assert source_share["period_by_source"]["parse"]["count"] == 2
    assert source_share["period_by_source"]["parse"]["percent"] == 50
    assert source_share["period_by_source"]["telegram"]["count"] == 2
    assert source_share["period_by_source"]["telegram"]["percent"] == 50
    assert source_share["period_by_source"]["vk"]["count"] == 1
    assert source_share["period_by_source"]["vk"]["percent"] == 25
    assert source_share["future_total_events"] == 4
    assert source_share["future_by_source"]["parse"]["count"] == 2
    assert source_share["future_by_source"]["telegram"]["count"] == 2
    assert source_share["future_by_source"]["vk"]["count"] == 1

    assert metrics["geo"]["new_cities_count"] == 1
    assert metrics["geo"]["new_cities"][0]["city_norm"] == "newcity"

    assert metrics["festivals"]["festival_queue_added"] == 1
    assert metrics["festivals"]["festival_queue_total_now"] == 6
    assert metrics["festivals"]["festival_queue_pending_now"] == 3
    assert metrics["festivals"]["festival_queue_running_now"] == 1
    assert metrics["festivals"]["festival_queue_done_now"] == 1
    assert metrics["festivals"]["festival_queue_error_now"] == 1
    assert metrics["festivals"]["festival_queue_active_now"] == 4
    assert metrics["festivals"]["festival_queue_active_by_source_now"] == {"tg": 1, "url": 1, "vk": 2}
    assert metrics["festivals"]["festivals_created"] == 1
    assert metrics["festivals"]["festivals_updated"] == 2
    assert len(metrics["festivals"]["festival_queue_runs"]) == 1

    parse_breakdown = metrics["parse"]["source_breakdown"]
    assert parse_breakdown["dramteatr"]["processed"] == 4
    assert parse_breakdown["dramteatr"]["new_events"] == 2
    assert parse_breakdown["dramteatr"]["updated_events"] == 1
    assert parse_breakdown["qtickets"]["processed"] == 1
    assert metrics["parse"]["runs"][0]["metrics"]["events_created"] == 2
    assert metrics["parse"]["runs"][0]["metrics"]["events_updated"] == 3

    text = format_general_stats_message(snapshot)
    assert "events_created: 1" in text
    assert "events_updated: 2" in text
    assert "events_created=2 events_updated=3" in text
    assert "dramteatr: processed=4 new=2 updated=1" in text
    assert "source_share_period: total_events=4" in text
    assert "/parse: 50% (2/4)" in text
    assert "telegram: 50% (2/4)" in text
    assert "vk: 25% (1/4)" in text
    assert "source_share_future_active: total_events=4" in text
    assert "source_share_note: проценты могут пересекаться" in text
    assert "festival_queue_status_now: total=6 pending=3 running=1 done=1 error=1" in text
    assert "festival_queue_active_now: 4" in text
    assert "festival_queue_active_by_source_now: vk=2 tg=1 url=1" in text


@pytest.mark.asyncio
async def test_collect_general_stats_uses_half_open_window(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    tz = ZoneInfo("Europe/Kaliningrad")
    now_local = datetime(2026, 2, 16, 7, 30, tzinfo=tz)
    start_utc = (now_local - timedelta(hours=24)).astimezone(timezone.utc)
    end_utc = now_local.astimezone(timezone.utc)

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, has_date, status, created_at) "
            "VALUES(1, 1, 1, 0, 'start_included', 1, 'pending', ?)",
            (_utc_sql(start_utc),),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, has_date, status, created_at) "
            "VALUES(2, 1, 2, 0, 'end_excluded', 1, 'pending', ?)",
            (_utc_sql(end_utc),),
        )
        await conn.commit()

    snapshot = await collect_general_stats(db, tz_name="Europe/Kaliningrad", now=now_local)
    assert snapshot.metrics["vk"]["vk_posts_added"] == 1


class _FakeSupabaseResponse:
    def __init__(self, *, count: int | None = None, data: list[dict[str, object]] | None = None) -> None:
        self.count = count
        self.data = data or []


class _FakeSupabaseQuery:
    def __init__(self, *, client: "_FakeSupabaseClient", table: str) -> None:
        self._client = client
        self._table = table
        self._gte: dict[str, str] = {}
        self._lt: dict[str, str] = {}
        self._like: dict[str, str] = {}
        self._range: tuple[int, int] | None = None

    def select(self, _cols: str, *, count: str | None = None, head: bool | None = None):  # noqa: ANN001
        self._select_count = count
        self._head = head
        return self

    def gte(self, col: str, value: str):
        self._gte[col] = value
        return self

    def lt(self, col: str, value: str):
        self._lt[col] = value
        return self

    def like(self, col: str, pattern: str):
        self._like[col] = pattern
        return self

    def range(self, start: int, end: int):
        self._range = (int(start), int(end))
        return self

    @staticmethod
    def _parse_iso(raw: str) -> datetime:
        txt = (raw or "").strip()
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt)

    def _matches(self, row: dict[str, object]) -> bool:
        for col, raw in self._gte.items():
            if col not in row:
                return False
            if self._parse_iso(str(row[col])) < self._parse_iso(raw):
                return False
        for col, raw in self._lt.items():
            if col not in row:
                return False
            if self._parse_iso(str(row[col])) >= self._parse_iso(raw):
                return False
        for col, pat in self._like.items():
            if col not in row:
                return False
            # Only supports "prefix%" patterns used by the code.
            if pat.endswith("%"):
                prefix = pat[:-1]
                if not str(row[col]).startswith(prefix):
                    return False
            elif str(row[col]) != pat:
                return False
        return True

    def execute(self) -> _FakeSupabaseResponse:
        if self._table == "google_ai_requests" and self._client.fail_google_ai_requests:
            raise RuntimeError("table not available")
        if self._table == "google_ai_usage_counters" and self._client.fail_google_ai_usage_counters:
            raise RuntimeError("table not available")
        if self._table == "token_usage" and self._client.fail_token_usage:
            raise RuntimeError("table not available")
        rows = list(self._client.data.get(self._table, []))
        rows = [row for row in rows if self._matches(row)]
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        return _FakeSupabaseResponse(count=len(rows), data=[] if getattr(self, "_head", False) else rows)


class _FakeSupabaseClient:
    def __init__(
        self,
        *,
        data: dict[str, list[dict[str, object]]] | None = None,
        fail_google_ai_requests: bool = False,
        fail_google_ai_usage_counters: bool = False,
        fail_token_usage: bool = False,
    ) -> None:
        self.data = data or {}
        self.fail_google_ai_requests = fail_google_ai_requests
        self.fail_google_ai_usage_counters = fail_google_ai_usage_counters
        self.fail_token_usage = fail_token_usage

    def table(self, name: str) -> _FakeSupabaseQuery:
        return _FakeSupabaseQuery(client=self, table=name)

    def schema(self, _schema: str):  # noqa: ANN001
        return self


@pytest.mark.asyncio
async def test_collect_gemma_requests_count_filters_by_model_and_window():
    from general_stats import _collect_gemma_requests_count

    start_utc = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)
    end_utc = datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc)

    supabase = _FakeSupabaseClient(
        data={
            "google_ai_requests": [
                {"request_uid": "1", "created_at": "2026-02-23T01:00:00+00:00", "model": "gemma-3-27b"},
                {"request_uid": "2", "created_at": "2026-02-23T02:00:00+00:00", "model": "gemini-2.5-flash"},
                {"request_uid": "3", "created_at": "2026-02-22T23:00:00+00:00", "model": "gemma-3-27b"},
            ]
        }
    )

    count = await _collect_gemma_requests_count(
        supabase_client=supabase,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    assert count == 1


@pytest.mark.asyncio
async def test_collect_gemma_requests_count_falls_back_to_usage_counters():
    from general_stats import _collect_gemma_requests_count

    start_utc = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)
    end_utc = datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc)

    supabase = _FakeSupabaseClient(
        fail_google_ai_requests=True,
        data={
            "google_ai_usage_counters": [
                {"minute_bucket": "2026-02-23T01:00:00+00:00", "model": "gemma-3-27b", "rpm_used": 3},
                {"minute_bucket": "2026-02-23T02:00:00+00:00", "model": "gemma-3-27b", "rpm_used": 2},
                {"minute_bucket": "2026-02-23T03:00:00+00:00", "model": "gemini-2.5-flash", "rpm_used": 10},
                {"minute_bucket": "2026-02-22T23:00:00+00:00", "model": "gemma-3-27b", "rpm_used": 7},
            ]
        },
    )

    count = await _collect_gemma_requests_count(
        supabase_client=supabase,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    assert count == 5


@pytest.mark.asyncio
async def test_collect_gemma_requests_count_falls_back_to_token_usage():
    from general_stats import _collect_gemma_requests_count

    start_utc = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)
    end_utc = datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc)

    supabase = _FakeSupabaseClient(
        fail_google_ai_requests=True,
        fail_google_ai_usage_counters=True,
        data={
            "token_usage": [
                {
                    "request_id": "1",
                    "at": "2026-02-23T01:00:00+00:00",
                    "model": "gemma-3-27b",
                    "endpoint": "google_ai.generate_content",
                },
                {
                    "request_id": "2",
                    "at": "2026-02-23T02:00:00+00:00",
                    "model": "gemma-3-27b",
                    "endpoint": "google_ai.generate_content",
                },
                {
                    "request_id": "3",
                    "at": "2026-02-23T03:00:00+00:00",
                    "model": "gpt-4o-mini",
                    "endpoint": "chat.completions",
                },
                {
                    "request_id": "4",
                    "at": "2026-02-22T23:00:00+00:00",
                    "model": "gemma-3-27b",
                    "endpoint": "google_ai.generate_content",
                },
            ]
        },
    )

    count = await _collect_gemma_requests_count(
        supabase_client=supabase,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    assert count == 2
