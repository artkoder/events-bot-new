from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from admin_chat import resolve_superadmin_chat_id
from db import Database
from models import Setting
from ops_run import finish_ops_run, start_ops_run

logger = logging.getLogger(__name__)

DEFAULT_GENERAL_STATS_TZ = "Europe/Kaliningrad"
_MB = 1024 * 1024


@dataclass(slots=True)
class GeneralStatsWindow:
    tz_name: str
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime


@dataclass(slots=True)
class GeneralStatsSnapshot:
    window: GeneralStatsWindow
    metrics: dict[str, Any]


def _safe_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logger.warning("general_stats: invalid timezone=%s; falling back to UTC", tz_name)
        return ZoneInfo("UTC")


def _utc_sql(value: datetime) -> str:
    dt = value.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _local_ts(value: datetime, tz: ZoneInfo) -> str:
    return value.astimezone(tz).strftime("%Y-%m-%d %H:%M")


def _parse_sqlite_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _json_load(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _parse_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _build_window(*, tz_name: str, now: datetime | None = None) -> GeneralStatsWindow:
    tz = _safe_zoneinfo(tz_name)
    if now is None:
        end_local = datetime.now(tz)
    elif now.tzinfo is None:
        end_local = now.replace(tzinfo=tz)
    else:
        end_local = now.astimezone(tz)
    start_local = end_local - timedelta(hours=24)
    return GeneralStatsWindow(
        tz_name=tz.key,
        start_local=start_local,
        end_local=end_local,
        start_utc=start_local.astimezone(timezone.utc),
        end_utc=end_local.astimezone(timezone.utc),
    )


async def _fetch_int(conn, sql: str, params: Sequence[Any]) -> int:
    cur = await conn.execute(sql, tuple(params))
    row = await cur.fetchone()
    return _parse_int(row[0] if row else 0)


async def _fetch_ops_runs(
    db: Database,
    *,
    kind: str,
    start_utc: datetime,
    end_utc: datetime,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    start_raw = _utc_sql(start_utc)
    end_raw = _utc_sql(end_utc)
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT
                trigger,
                chat_id,
                operator_id,
                started_at,
                finished_at,
                status,
                metrics_json,
                details_json
            FROM ops_run
            WHERE kind = ?
              AND datetime(started_at) >= datetime(?)
              AND datetime(started_at) < datetime(?)
            ORDER BY datetime(started_at) ASC, id ASC
            """,
            (kind, start_raw, end_raw),
        )
        rows = await cur.fetchall()
    for row in rows or []:
        started_at = _parse_sqlite_ts(row[3])
        finished_at = _parse_sqlite_ts(row[4])
        rows_out.append(
            {
                "trigger": row[0] or "manual",
                "chat_id": _parse_int(row[1]) if row[1] is not None else None,
                "operator_id": _parse_int(row[2]) if row[2] is not None else None,
                "started_at": started_at,
                "finished_at": finished_at,
                "status": str(row[5] or "unknown"),
                "metrics": _json_load(row[6]),
                "details": _json_load(row[7]),
            }
        )
    return rows_out


def _aggregate_parse_breakdown(parse_runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    breakdown: dict[str, dict[str, int]] = {}
    for run in parse_runs:
        details = run.get("details")
        if not isinstance(details, Mapping):
            continue
        sources = details.get("sources")
        if not isinstance(sources, Mapping):
            continue
        for source, payload in sources.items():
            if not isinstance(payload, Mapping):
                continue
            bucket = breakdown.setdefault(
                str(source),
                {"processed": 0, "new_events": 0, "updated_events": 0, "failed": 0, "skipped": 0},
            )
            bucket["processed"] += _parse_int(
                payload.get("processed", payload.get("total_received", 0))
            )
            bucket["new_events"] += _parse_int(
                payload.get("new_events", payload.get("new_added", 0))
            )
            bucket["updated_events"] += _parse_int(
                payload.get(
                    "updated_events",
                    _parse_int(payload.get("updated", 0)) + _parse_int(payload.get("already_exists", 0)),
                )
            )
            bucket["failed"] += _parse_int(payload.get("failed", 0))
            bucket["skipped"] += _parse_int(payload.get("skipped", 0))
    return breakdown


def _sum_run_metric(runs: Sequence[Mapping[str, Any]], metric_key: str) -> int:
    total = 0
    for run in runs:
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        total += _parse_int(metrics.get(metric_key))
    return int(total)


def _extract_city_example(details: Mapping[str, Any]) -> str | None:
    raw = details.get("raw")
    if not isinstance(raw, str):
        return None
    text = " ".join(raw.split())
    if len(text) > 80:
        text = text[:79].rstrip() + "…"
    return text or None


async def _collect_gemma_requests_count(
    *,
    supabase_client: Any | None,
    start_utc: datetime,
    end_utc: datetime,
) -> int | None:
    if supabase_client is None:
        return None
    start_iso = start_utc.astimezone(timezone.utc).isoformat()
    end_iso = end_utc.astimezone(timezone.utc).isoformat()

    def _get_postgrest_client(schema: str | None) -> Any:
        if schema:
            getter = getattr(supabase_client, "schema", None)
            if callable(getter):
                try:
                    return getter(schema)
                except Exception:
                    pass
        return supabase_client

    def _query_requests(schema: str | None) -> int:
        client = _get_postgrest_client(schema)
        query = (
            client.table("google_ai_requests")
            .select("request_uid", count="exact", head=True)
            .gte("created_at", start_iso)
            .lt("created_at", end_iso)
            .like("model", "gemma-%")
        )
        response = query.execute()
        count = getattr(response, "count", None)
        if count is not None:
            return int(_parse_int(count))
        rows = getattr(response, "data", None) or []
        return int(len(rows))

    def _sum_usage_counters(schema: str | None) -> int:
        client = _get_postgrest_client(schema)
        total = 0
        offset = 0
        limit = 1000
        while True:
            response = (
                client.table("google_ai_usage_counters")
                .select("rpm_used")
                .gte("minute_bucket", start_iso)
                .lt("minute_bucket", end_iso)
                .like("model", "gemma-%")
                .range(offset, offset + limit - 1)
                .execute()
            )
            rows = getattr(response, "data", None) or []
            for row in rows:
                if isinstance(row, Mapping):
                    total += _parse_int(row.get("rpm_used"))
            if len(rows) < limit:
                break
            offset += limit
        return int(total)

    def _query_token_usage(schema: str | None) -> int:
        """Fallback counter using `token_usage` if Google AI audit tables are missing.

        `token_usage` is used by `main.log_token_usage` and includes Gemma parse calls
        (`endpoint=google_ai.generate_content`), so it's still useful for daily ops
        reporting when `google_ai_requests`/`google_ai_usage_counters` aren't available.
        """

        client = _get_postgrest_client(schema)

        def _attempt(*, time_col: str, with_endpoint_filter: bool) -> int:
            query = (
                client.table("token_usage")
                .select("request_id", count="exact", head=True)
                .gte(time_col, start_iso)
                .lt(time_col, end_iso)
                .like("model", "gemma-%")
            )
            if with_endpoint_filter:
                query = query.like("endpoint", "google_ai%")
            response = query.execute()
            count = getattr(response, "count", None)
            if count is not None:
                return int(_parse_int(count))
            rows = getattr(response, "data", None) or []
            return int(len(rows))

        # Prefer the explicit `at` timestamp used by our logger; fall back to `created_at`
        # for older schemas.
        last_exc: Exception | None = None
        for time_col in ("at", "created_at"):
            for with_endpoint_filter in (True, False):
                try:
                    return _attempt(time_col=time_col, with_endpoint_filter=with_endpoint_filter)
                except Exception as exc:
                    last_exc = exc
        if last_exc is not None:
            raise last_exc
        return 0

    schema_candidates = []
    for candidate in (None, os.getenv("SUPABASE_SCHEMA"), "public"):
        value = (candidate or "").strip() if isinstance(candidate, str) else None
        schema_candidates.append(value or None)
    seen: set[str | None] = set()
    schemas: list[str | None] = []
    for schema in schema_candidates:
        if schema in seen:
            continue
        seen.add(schema)
        schemas.append(schema)

    last_error: Exception | None = None
    for schema in schemas:
        try:
            return await asyncio.to_thread(_query_requests, schema)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "general_stats: google_ai_requests count failed schema=%s",
                schema or "<default>",
                exc_info=True,
            )

    for schema in schemas:
        try:
            return await asyncio.to_thread(_sum_usage_counters, schema)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "general_stats: google_ai_usage_counters sum failed schema=%s",
                schema or "<default>",
                exc_info=True,
            )

    for schema in schemas:
        try:
            return await asyncio.to_thread(_query_token_usage, schema)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "general_stats: token_usage gemma count failed schema=%s",
                schema or "<default>",
                exc_info=True,
            )
    if last_error is not None:
        return None
    return 0


def _storage_list(storage: Any, path: str, *, limit: int, offset: int) -> list[dict[str, Any]]:
    options = {"limit": limit, "offset": offset, "sortBy": {"column": "name", "order": "asc"}}
    try:
        rows = storage.list(path=path, options=options)
    except TypeError:
        try:
            rows = storage.list(path, options)
        except TypeError:
            rows = storage.list(path)
    return list(rows or [])


def _compute_bucket_size_bytes(supabase_client: Any, bucket: str) -> int:
    storage = supabase_client.storage.from_(bucket)
    total = 0
    seen_paths: set[str] = set()
    stack = [""]
    while stack:
        path = stack.pop()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        offset = 0
        limit = 1000
        while True:
            rows = _storage_list(storage, path, limit=limit, offset=offset)
            if not rows:
                break
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("name") or "").strip().strip("/")
                if not name:
                    continue
                metadata = row.get("metadata")
                size_val: int | None = None
                if isinstance(metadata, Mapping):
                    size_val = _parse_int(
                        metadata.get("size")
                        or metadata.get("contentLength")
                        or metadata.get("content_length")
                    )
                if not size_val:
                    size_val = _parse_int(row.get("size") or row.get("bytes"))
                if size_val and size_val > 0:
                    total += size_val
                    continue
                is_dir = metadata is None and row.get("id") is None
                if is_dir:
                    child = f"{path.rstrip('/')}/{name}".strip("/")
                    if child and child not in seen_paths:
                        stack.append(child)
            if len(rows) < limit:
                break
            offset += limit
    return int(total)


async def _get_bucket_usage_mb(
    db: Database,
    *,
    supabase_client: Any | None,
    bucket_name: str,
) -> float | None:
    if supabase_client is None:
        return None
    bucket = (bucket_name or "events-ics").strip() or "events-ics"
    ttl_sec = max(300, _parse_int(os.getenv("GENERAL_STATS_BUCKET_USAGE_CACHE_SEC") or 21600))
    cache_key = f"general_stats.bucket_usage.{bucket}"
    now = datetime.now(timezone.utc)

    async with db.get_session() as session:
        cached = await session.get(Setting, cache_key)
    if cached and cached.value:
        try:
            payload = json.loads(cached.value)
            cached_at = _parse_sqlite_ts(payload.get("updated_at"))
            cached_bytes = _parse_int(payload.get("bytes"))
            if cached_at and (now - cached_at).total_seconds() <= ttl_sec and cached_bytes >= 0:
                return round(float(cached_bytes) / _MB, 2)
        except Exception:
            logger.debug("general_stats: invalid bucket cache payload", exc_info=True)

    try:
        total_bytes = await asyncio.to_thread(_compute_bucket_size_bytes, supabase_client, bucket)
    except Exception:
        logger.warning("general_stats: bucket usage collection failed bucket=%s", bucket, exc_info=True)
        return None

    payload = json.dumps({"bytes": int(total_bytes), "updated_at": now.isoformat()}, ensure_ascii=False)
    async with db.get_session() as session:
        row = await session.get(Setting, cache_key)
        if row is None:
            row = Setting(key=cache_key, value=payload)
        else:
            row.value = payload
        session.add(row)
        await session.commit()
    return round(float(total_bytes) / _MB, 2)


async def collect_general_stats(
    db: Database,
    *,
    tz_name: str | None = None,
    now: datetime | None = None,
    supabase_client: Any | None = None,
    bucket_name: str | None = None,
) -> GeneralStatsSnapshot:
    tz_value = (tz_name or os.getenv("GENERAL_STATS_TZ") or DEFAULT_GENERAL_STATS_TZ).strip()
    window = _build_window(tz_name=tz_value, now=now)
    start_raw = _utc_sql(window.start_utc)
    end_raw = _utc_sql(window.end_utc)

    metrics: dict[str, Any] = {}

    async with db.raw_conn() as conn:
        vk_posts_added = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM vk_inbox
            WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        vk_posts_auto_imported = await _fetch_int(
            conn,
            """
            SELECT COUNT(DISTINCT inbox_id) FROM vk_inbox_import_event
            WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        vk_events_from_auto_import = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM vk_inbox_import_event
            WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        vk_queue_unresolved_now = await _fetch_int(
            conn,
            """
            SELECT COUNT(*)
            FROM vk_inbox
            WHERE COALESCE(status, 'pending') NOT IN ('imported', 'rejected')
            """,
            (),
        )

        tg_sources_scanned = await _fetch_int(
            conn,
            """
            SELECT COUNT(DISTINCT source_id) FROM telegram_scanned_message
            WHERE datetime(processed_at) >= datetime(?) AND datetime(processed_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        tg_messages_with_events = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM telegram_scanned_message
            WHERE datetime(processed_at) >= datetime(?) AND datetime(processed_at) < datetime(?)
              AND (COALESCE(events_extracted, 0) > 0 OR COALESCE(events_imported, 0) > 0)
            """,
            (start_raw, end_raw),
        )
        tg_sources_with_events = await _fetch_int(
            conn,
            """
            SELECT COUNT(DISTINCT source_id) FROM telegram_scanned_message
            WHERE datetime(processed_at) >= datetime(?) AND datetime(processed_at) < datetime(?)
              AND (COALESCE(events_extracted, 0) > 0 OR COALESCE(events_imported, 0) > 0)
            """,
            (start_raw, end_raw),
        )
        guide_sources_scanned = await _fetch_int(
            conn,
            """
            SELECT COUNT(DISTINCT source_id) FROM guide_monitor_post
            WHERE datetime(last_scanned_at) >= datetime(?) AND datetime(last_scanned_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        guide_posts_prefiltered = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM guide_monitor_post
            WHERE datetime(last_scanned_at) >= datetime(?) AND datetime(last_scanned_at) < datetime(?)
              AND COALESCE(prefilter_passed, 0) = 1
            """,
            (start_raw, end_raw),
        )
        guide_occurrences_new = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM guide_occurrence
            WHERE datetime(first_seen_at) >= datetime(?) AND datetime(first_seen_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        guide_occurrences_updated = await _fetch_int(
            conn,
            """
            SELECT COUNT(*)
            FROM guide_occurrence
            WHERE datetime(updated_at) >= datetime(?) AND datetime(updated_at) < datetime(?)
              AND datetime(first_seen_at) < datetime(?)
            """,
            (start_raw, end_raw, start_raw),
        )
        guide_occurrences_future_now = await _fetch_int(
            conn,
            """
            SELECT COUNT(*)
            FROM guide_occurrence
            WHERE date IS NOT NULL
              AND date >= ?
            """,
            (window.end_local.date().isoformat(),),
        )
        guide_templates_total = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM guide_template
            """,
            (),
        )
        guide_digest_published = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM guide_digest_issue
            WHERE status='published'
              AND datetime(published_at) >= datetime(?) AND datetime(published_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )

        events_created = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM event
            WHERE datetime(added_at) >= datetime(?) AND datetime(added_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        events_updated = await _fetch_int(
            conn,
            """
            SELECT COUNT(DISTINCT es.event_id)
            FROM event_source es
            JOIN event e ON e.id = es.event_id
            WHERE datetime(es.imported_at) >= datetime(?) AND datetime(es.imported_at) < datetime(?)
              AND datetime(e.added_at) < datetime(?)
            """,
            (start_raw, end_raw, start_raw),
        )

        cur = await conn.execute(
            """
            WITH updated_events AS (
                SELECT DISTINCT es.event_id
                FROM event_source es
                JOIN event e ON e.id = es.event_id
                WHERE datetime(es.imported_at) >= datetime(?) AND datetime(es.imported_at) < datetime(?)
                  AND datetime(e.added_at) < datetime(?)
            )
            SELECT source_count, COUNT(*)
            FROM (
                SELECT ue.event_id, COUNT(*) AS source_count
                FROM updated_events ue
                JOIN event_source es2 ON es2.event_id = ue.event_id
                GROUP BY ue.event_id
            )
            GROUP BY source_count
            ORDER BY source_count
            """,
            (start_raw, end_raw, start_raw),
        )
        source_distribution_rows = await cur.fetchall()
        updated_distribution = {
            _parse_int(source_count): _parse_int(amount)
            for source_count, amount in (source_distribution_rows or [])
            if _parse_int(source_count) > 0 and _parse_int(amount) > 0
        }

        cur = await conn.execute(
            """
            SELECT city_norm, details
            FROM geo_city_region_cache
            WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)
            ORDER BY datetime(created_at) ASC, city_norm ASC
            """,
            (start_raw, end_raw),
        )
        geo_rows = await cur.fetchall()
        new_cities: list[dict[str, str | None]] = []
        for city_norm, details_raw in geo_rows or []:
            details = _json_load(details_raw)
            new_cities.append(
                {
                    "city_norm": str(city_norm or "").strip(),
                    "raw_example": _extract_city_example(details),
                }
            )

        festival_queue_added = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM festival_queue
            WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        festivals_created = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM festival
            WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )
        festivals_updated = await _fetch_int(
            conn,
            """
            SELECT COUNT(*) FROM festival
            WHERE datetime(updated_at) >= datetime(?) AND datetime(updated_at) < datetime(?)
            """,
            (start_raw, end_raw),
        )

    vk_runs = await _fetch_ops_runs(db, kind="vk_auto_import", start_utc=window.start_utc, end_utc=window.end_utc)
    vk_queue_parsed = _sum_run_metric(vk_runs, "inbox_processed")
    if vk_queue_parsed <= 0:
        vk_queue_parsed = int(vk_posts_auto_imported)
    parse_runs = await _fetch_ops_runs(db, kind="parse", start_utc=window.start_utc, end_utc=window.end_utc)
    threed_runs = await _fetch_ops_runs(db, kind="3di", start_utc=window.start_utc, end_utc=window.end_utc)
    festival_queue_runs = await _fetch_ops_runs(
        db,
        kind="festival_queue",
        start_utc=window.start_utc,
        end_utc=window.end_utc,
    )
    tg_monitor_runs = await _fetch_ops_runs(
        db,
        kind="tg_monitoring",
        start_utc=window.start_utc,
        end_utc=window.end_utc,
    )
    guide_monitor_runs = await _fetch_ops_runs(
        db,
        kind="guide_monitoring",
        start_utc=window.start_utc,
        end_utc=window.end_utc,
    )
    parse_breakdown = _aggregate_parse_breakdown(parse_runs)

    gemma_requests_count = await _collect_gemma_requests_count(
        supabase_client=supabase_client,
        start_utc=window.start_utc,
        end_utc=window.end_utc,
    )
    bucket_usage_mb = await _get_bucket_usage_mb(
        db,
        supabase_client=supabase_client,
        bucket_name=(bucket_name or os.getenv("SUPABASE_BUCKET", "events-ics")),
    )

    metrics["vk"] = {
        "vk_posts_added": vk_posts_added,
        "vk_posts_auto_imported": vk_posts_auto_imported,
        "vk_events_from_auto_import": vk_events_from_auto_import,
        "vk_queue_added_period": vk_posts_added,
        "vk_queue_parsed_period": vk_queue_parsed,
        "vk_queue_unresolved_now": vk_queue_unresolved_now,
        "vk_auto_import_runs": vk_runs,
    }
    metrics["telegram"] = {
        "sources_scanned": tg_sources_scanned,
        "messages_with_events": tg_messages_with_events,
        "sources_with_events": tg_sources_with_events,
        "tg_monitoring_runs": tg_monitor_runs,
    }
    metrics["guide_excursions"] = {
        "sources_scanned": guide_sources_scanned,
        "posts_prefiltered": guide_posts_prefiltered,
        "occurrences_new": guide_occurrences_new,
        "occurrences_updated": guide_occurrences_updated,
        "occurrences_future_now": guide_occurrences_future_now,
        "templates_total": guide_templates_total,
        "digest_published": guide_digest_published,
        "guide_monitoring_runs": guide_monitor_runs,
    }
    metrics["parse"] = {
        "runs": parse_runs,
        "source_breakdown": parse_breakdown,
    }
    metrics["three_di"] = {
        "runs": threed_runs,
    }
    metrics["events"] = {
        "events_created": events_created,
        "events_updated": events_updated,
        "updated_sources_distribution": updated_distribution,
    }
    metrics["geo"] = {
        "new_cities_count": len(new_cities),
        "new_cities": new_cities,
    }
    metrics["festivals"] = {
        "festival_queue_added": festival_queue_added,
        "festival_queue_runs": festival_queue_runs,
        "festivals_created": festivals_created,
        "festivals_updated": festivals_updated,
    }
    metrics["tech"] = {
        "gemma_requests_count": gemma_requests_count,
        "bucket_usage_mb": bucket_usage_mb,
    }
    return GeneralStatsSnapshot(window=window, metrics=metrics)


def _format_run_lines(
    runs: Sequence[Mapping[str, Any]],
    *,
    tz: ZoneInfo,
    metric_keys: Sequence[str],
    limit: int = 8,
) -> list[str]:
    if not runs:
        return ["- запусков нет"]
    out: list[str] = []
    for run in list(runs)[-limit:]:
        started = run.get("started_at")
        started_txt = "?"
        if isinstance(started, datetime):
            started_txt = started.astimezone(tz).strftime("%Y-%m-%d %H:%M")
        status = str(run.get("status") or "unknown")
        trigger = str(run.get("trigger") or "manual")
        metrics = run.get("metrics")
        metrics_map = metrics if isinstance(metrics, Mapping) else {}
        details = run.get("details")
        details_map = details if isinstance(details, Mapping) else {}
        parts = [f"- {started_txt}", f"status={status}", f"trigger={trigger}"]

        skip_reason = str(details_map.get("skip_reason") or "").strip()
        if skip_reason:
            parts.append(f"reason={skip_reason}")
        blocked_by = str(details_map.get("blocked_by_kind") or "").strip()
        if blocked_by:
            parts.append(f"blocked_by={blocked_by}")

        took_sec = None
        try:
            took_sec = float(metrics_map.get("duration_sec")) if "duration_sec" in metrics_map else None
        except Exception:
            took_sec = None
        if took_sec is None:
            finished = run.get("finished_at")
            if isinstance(started, datetime) and isinstance(finished, datetime):
                took_sec = max(0.0, float((finished - started).total_seconds()))
            elif isinstance(started, datetime) and status == "running":
                took_sec = max(0.0, float((datetime.now(timezone.utc) - started).total_seconds()))
        if took_sec is not None:
            if took_sec >= 3600:
                parts.append(f"took={took_sec / 3600.0:.1f}h")
            elif took_sec >= 60:
                parts.append(f"took={took_sec / 60.0:.1f}m")
            else:
                parts.append(f"took={took_sec:.0f}s")
        for key in metric_keys:
            if key in metrics_map:
                parts.append(f"{key}={_parse_int(metrics_map.get(key))}")
        out.append(" ".join(parts))
    return out


def format_general_stats_message(snapshot: GeneralStatsSnapshot) -> str:
    tz = _safe_zoneinfo(snapshot.window.tz_name)
    vk = snapshot.metrics.get("vk", {})
    tg = snapshot.metrics.get("telegram", {})
    parse = snapshot.metrics.get("parse", {})
    three_di = snapshot.metrics.get("three_di", {})
    events = snapshot.metrics.get("events", {})
    geo = snapshot.metrics.get("geo", {})
    festivals = snapshot.metrics.get("festivals", {})
    guide = snapshot.metrics.get("guide_excursions", {})
    tech = snapshot.metrics.get("tech", {})

    lines = [
        "📊 Общий суточный отчёт",
        f"Период ({snapshot.window.tz_name}):",
        f"- start: {_local_ts(snapshot.window.start_local, tz)}",
        f"- end: {_local_ts(snapshot.window.end_local, tz)}",
        "",
        "VK:",
        f"- vk_queue_added_period: {_parse_int(vk.get('vk_queue_added_period'))}",
        f"- vk_queue_parsed_period: {_parse_int(vk.get('vk_queue_parsed_period'))}",
        f"- vk_queue_unresolved_now: {_parse_int(vk.get('vk_queue_unresolved_now'))}",
        f"- vk_posts_added: {_parse_int(vk.get('vk_posts_added'))}",
        f"- vk_posts_auto_imported: {_parse_int(vk.get('vk_posts_auto_imported'))}",
        f"- vk_events_from_auto_import: {_parse_int(vk.get('vk_events_from_auto_import'))}",
        "- vk_auto_import runs:",
    ]
    lines.extend(
        _format_run_lines(
            vk.get("vk_auto_import_runs") or [],
            tz=tz,
            metric_keys=("inbox_processed", "inbox_imported", "inbox_rejected"),
        )
    )

    lines.extend(
        [
            "",
            "Telegram monitoring:",
            f"- sources_scanned: {_parse_int(tg.get('sources_scanned'))}",
            f"- messages_with_events: {_parse_int(tg.get('messages_with_events'))}",
            f"- sources_with_events: {_parse_int(tg.get('sources_with_events'))}",
            "- tg_monitoring runs:",
        ]
    )
    lines.extend(
        _format_run_lines(
            tg.get("tg_monitoring_runs") or [],
            tz=tz,
            metric_keys=("sources_scanned", "messages_processed", "messages_with_events"),
        )
    )

    lines.extend(
        [
            "",
            "Guide excursions:",
            f"- sources_scanned: {_parse_int(guide.get('sources_scanned'))}",
            f"- posts_prefiltered: {_parse_int(guide.get('posts_prefiltered'))}",
            f"- occurrences_new: {_parse_int(guide.get('occurrences_new'))}",
            f"- occurrences_updated: {_parse_int(guide.get('occurrences_updated'))}",
            f"- occurrences_future_now: {_parse_int(guide.get('occurrences_future_now'))}",
            f"- templates_total: {_parse_int(guide.get('templates_total'))}",
            f"- digest_published: {_parse_int(guide.get('digest_published'))}",
            "- guide_monitoring runs:",
        ]
    )
    lines.extend(
        _format_run_lines(
            guide.get("guide_monitoring_runs") or [],
            tz=tz,
            metric_keys=("sources_scanned", "posts_scanned", "occurrences_created", "occurrences_updated"),
        )
    )

    lines.extend(["", "/parse runs:"])
    lines.extend(
        _format_run_lines(
            parse.get("runs") or [],
            tz=tz,
            metric_keys=("total_events", "events_created", "events_updated"),
        )
    )
    breakdown = parse.get("source_breakdown") or {}
    if isinstance(breakdown, Mapping) and breakdown:
        lines.append("- /parse breakdown:")
        for source in sorted(breakdown):
            payload = breakdown[source] if isinstance(breakdown[source], Mapping) else {}
            lines.append(
                f"  {source}: processed={_parse_int(payload.get('processed'))} "
                f"new={_parse_int(payload.get('new_events'))}"
            )

    lines.extend(["", "/3di runs:"])
    lines.extend(
        _format_run_lines(
            three_di.get("runs") or [],
            tz=tz,
            metric_keys=("events_considered", "previews_rendered", "preview_errors", "preview_skipped"),
        )
    )

    lines.extend(
        [
            "",
            "Events:",
            f"- events_created: {_parse_int(events.get('events_created'))}",
            f"- events_updated: {_parse_int(events.get('events_updated'))}",
        ]
    )
    distribution = events.get("updated_sources_distribution") or {}
    if isinstance(distribution, Mapping) and distribution:
        dist_parts = [f"{k}src={_parse_int(v)}" for k, v in sorted(distribution.items(), key=lambda item: int(item[0]))]
        lines.append(f"- updated_distribution: {', '.join(dist_parts)}")

    lines.extend(
        [
            "",
            "Geo:",
            f"- new_cities: {_parse_int(geo.get('new_cities_count'))}",
        ]
    )
    cities = geo.get("new_cities") or []
    if isinstance(cities, Sequence):
        for city in list(cities)[:20]:
            if not isinstance(city, Mapping):
                continue
            city_name = str(city.get("city_norm") or "").strip()
            if not city_name:
                continue
            raw_example = str(city.get("raw_example") or "").strip()
            if raw_example:
                lines.append(f"  - {city_name} ({raw_example})")
            else:
                lines.append(f"  - {city_name}")

    lines.extend(
        [
            "",
            "Festivals:",
            f"- festival_queue_added: {_parse_int(festivals.get('festival_queue_added'))}",
            "- festival_queue runs:",
        ]
    )
    lines.extend(
        _format_run_lines(
            festivals.get("festival_queue_runs") or [],
            tz=tz,
            metric_keys=("processed", "success", "failed", "skipped"),
        )
    )
    lines.extend(
        [
            f"- festivals_created: {_parse_int(festivals.get('festivals_created'))}",
            f"- festivals_updated: {_parse_int(festivals.get('festivals_updated'))}",
            "",
            "Tech:",
            f"- gemma_requests_count: {tech.get('gemma_requests_count') if tech.get('gemma_requests_count') is not None else 'n/a'}",
            f"- bucket_usage_mb: {tech.get('bucket_usage_mb') if tech.get('bucket_usage_mb') is not None else 'n/a'}",
        ]
    )
    return "\n".join(lines)


def _parse_chat_id(raw: str | None, *, env_name: str) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        logger.warning("general_stats: invalid %s=%r", env_name, raw)
        return None


async def send_general_stats_report(
    db: Database,
    bot: Any,
    *,
    chat_ids: Sequence[int],
    trigger: str,
    operator_id: int | None = None,
    run_id: str | None = None,
    tz_name: str | None = None,
    supabase_client: Any | None = None,
    bucket_name: str | None = None,
    now: datetime | None = None,
) -> GeneralStatsSnapshot | None:
    unique_chat_ids: list[int] = []
    for chat_id in chat_ids:
        value = _parse_int(chat_id)
        if value and value not in unique_chat_ids:
            unique_chat_ids.append(value)

    ops_run_id = await start_ops_run(
        db,
        kind="general_stats",
        trigger=trigger,
        chat_id=(unique_chat_ids[0] if unique_chat_ids else None),
        operator_id=operator_id,
        details={"run_id": run_id, "chat_ids": unique_chat_ids},
    )
    status = "success"
    snapshot: GeneralStatsSnapshot | None = None
    sent = 0
    failed: dict[int, str] = {}
    try:
        if not unique_chat_ids:
            status = "skipped"
            logger.warning("general_stats: no target chats configured for trigger=%s", trigger)
            return None

        snapshot = await collect_general_stats(
            db,
            tz_name=tz_name,
            now=now,
            supabase_client=supabase_client,
            bucket_name=bucket_name,
        )
        text = format_general_stats_message(snapshot)
        for chat_id in unique_chat_ids:
            try:
                await bot.send_message(chat_id, text, disable_web_page_preview=True)
                sent += 1
            except Exception as exc:
                logger.warning("general_stats: send failed chat_id=%s", chat_id, exc_info=True)
                failed[int(chat_id)] = str(exc)
        if sent <= 0:
            status = "error"
        return snapshot
    except Exception:
        status = "error"
        raise
    finally:
        await finish_ops_run(
            db,
            run_id=ops_run_id,
            status=status,
            metrics={
                "sent_chats": int(sent),
                "failed_chats": int(len(failed)),
            },
            details={
                "run_id": run_id,
                "chat_ids": unique_chat_ids,
                "failed": failed,
            },
        )


def _resolve_supabase_client() -> Any | None:
    import sys

    for module_name in ("__main__", "main"):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        getter = getattr(mod, "get_supabase_client", None)
        if not callable(getter):
            continue
        try:
            return getter()
        except Exception:
            logger.warning(
                "general_stats: failed to resolve supabase client from %s",
                module_name,
                exc_info=True,
            )
            continue

    try:
        import main as main_mod  # local import to avoid circular deps at module import time
    except Exception:
        return None
    getter = getattr(main_mod, "get_supabase_client", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        logger.warning("general_stats: failed to resolve supabase client", exc_info=True)
        return None


async def general_stats_scheduler(
    db: Database,
    bot,
    *,
    run_id: str | None = None,
) -> None:
    operator_chat_id = _parse_chat_id(os.getenv("OPERATOR_CHAT_ID"), env_name="OPERATOR_CHAT_ID")
    admin_chat_id = await resolve_superadmin_chat_id(db)
    targets = [chat_id for chat_id in (operator_chat_id, admin_chat_id) if chat_id is not None]
    if not targets:
        logger.warning("general_stats: no valid target chat IDs configured")
        return
    await send_general_stats_report(
        db,
        bot,
        chat_ids=targets,
        trigger="scheduled",
        operator_id=0,
        run_id=run_id,
        tz_name=os.getenv("GENERAL_STATS_TZ", DEFAULT_GENERAL_STATS_TZ),
        supabase_client=_resolve_supabase_client(),
        bucket_name=os.getenv("SUPABASE_BUCKET", "events-ics"),
    )
