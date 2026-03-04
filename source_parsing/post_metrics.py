from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from db import Database


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    data = sorted(int(v) for v in values)
    n = len(data)
    mid = n // 2
    if n % 2 == 1:
        return int(data[mid])
    return int((data[mid - 1] + data[mid]) // 2)


@dataclass(slots=True)
class PopularityBaseline:
    median_views: int | None
    median_likes: int | None
    sample: int


@dataclass(slots=True)
class PopularityOverview:
    """Helper payload for operator UIs (e.g. /tg sources list).

    Coverage is expressed as "how many distinct publish days have at least one post"
    inside the horizon window that contributes to the baseline sample.
    """

    baseline: PopularityBaseline
    days_covered: int
    horizon_days: int
    used_fallback: bool


@dataclass(slots=True)
class PopularityMarks:
    views_level: int = 0
    likes_level: int = 0

    @property
    def text(self) -> str:
        out = ""
        if int(self.views_level or 0) > 0:
            out += "⭐" * int(self.views_level)
        if int(self.likes_level or 0) > 0:
            out += "👍" * int(self.likes_level)
        return out


def _popularity_level(
    *,
    value: int | None,
    median: int | None,
    mult: float,
    step: float,
    max_level: int,
) -> int:
    if not isinstance(value, int) or value < 0:
        return 0
    if not isinstance(median, int) or median < 0:
        return 0
    if max_level <= 0:
        return 0
    threshold_1 = float(median) * float(mult)
    if float(value) <= threshold_1:
        return 0
    if max_level == 1:
        return 1
    if float(step) <= 0.0:
        return 1

    # Extra levels: each additional icon requires +step*median above threshold_1.
    # We clamp delta to >= 1 to keep behavior sane when median is 0 or step is tiny.
    delta = int(round(float(median) * float(step)))
    if delta <= 0:
        delta = 1
    pos = float(value) - threshold_1
    level = int(math.ceil(pos / float(delta)))
    if level < 1:
        level = 1
    if level > int(max_level):
        level = int(max_level)
    return int(level)


def popularity_marks(
    *,
    views: int | None,
    likes: int | None,
    baseline: PopularityBaseline | None,
) -> PopularityMarks:
    if not baseline or baseline.sample <= 0:
        return PopularityMarks()
    # Default: allow markers to appear even in fresh environments (one monitoring run
    # can already provide a meaningful per-channel median baseline).
    min_sample = _env_int("POST_POPULARITY_MIN_SAMPLE", 2)
    if baseline.sample < min_sample:
        return PopularityMarks()

    # Default: compare against the channel median (multiplier=1.0). Operators can
    # raise multipliers to highlight only "significantly above median" posts.
    views_mult = _env_float("POST_POPULARITY_VIEWS_MULT", 1.0)
    likes_mult = _env_float("POST_POPULARITY_LIKES_MULT", 1.0)
    max_level = max(1, _env_int("POST_POPULARITY_MAX_LEVEL", 4))
    views_step = max(0.0, _env_float("POST_POPULARITY_VIEWS_STEP", 0.5))
    likes_step = max(0.0, _env_float("POST_POPULARITY_LIKES_STEP", 0.5))

    out = PopularityMarks()
    out.views_level = _popularity_level(
        value=views,
        median=baseline.median_views,
        mult=views_mult,
        step=views_step,
        max_level=max_level,
    )
    out.likes_level = _popularity_level(
        value=likes,
        median=baseline.median_likes,
        mult=likes_mult,
        step=likes_step,
        max_level=max_level,
    )
    return out


def compute_age_day(*, published_ts: int | None, collected_ts: int | None) -> int | None:
    if not isinstance(published_ts, int) or not isinstance(collected_ts, int):
        return None
    if published_ts <= 0 or collected_ts <= 0:
        return None
    delta = collected_ts - published_ts
    if delta < 0:
        delta = 0
    return int(delta // 86400)


def _max_age_day() -> int:
    return max(0, _env_int("POST_POPULARITY_MAX_AGE_DAY", 2))


def normalize_age_day(age_day: int | None) -> int | None:
    if not isinstance(age_day, int) or age_day < 0:
        return None
    if age_day > _max_age_day():
        return None
    return int(age_day)


def default_retention_days() -> int:
    return max(
        1,
        _env_int(
            "POST_METRICS_RETENTION_DAYS",
            _env_int("POST_POPULARITY_HORIZON_DAYS", 90),
        ),
    )


async def cleanup_post_metrics(
    db: Database,
    *,
    retention_days: int | None = None,
    now_ts: int | None = None,
) -> dict[str, int]:
    """Delete stale TG/VK metric snapshots to bound DB growth.

    Retention is based on publish timestamp when available, otherwise on collected timestamp.
    """
    keep_days = int(retention_days or default_retention_days())
    now_ts = int(now_ts or _utc_now_ts())
    cutoff = now_ts - keep_days * 86400
    deleted_tg = 0
    deleted_vk = 0
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            DELETE FROM telegram_post_metric
            WHERE (message_ts IS NOT NULL AND message_ts < ?)
               OR (message_ts IS NULL AND collected_ts < ?)
            """,
            (int(cutoff), int(cutoff)),
        )
        deleted_tg = int(getattr(cur, "rowcount", 0) or 0)
        cur2 = await conn.execute(
            """
            DELETE FROM vk_post_metric
            WHERE (post_ts IS NOT NULL AND post_ts < ?)
               OR (post_ts IS NULL AND collected_ts < ?)
            """,
            (int(cutoff), int(cutoff)),
        )
        deleted_vk = int(getattr(cur2, "rowcount", 0) or 0)
        await conn.commit()
    return {"telegram_post_metric": deleted_tg, "vk_post_metric": deleted_vk}


async def upsert_telegram_post_metric(
    db: Database,
    *,
    source_id: int,
    message_id: int,
    age_day: int,
    source_url: str | None,
    message_ts: int | None,
    views: int | None,
    likes: int | None,
    reactions: dict[str, Any] | None = None,
    collected_ts: int | None = None,
) -> None:
    collected_ts = int(collected_ts or _utc_now_ts())
    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO telegram_post_metric(
                source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes, reactions_json
            )
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source_id, message_id, age_day) DO UPDATE SET
                collected_ts=excluded.collected_ts,
                source_url=COALESCE(excluded.source_url, telegram_post_metric.source_url),
                message_ts=COALESCE(excluded.message_ts, telegram_post_metric.message_ts),
                views=MAX(COALESCE(telegram_post_metric.views, -1), COALESCE(excluded.views, -1)),
                likes=MAX(COALESCE(telegram_post_metric.likes, -1), COALESCE(excluded.likes, -1)),
                reactions_json=COALESCE(excluded.reactions_json, telegram_post_metric.reactions_json)
            """,
            (
                int(source_id),
                int(message_id),
                int(age_day),
                str(source_url) if source_url else None,
                int(message_ts) if isinstance(message_ts, int) else None,
                int(collected_ts),
                int(views) if isinstance(views, int) else None,
                int(likes) if isinstance(likes, int) else None,
                reactions if isinstance(reactions, dict) else None,
            ),
        )
        await conn.commit()


async def load_telegram_popularity_baseline(
    db: Database,
    *,
    source_id: int,
    age_day: int,
    horizon_days: int | None = None,
    now_ts: int | None = None,
) -> PopularityBaseline:
    horizon_days = int(horizon_days or _env_int("POST_POPULARITY_HORIZON_DAYS", 90))
    now_ts = int(now_ts or _utc_now_ts())
    since_ts = now_ts - max(1, horizon_days) * 86400
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT m.message_id, MAX(m.views) AS v, MAX(m.likes) AS l
            FROM telegram_post_metric m
            JOIN telegram_scanned_message s
              ON s.source_id = m.source_id
             AND s.message_id = m.message_id
            WHERE m.source_id=?
              AND m.age_day=?
              AND m.message_ts IS NOT NULL
              AND m.message_ts >= ?
              AND COALESCE(s.events_extracted, 0) > 0
            GROUP BY m.message_id
            """,
            (int(source_id), int(age_day), int(since_ts)),
        )
        rows = await cur.fetchall()
        # If the exact age bucket is too sparse (common on the first run), fall back to a
        # per-channel baseline across all "recent" buckets. This keeps ⭐/👍 markers
        # responsive even with a small sample.
        min_sample = _env_int("POST_POPULARITY_MIN_SAMPLE", 2)
        if len({int(r[0]) for r in rows if r and r[0] is not None}) < min_sample:
            max_age = _max_age_day()
            cur2 = await conn.execute(
                """
                SELECT m.message_id, MAX(m.views) AS v, MAX(m.likes) AS l
                FROM telegram_post_metric m
                JOIN telegram_scanned_message s
                  ON s.source_id = m.source_id
                 AND s.message_id = m.message_id
                WHERE m.source_id=?
                  AND m.age_day <= ?
                  AND m.message_ts IS NOT NULL
                  AND m.message_ts >= ?
                  AND COALESCE(s.events_extracted, 0) > 0
                GROUP BY m.message_id
                """,
                (int(source_id), int(max_age), int(since_ts)),
            )
            rows = await cur2.fetchall()
    views_vals: list[int] = []
    likes_vals: list[int] = []
    for _mid, v, l in rows:
        if isinstance(v, int) and v >= 0:
            views_vals.append(int(v))
        if isinstance(l, int) and l >= 0:
            likes_vals.append(int(l))
    sample = len({int(r[0]) for r in rows if r and r[0] is not None})
    return PopularityBaseline(
        median_views=_median_int(views_vals),
        median_likes=_median_int(likes_vals),
        sample=int(sample),
    )


async def load_telegram_popularity_overview(
    db: Database,
    *,
    source_id: int,
    age_day: int,
    horizon_days: int | None = None,
    now_ts: int | None = None,
) -> PopularityOverview:
    """Compute a per-source popularity baseline plus "days covered" for operator display.

    The baseline logic matches `load_telegram_popularity_baseline` including its
    fallback when the exact `age_day` bucket is too sparse.
    """

    horizon_days = int(horizon_days or _env_int("POST_POPULARITY_HORIZON_DAYS", 90))
    now_ts = int(now_ts or _utc_now_ts())
    since_ts = now_ts - max(1, horizon_days) * 86400

    async def _query_rows(*, fallback: bool) -> list[tuple[int, int, int | None, int | None]]:
        async with db.raw_conn() as conn:
            if not fallback:
                cur = await conn.execute(
                    """
                    SELECT m.message_id, MAX(m.message_ts) AS ts, MAX(m.views) AS v, MAX(m.likes) AS l
                    FROM telegram_post_metric m
                    JOIN telegram_scanned_message s
                      ON s.source_id = m.source_id
                     AND s.message_id = m.message_id
                    WHERE m.source_id=?
                      AND m.age_day=?
                      AND m.message_ts IS NOT NULL
                      AND m.message_ts >= ?
                      AND COALESCE(s.events_extracted, 0) > 0
                    GROUP BY m.message_id
                    """,
                    (int(source_id), int(age_day), int(since_ts)),
                )
                return await cur.fetchall()

            max_age = _max_age_day()
            cur2 = await conn.execute(
                """
                SELECT m.message_id, MAX(m.message_ts) AS ts, MAX(m.views) AS v, MAX(m.likes) AS l
                FROM telegram_post_metric m
                JOIN telegram_scanned_message s
                  ON s.source_id = m.source_id
                 AND s.message_id = m.message_id
                WHERE m.source_id=?
                  AND m.age_day <= ?
                  AND m.message_ts IS NOT NULL
                  AND m.message_ts >= ?
                  AND COALESCE(s.events_extracted, 0) > 0
                GROUP BY m.message_id
                """,
                (int(source_id), int(max_age), int(since_ts)),
            )
            return await cur2.fetchall()

    rows = await _query_rows(fallback=False)
    min_sample = _env_int("POST_POPULARITY_MIN_SAMPLE", 2)
    used_fallback = False
    unique_ids = {int(r[0]) for r in rows if r and r[0] is not None}
    if len(unique_ids) < min_sample:
        used_fallback = True
        rows = await _query_rows(fallback=True)
        unique_ids = {int(r[0]) for r in rows if r and r[0] is not None}

    views_vals: list[int] = []
    likes_vals: list[int] = []
    day_keys: set[int] = set()
    for _mid, ts, v, l in rows:
        if isinstance(v, int) and v >= 0:
            views_vals.append(int(v))
        if isinstance(l, int) and l >= 0:
            likes_vals.append(int(l))
        if isinstance(ts, int) and ts > 0:
            day_keys.add(int(ts // 86400))

    baseline = PopularityBaseline(
        median_views=_median_int(views_vals),
        median_likes=_median_int(likes_vals),
        sample=int(len(unique_ids)),
    )
    return PopularityOverview(
        baseline=baseline,
        days_covered=int(len(day_keys)),
        horizon_days=int(horizon_days),
        used_fallback=bool(used_fallback),
    )


async def upsert_vk_post_metric(
    db: Database,
    *,
    group_id: int,
    post_id: int,
    age_day: int,
    source_url: str | None,
    post_ts: int | None,
    views: int | None,
    likes: int | None,
    collected_ts: int | None = None,
) -> None:
    collected_ts = int(collected_ts or _utc_now_ts())
    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO vk_post_metric(
                group_id, post_id, age_day, source_url, post_ts, collected_ts, views, likes
            )
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(group_id, post_id, age_day) DO UPDATE SET
                collected_ts=excluded.collected_ts,
                source_url=COALESCE(excluded.source_url, vk_post_metric.source_url),
                post_ts=COALESCE(excluded.post_ts, vk_post_metric.post_ts),
                views=MAX(COALESCE(vk_post_metric.views, -1), COALESCE(excluded.views, -1)),
                likes=MAX(COALESCE(vk_post_metric.likes, -1), COALESCE(excluded.likes, -1))
            """,
            (
                int(group_id),
                int(post_id),
                int(age_day),
                str(source_url) if source_url else None,
                int(post_ts) if isinstance(post_ts, int) else None,
                int(collected_ts),
                int(views) if isinstance(views, int) else None,
                int(likes) if isinstance(likes, int) else None,
            ),
        )
        await conn.commit()


async def load_vk_popularity_baseline(
    db: Database,
    *,
    group_id: int,
    age_day: int,
    horizon_days: int | None = None,
    now_ts: int | None = None,
) -> PopularityBaseline:
    horizon_days = int(horizon_days or _env_int("POST_POPULARITY_HORIZON_DAYS", 90))
    now_ts = int(now_ts or _utc_now_ts())
    since_ts = now_ts - max(1, horizon_days) * 86400
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT m.post_id, MAX(m.views) AS v, MAX(m.likes) AS l
            FROM vk_post_metric m
            JOIN vk_inbox i
              ON i.group_id = m.group_id
             AND i.post_id = m.post_id
            JOIN vk_inbox_import_event ie
              ON ie.inbox_id = i.id
            WHERE m.group_id=?
              AND m.age_day=?
              AND m.post_ts IS NOT NULL
              AND m.post_ts >= ?
            GROUP BY m.post_id
            """,
            (int(group_id), int(age_day), int(since_ts)),
        )
        rows = await cur.fetchall()
        min_sample = _env_int("POST_POPULARITY_MIN_SAMPLE", 2)
        if len({int(r[0]) for r in rows if r and r[0] is not None}) < min_sample:
            max_age = _max_age_day()
            cur2 = await conn.execute(
                """
                SELECT m.post_id, MAX(m.views) AS v, MAX(m.likes) AS l
                FROM vk_post_metric m
                JOIN vk_inbox i
                  ON i.group_id = m.group_id
                 AND i.post_id = m.post_id
                JOIN vk_inbox_import_event ie
                  ON ie.inbox_id = i.id
                WHERE m.group_id=?
                  AND m.age_day <= ?
                  AND m.post_ts IS NOT NULL
                  AND m.post_ts >= ?
                GROUP BY m.post_id
                """,
                (int(group_id), int(max_age), int(since_ts)),
            )
            rows = await cur2.fetchall()
    views_vals: list[int] = []
    likes_vals: list[int] = []
    for _pid, v, l in rows:
        if isinstance(v, int) and v >= 0:
            views_vals.append(int(v))
        if isinstance(l, int) and l >= 0:
            likes_vals.append(int(l))
    sample = len({int(r[0]) for r in rows if r and r[0] is not None})
    return PopularityBaseline(
        median_views=_median_int(views_vals),
        median_likes=_median_int(likes_vals),
        sample=int(sample),
    )
