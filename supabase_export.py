"""Supabase export helpers for VK crawler telemetry."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_MISS_SAMPLE_RATE = 0.1


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except ValueError:
        return fallback
    return max(0, parsed)


def _parse_float(value: str | None, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except ValueError:
        return fallback
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def _ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()
    except Exception:
        return None


class SBExporter:
    """Helper for exporting crawler telemetry to Supabase."""

    def __init__(self, client_factory: Callable[[], Any]) -> None:
        self._client_factory = client_factory
        self._client: Any | None = None
        self._enabled = _parse_bool(os.getenv("SUPABASE_EXPORT_ENABLED"))
        self._retention_days = _parse_int(
            os.getenv("SUPABASE_RETENTION_DAYS"), _DEFAULT_RETENTION_DAYS
        )
        self._miss_sample_rate = _parse_float(
            os.getenv("VK_MISSES_SAMPLE_RATE"), _DEFAULT_MISS_SAMPLE_RATE
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_client(self) -> Any | None:
        if not self._enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            client = self._client_factory()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Supabase export failed: %s", exc)
            return None
        if client is None:
            return None
        self._client = client
        return client

    def upsert_group_meta(
        self,
        group_id: int,
        *,
        screen_name: str | None = None,
        name: str | None = None,
        location: str | None = None,
        default_time: str | None = None,
        default_ticket_link: str | None = None,
    ) -> None:
        client = self._get_client()
        if client is None:
            return
        payload = {
            "group_id": group_id,
            "screen_name": screen_name,
            "name": name,
            "location": location,
            "default_time": default_time,
            "default_ticket_link": default_ticket_link,
        }
        try:
            client.table("vk_group_meta").upsert(  # type: ignore[operator]
                payload,
                on_conflict="group_id",
            ).execute()
            logger.debug("Supabase group meta upserted: group_id=%s", group_id)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Supabase export failed: %s", exc)

    def write_snapshot(
        self,
        *,
        group_id: int,
        post_id: int,
        post_ts: int | None,
        url: str | None,
        matched_keywords: Sequence[str] | None,
        has_date: bool,
        event_ts_hint: int | None,
        photos_count: int,
        text: str | None,
    ) -> None:
        client = self._get_client()
        if client is None:
            return
        truncated = (text or "")[:2000]
        keywords = list(matched_keywords or [])[:20]
        payload = {
            "group_id": group_id,
            "post_id": post_id,
            "post_ts": _ts_to_iso(post_ts),
            "post_url": url,
            "matched_keywords": keywords,
            "has_date": has_date,
            "event_ts_hint": _ts_to_iso(event_ts_hint),
            "photos_count": photos_count,
            "text_excerpt": truncated,
        }
        try:
            client.table("vk_post_snapshots").upsert(  # type: ignore[operator]
                payload,
                on_conflict="group_id,post_id",
            ).execute()
            logger.debug(
                "Supabase snapshot stored: group_id=%s post_id=%s", group_id, post_id
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Supabase export failed: %s", exc)

    def log_miss(
        self,
        *,
        group_id: int,
        post_id: int,
        url: str | None,
        reason: str,
        matched_keywords: Sequence[str] | None = None,
        post_ts: int | None = None,
        event_ts_hint: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        if self._miss_sample_rate <= 0:
            return
        if random.random() > self._miss_sample_rate:
            return
        client = self._get_client()
        if client is None:
            return
        payload: dict[str, Any] = {
            "group_id": group_id,
            "post_id": post_id,
            "post_url": url,
            "reason": reason,
            "matched_keywords": list(matched_keywords or [])[:20],
            "post_ts": _ts_to_iso(post_ts),
            "event_ts_hint": _ts_to_iso(event_ts_hint),
        }
        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value
        try:
            client.table("vk_post_misses").insert(payload).execute()  # type: ignore[operator]
            logger.debug(
                "Supabase miss logged: group_id=%s post_id=%s reason=%s",
                group_id,
                post_id,
                reason,
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Supabase export failed: %s", exc)

    def retention(self) -> None:
        client = self._get_client()
        if client is None:
            return
        if self._retention_days <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        cutoff_iso = cutoff.isoformat()
        for table in ("vk_post_snapshots", "vk_post_misses"):
            try:
                client.table(table).delete().lt(  # type: ignore[operator]
                    "created_at", cutoff_iso
                ).execute()
                logger.debug(
                    "Supabase retention applied: table=%s cutoff=%s", table, cutoff_iso
                )
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("Supabase export failed: %s", exc)
