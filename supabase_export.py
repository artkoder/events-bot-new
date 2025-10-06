"""Supabase export helpers for VK crawler telemetry."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_DAYS = 60
_DEFAULT_MISS_SAMPLE_RATE = 0.1


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
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
        self._enabled = _parse_bool(
            os.getenv("SUPABASE_EXPORT_ENABLED"), default=True
        )
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
        **_: Any,
    ) -> None:
        client = self._get_client()
        if client is None:
            return
        payload = {
            "group_id": group_id,
            "group_screen_name": screen_name,
            "group_title": name,
        }
        try:
            client.table("vk_groups").upsert(  # type: ignore[operator]
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
        counters: Mapping[str, Any],
    ) -> None:
        client = self._get_client()
        if client is None:
            return
        payload: dict[str, Any] = {"group_id": group_id}
        for key, value in counters.items():
            if value is None:
                continue
            if isinstance(value, (int, float, bool, str)):
                payload[key] = value
            else:
                payload[key] = str(value)
        try:
            client.table("vk_crawl_snapshots").insert(payload).execute()  # type: ignore[operator]
            logger.debug("Supabase snapshot stored: group_id=%s", group_id)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Supabase export failed: %s", exc)

    def should_log_miss(self, kw_ok: bool, has_date: bool) -> bool:
        """Return True when a miss must be logged regardless of sampling."""

        return bool(kw_ok) ^ bool(has_date)

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
        flags: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        kw_ok: bool | None = None,
        has_date: bool | None = None,
    ) -> None:
        if not self._enabled:
            return
        mandatory = False
        if kw_ok is not None and has_date is not None:
            mandatory = self.should_log_miss(kw_ok, has_date)
            if mandatory:
                logger.debug(
                    "Supabase miss mandatory logging triggered: kw_ok=%s has_date=%s",
                    kw_ok,
                    has_date,
                )
        if not mandatory:
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
        if flags:
            serialized_flags: dict[str, Any] = {}
            for key, value in flags.items():
                if value is None:
                    continue
                if isinstance(value, (int, float, bool, str)):
                    serialized_flags[key] = value
                else:
                    serialized_flags[key] = str(value)
            if serialized_flags:
                payload["flags"] = serialized_flags
        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value
        try:
            client.table("vk_misses_sample").upsert(  # type: ignore[operator]
                payload,
                on_conflict="group_id,post_id",
            ).execute()
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
        for table in ("vk_crawl_snapshots", "vk_misses_sample"):
            try:
                client.table(table).delete().lt(  # type: ignore[operator]
                    "created_at", cutoff_iso
                ).execute()
                logger.debug(
                    "Supabase retention applied: table=%s cutoff=%s", table, cutoff_iso
                )
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("Supabase export failed: %s", exc)
