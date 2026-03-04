from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import ssl
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal

from db import Database
from heavy_ops import heavy_operation
from kaggle_registry import register_job, remove_job
from ops_run import finish_ops_run, start_ops_run
from source_parsing.telegram.split_secrets import encrypt_secret
from video_announce.kaggle_client import KaggleClient

logger = logging.getLogger(__name__)

TargetKind = Literal["event", "festival", "month", "weekend", "festivals_index"]

KERNEL_PATH = Path("kaggle") / "TelegraphCacheProbe"
DEFAULT_KERNEL_REF = "zigomaro/telegraph-cache-probe"

CONFIG_DATASET_CIPHER = "telegraph-cache-probe-cipher"
CONFIG_DATASET_KEY = "telegraph-cache-probe-key"

POLL_INTERVAL_SECONDS = 20
KAGGLE_DATASET_WAIT_SECONDS = 25

# Keep datasets for debugging (disabled by default).
KEEP_DATASETS = (os.getenv("TELEGRAPH_CACHE_KEEP_DATASETS") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True)
class ProbeTarget:
    kind: TargetKind
    url: str
    ref_id: int | None = None
    ref_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "url": self.url,
            "ref_id": self.ref_id,
            "ref_key": self.ref_key,
        }


def _read_env_file_value(key: str) -> str | None:
    path = Path(".env")
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip()
    except Exception:
        return None
    return None


def _get_env_value(key: str) -> str:
    value = (os.getenv(key) or "").strip()
    if value:
        return value
    return (_read_env_file_value(key) or "").strip()


def _require_env(key: str) -> str:
    value = _get_env_value(key)
    if not value:
        raise RuntimeError(f"{key} is missing")
    return value


def _slugify(value: str, *, max_len: int = 60) -> str:
    raw = (value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = raw.strip("-")
    if not raw:
        raw = uuid.uuid4().hex[:8]
    if len(raw) > max_len:
        raw = raw[:max_len].rstrip("-")
    return raw


def _build_dataset_slug(prefix: str, run_id: str) -> str:
    safe_prefix = _slugify(prefix, max_len=40)
    safe_run = _slugify(run_id, max_len=16)
    slug = f"{safe_prefix}-{safe_run}" if safe_run else safe_prefix
    return slug[:60].rstrip("-")


def _require_kaggle_username() -> str:
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not set")
    return username


def _parse_auth_bundle(env_key: str) -> dict[str, Any] | None:
    bundle_b64 = _get_env_value(env_key)
    if not bundle_b64:
        return None
    try:
        raw = base64.urlsafe_b64decode(bundle_b64.encode("ascii")).decode("utf-8")
        bundle = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid {env_key}: {exc}") from exc
    if not isinstance(bundle, dict) or not bundle.get("session"):
        raise RuntimeError(f"Invalid {env_key}: missing session")
    return bundle


def _resolve_auth_bundle_env_key() -> str | None:
    override = (_get_env_value("TELEGRAPH_CACHE_AUTH_BUNDLE_ENV") or "").strip()
    if override:
        return override
    if _get_env_value("TELEGRAM_AUTH_BUNDLE_S22"):
        return "TELEGRAM_AUTH_BUNDLE_S22"
    if _get_env_value("TELEGRAM_AUTH_BUNDLE_E2E"):
        return "TELEGRAM_AUTH_BUNDLE_E2E"
    return None


def _build_secrets_payload() -> str:
    api_id = _get_env_value("TG_API_ID") or _get_env_value("TELEGRAM_API_ID")
    api_hash = _get_env_value("TG_API_HASH") or _get_env_value("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        raise RuntimeError("Missing TG_API_ID/TG_API_HASH (or TELEGRAM_API_ID/TELEGRAM_API_HASH)")

    bundle_env_key = _resolve_auth_bundle_env_key()
    bundle_raw = _get_env_value(bundle_env_key) if bundle_env_key else ""
    bundle_ok = False
    if bundle_raw:
        try:
            _parse_auth_bundle(bundle_env_key or "TELEGRAM_AUTH_BUNDLE_S22")
            bundle_ok = True
        except Exception:
            bundle_ok = False

    payload: dict[str, Any] = {
        "TG_API_ID": api_id,
        "TG_API_HASH": api_hash,
    }
    if bundle_raw:
        # Kaggle probe reads TELEGRAM_AUTH_BUNDLE_S22 first; map the selected bundle there.
        payload["TELEGRAM_AUTH_BUNDLE_S22"] = bundle_raw
        if bundle_ok:
            try:
                bundle = _parse_auth_bundle(bundle_env_key or "TELEGRAM_AUTH_BUNDLE_S22")
                if bundle and bundle.get("session"):
                    payload["TG_SESSION"] = bundle["session"]
            except Exception:
                pass
    else:
        # Fallback: plain StringSession (less stable fingerprint, but better than failing).
        payload["TG_SESSION"] = _get_env_value("TG_SESSION") or _get_env_value("TELEGRAM_SESSION")

    logger.info(
        "telegraph_cache.secrets bundle_env=%s bundle_len=%s bundle_ok=%s tg_session=%s",
        bundle_env_key or "-",
        len(bundle_raw) if bundle_raw else 0,
        bundle_ok,
        bool(payload.get("TG_SESSION")),
    )
    return json.dumps(payload, ensure_ascii=False)


def _create_dataset(
    client: KaggleClient,
    username: str,
    slug_suffix: str,
    title: str,
    writer,
) -> str:
    slug = f"{username}/{slug_suffix}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        writer(tmp_path)
        metadata = {
            "title": title,
            "id": slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp_path / "dataset-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            client.create_dataset(tmp_path)
        except Exception:
            logger.exception("telegraph_cache.dataset_create_failed retry=version dataset=%s", slug)
            try:
                client.create_dataset_version(
                    tmp_path,
                    version_notes=f"refresh {slug_suffix}",
                    quiet=True,
                    convert_to_csv=False,
                    dir_mode="zip",
                )
                return slug
            except Exception:
                logger.exception("telegraph_cache.dataset_version_failed retry=delete dataset=%s", slug)
            try:
                client.delete_dataset(slug, no_confirm=True)
            except Exception:
                logger.exception("telegraph_cache.dataset_delete_failed dataset=%s", slug)
            client.create_dataset(tmp_path)
    return slug


def _kernel_has_code(kernel_path: Path) -> bool:
    if not kernel_path.exists():
        return False
    meta_path = kernel_path / "kernel-metadata.json"
    code_file = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            code_file = meta.get("code_file")
        except Exception:
            code_file = None
    if code_file and (kernel_path / code_file).exists():
        return True
    return bool(list(kernel_path.glob("*.ipynb")) or list(kernel_path.glob("*.py")))


def _kernel_ref_from_meta(kernel_path: Path) -> str:
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        return DEFAULT_KERNEL_REF
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_id = meta.get("id") or meta.get("slug") or DEFAULT_KERNEL_REF
        username = (os.getenv("KAGGLE_USERNAME") or "").strip()
        if username and isinstance(kernel_id, str):
            if "/" in kernel_id:
                owner, slug = kernel_id.split("/", 1)
                if slug and owner != username:
                    return f"{username}/{slug}"
            else:
                return f"{username}/{kernel_id}"
        return str(kernel_id)
    except Exception:
        return DEFAULT_KERNEL_REF


async def _push_kernel(
    client: KaggleClient,
    dataset_sources: list[str],
) -> str:
    kernel_ref = _kernel_ref_from_meta(KERNEL_PATH)
    if _kernel_has_code(KERNEL_PATH):
        logger.info("telegraph_cache: pushing local kernel %s", KERNEL_PATH)
        await asyncio.to_thread(
            client.push_kernel,
            kernel_path=KERNEL_PATH,
            dataset_sources=dataset_sources,
        )
        return kernel_ref
    logger.info("telegraph_cache: local kernel code missing, deploying remote kernel")
    for slug in dataset_sources:
        kernel_ref = await asyncio.to_thread(client.deploy_kernel_update, kernel_ref, slug)
    return kernel_ref


async def _poll_kaggle_kernel(
    client: KaggleClient,
    kernel_ref: str,
    *,
    run_id: str | None = None,
    timeout_minutes: int,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> tuple[str, dict | None, float]:
    started = time.monotonic()
    timeout_minutes = max(1, int(timeout_minutes))
    deadline = started + timeout_minutes * 60
    last_status: dict | None = None
    consecutive_poll_errors = 0
    attempt = 0

    async def _notify(phase: str, status: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, status)
        except Exception:
            logger.exception("telegraph_cache: status callback failed phase=%s", phase)

    await _notify("poll", {"_poll_timeout_minutes": timeout_minutes, "_elapsed_seconds": 0.0})
    while time.monotonic() < deadline:
        attempt += 1
        try:
            status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            last_status = status
            consecutive_poll_errors = 0
        except Exception as exc:
            consecutive_poll_errors += 1
            is_ssl = isinstance(exc, ssl.SSLError) or exc.__class__.__name__.endswith("SSLError")
            is_conn = isinstance(exc, ConnectionError) or exc.__class__.__name__.endswith("ConnectionError")
            is_timeout = exc.__class__.__name__.endswith("Timeout")
            if not (is_ssl or is_conn or is_timeout):
                raise
            msg = str(exc) or repr(exc)
            if len(msg) > 280:
                msg = msg[:277] + "..."
            logger.warning(
                "telegraph_cache.kernel_poll_transient_error run_id=%s kernel_ref=%s attempt=%s consecutive=%s err=%s",
                run_id,
                kernel_ref,
                attempt,
                consecutive_poll_errors,
                msg,
            )
            shown_state = ""
            if isinstance(last_status, dict):
                shown_state = str(last_status.get("status") or "").upper()
            await _notify(
                "poll_error",
                {
                    "status": shown_state or "RUNNING",
                    "failureMessage": f"⚠️ временная ошибка Kaggle API (SSL/сеть), продолжаю опрос: {msg}",
                    "_poll_timeout_minutes": timeout_minutes,
                    "_elapsed_seconds": time.monotonic() - started,
                },
            )
            await asyncio.sleep(min(60, 2 ** min(consecutive_poll_errors, 5)))
            continue

        state = str((status or {}).get("status") or "").upper()
        payload = dict(status or {})
        payload["_poll_timeout_minutes"] = timeout_minutes
        payload["_elapsed_seconds"] = time.monotonic() - started
        await _notify("poll", payload)
        logger.info(
            "telegraph_cache.kernel_poll run_id=%s kernel_ref=%s attempt=%s status=%s elapsed=%.1fs",
            run_id,
            kernel_ref,
            attempt,
            state or "UNKNOWN",
            time.monotonic() - started,
        )
        if state == "COMPLETE":
            done = dict(last_status or {})
            done["_poll_timeout_minutes"] = timeout_minutes
            done["_elapsed_seconds"] = time.monotonic() - started
            await _notify("complete", done)
            return "complete", last_status, time.monotonic() - started
        if state in ("ERROR", "FAILED", "CANCELLED"):
            failed = dict(last_status or {})
            failed["_poll_timeout_minutes"] = timeout_minutes
            failed["_elapsed_seconds"] = time.monotonic() - started
            await _notify("failed", failed)
            return "failed", last_status, time.monotonic() - started
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    # Final status fetch near deadline (avoid false timeouts).
    try:
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status or last_status
        state = str((status or {}).get("status") or "").upper()
        if state == "COMPLETE":
            done = dict(last_status or {})
            done["_poll_timeout_minutes"] = timeout_minutes
            done["_elapsed_seconds"] = time.monotonic() - started
            await _notify("complete", done)
            return "complete", last_status, time.monotonic() - started
    except Exception:
        pass

    timeout_payload = dict(last_status or {})
    timeout_payload.setdefault(
        "failureMessage",
        "локальный таймаут ожидания: Kaggle kernel мог продолжить выполнение",
    )
    timeout_payload["_poll_timeout_minutes"] = timeout_minutes
    timeout_payload["_elapsed_seconds"] = time.monotonic() - started
    await _notify("timeout", timeout_payload)
    return "timeout", last_status, time.monotonic() - started


def _format_kaggle_phase(phase: str) -> str:
    labels = {
        "prepare": "подготовка",
        "pushed": "запуск в Kaggle",
        "poll": "выполнение",
        "poll_error": "временная ошибка сети",
        "complete": "завершено",
        "failed": "ошибка",
        "timeout": "таймаут",
    }
    return labels.get(phase, phase)


def _extract_kaggle_failure_message(status: dict | None) -> str:
    if not status:
        return ""
    for key in (
        "failureMessage",
        "failure_message",
        "errorMessage",
        "error_message",
        "error",
        "message",
    ):
        value = status.get(key)
        if value:
            return str(value)
    return ""


def _format_kaggle_status(status: dict | None) -> str:
    if not status:
        return "неизвестен"
    state = status.get("status")
    if not state:
        return "неизвестен"
    failure = _extract_kaggle_failure_message(status)
    if failure and len(failure) > 400:
        failure = failure[:397] + "..."
    result = str(state)
    if failure:
        result += f" ({failure})"
    return result


def _format_kaggle_status_message(
    phase: str,
    kernel_ref: str,
    status: dict | None,
    *,
    run_id: str | None = None,
) -> str:
    lines = [
        "🛰️ Kaggle: Telegraph cache sanitizer",
        f"Kernel: {kernel_ref or '—'}",
        f"Этап: {_format_kaggle_phase(phase)}",
    ]
    if run_id:
        lines.append(f"run_id: {run_id}")
    if status is not None:
        lines.append(f"Статус Kaggle: {_format_kaggle_status(status)}")
        timeout_minutes = status.get("_poll_timeout_minutes")
        elapsed_seconds = status.get("_elapsed_seconds")
        try:
            timeout_i = int(timeout_minutes) if timeout_minutes is not None else None
        except Exception:
            timeout_i = None
        try:
            elapsed_f = float(elapsed_seconds) if elapsed_seconds is not None else None
        except Exception:
            elapsed_f = None
        if timeout_i:
            if elapsed_f is not None:
                lines.append(f"Ожидание: до {timeout_i} мин (прошло {elapsed_f / 60.0:.1f} мин)")
            else:
                lines.append(f"Ожидание: до {timeout_i} мин")
    return "\n".join(lines)


async def _download_results(
    client: KaggleClient,
    kernel_ref: str,
    run_id: str,
) -> Path:
    output_dir = Path(tempfile.gettempdir()) / f"telegraph-cache-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        files = await asyncio.to_thread(
            client.download_kernel_output, kernel_ref, path=str(output_dir), force=True
        )
        for name in files:
            if Path(name).name == "telegraph_cache_report.json":
                return output_dir / name
        if attempt < max_attempts:
            await asyncio.sleep(6)
    raise RuntimeError("telegraph_cache_report.json not found in Kaggle output")


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _norm_url(url: str | None) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        return "https:" + raw
    return raw


def _telegraph_url_from_path(path: str | None) -> str:
    p = str(path or "").strip().lstrip("/")
    return f"https://telegra.ph/{p}" if p else ""


def _read_bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int, *, min_value: int = 0, max_value: int = 10_000) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        val = int(raw)
    except Exception:
        return int(default)
    return max(min_value, min(max_value, val))


def _date_iso(d: date) -> str:
    return d.isoformat()


async def collect_probe_targets(
    db: Database,
    *,
    days_back: int,
    days_forward: int,
    limit_events: int,
    limit_festivals: int,
    include_month_pages: bool,
    include_weekend_pages: bool,
    include_festival_pages: bool,
    include_festivals_index: bool,
) -> tuple[list[ProbeTarget], dict[str, Any]]:
    today = date.today()
    start = today - timedelta(days=max(0, int(days_back)))
    end = today + timedelta(days=max(0, int(days_forward)))
    start_iso = _date_iso(start)
    end_iso = _date_iso(end)
    today_iso = _date_iso(today)
    current_month = today.strftime("%Y-%m")

    targets: list[ProbeTarget] = []
    meta: dict[str, Any] = {
        "limit_events": int(max(0, int(limit_events))),
        "limit_festivals": int(max(0, int(limit_festivals))),
        "eligible_event_pages": 0,
        "selected_event_pages": 0,
        "selected_total_targets": 0,
    }

    # Events (ongoing/upcoming within window).
    #
    # Important: do not always probe only the earliest upcoming events, otherwise the
    # sanitizer gets "stuck" around a stable LIMIT window (e.g. ~200 targets) and never
    # reaches other future pages. We rotate by the last probe timestamp (oldest/never
    # checked first) while still prioritizing persistently failing pages.
    async with db.raw_conn() as conn:
        rows = await conn.execute(
            """
            SELECT
              id,
              telegraph_url,
              telegraph_path,
              title,
              festival,
              substr(date, 1, 10) AS start_date,
              COALESCE(
                end_date,
                CASE
                  WHEN instr(date, '..') > 0 THEN substr(date, instr(date, '..') + 2, 10)
                  ELSE substr(date, 1, 10)
                END
              ) AS end_date
            FROM event
            WHERE lifecycle_status = 'active'
              AND (silent IS NULL OR silent = 0)
              AND (telegraph_url IS NOT NULL OR telegraph_path IS NOT NULL)
              AND substr(date, 1, 10) <= ?
              AND COALESCE(
                end_date,
                CASE
                  WHEN instr(date, '..') > 0 THEN substr(date, instr(date, '..') + 2, 10)
                  ELSE substr(date, 1, 10)
                END
              ) >= ?
            ORDER BY substr(date, 1, 10) ASC, time ASC, id ASC
            """,
            (end_iso, today_iso),
        )
        all_event_rows = list(await rows.fetchall())

        # Load probe metadata for rotation/prioritization.
        probe_meta: dict[str, tuple[str | None, int, int]] = {}
        try:
            cur_meta = await conn.execute(
                """
                SELECT url, last_checked_at, consecutive_failures, last_ok
                FROM telegraph_preview_probe
                WHERE kind = 'event'
                """
            )
            for url, last_checked_at, streak, last_ok in await cur_meta.fetchall():
                u = str(url or "").strip()
                if not u:
                    continue
                probe_meta[u] = (
                    str(last_checked_at).strip() if last_checked_at else None,
                    int(streak or 0),
                    int(last_ok or 0),
                )
        except Exception:
            probe_meta = {}

        def _checked_key(ts: str | None) -> str:
            s = (ts or "").strip()
            return s if s else "0000-01-01 00:00:00"

        scored: list[tuple[int, str, str, str, int, int, str]] = []
        for event_id, telegraph_url, telegraph_path, title, fest, start_d, end_d in all_event_rows:
            url = _norm_url(telegraph_url) or _telegraph_url_from_path(telegraph_path)
            if not url.startswith(("http://", "https://")):
                continue
            last_checked_at, streak, last_ok = probe_meta.get(url, (None, 0, 0))
            # Prioritize failing pages, then rotate by last check time (oldest/never checked first).
            scored.append(
                (
                    int(streak) if int(last_ok or 0) == 0 else 0,
                    _checked_key(last_checked_at),
                    str(start_d or ""),
                    str(end_d or ""),
                    int(event_id),
                    0,  # reserved for future: secondary ranking
                    url,
                )
            )

        scored.sort(key=lambda t: (-t[0], t[1], t[2], t[4], t[6]))
        meta["eligible_event_pages"] = int(len(scored))
        id_to_title: dict[int, str] = {}
        for r in all_event_rows:
            try:
                rid = int(r[0])
            except Exception:
                continue
            id_to_title[rid] = str(r[3] or "").strip()

        for _streak, _checked, _start_d, _end_d, event_id, _rank2, url in scored[: max(0, int(limit_events))]:
            ref_key = id_to_title.get(int(event_id)) or None
            targets.append(ProbeTarget(kind="event", url=str(url), ref_id=int(event_id), ref_key=ref_key))
        meta["selected_event_pages"] = int(len(targets))

    # Festivals index page (landing).
    if include_festivals_index:
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                """
                SELECT key, value
                FROM setting
                WHERE key IN ('festivals_index_url', 'fest_index_url', 'festivals_index_path', 'fest_index_path')
                """
            )
            kv = {str(k): str(v) for k, v in await cur.fetchall() if k and v}
        url = _norm_url(kv.get("festivals_index_url")) or _norm_url(kv.get("fest_index_url"))
        if not url:
            path = str(kv.get("festivals_index_path") or kv.get("fest_index_path") or "").strip()
            url = _telegraph_url_from_path(path)
        if url.startswith(("http://", "https://")):
            targets.append(ProbeTarget(kind="festivals_index", url=url, ref_key="festivals_index"))

    # Festival pages (only for festivals that still have upcoming events).
    if include_festival_pages:
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                """
                SELECT id, name, telegraph_url, telegraph_path, COALESCE(end_date, start_date, '') AS end_hint
                FROM festival
                WHERE (telegraph_url IS NOT NULL OR telegraph_path IS NOT NULL)
                ORDER BY COALESCE(end_date, start_date, '9999-12-31') ASC, id ASC
                LIMIT ?
                """,
                (max(0, int(limit_festivals)),),
            )
            festivals = await cur.fetchall()

            for fest_id, fest_name, telegraph_url, telegraph_path, end_hint in festivals:
                # Find an owner event id for regen via JobOutbox festival_pages handler.
                owner_event_id = None
                try:
                    cur2 = await conn.execute(
                        """
                        SELECT id
                        FROM event
                        WHERE festival = ?
                          AND lifecycle_status = 'active'
                          AND (silent IS NULL OR silent = 0)
                          AND COALESCE(
                            end_date,
                            CASE
                              WHEN instr(date, '..') > 0 THEN substr(date, instr(date, '..') + 2, 10)
                              ELSE substr(date, 1, 10)
                            END
                          ) >= ?
                        ORDER BY substr(date, 1, 10) ASC, time ASC, id ASC
                        LIMIT 1
                        """,
                        (str(fest_name or "").strip(), today_iso),
                    )
                    row2 = await cur2.fetchone()
                    if row2:
                        owner_event_id = int(row2[0])
                except Exception:
                    owner_event_id = None

                # Skip past festivals (no active/upcoming events => no operator value).
                if owner_event_id is None:
                    continue

                url = _norm_url(telegraph_url) or _telegraph_url_from_path(telegraph_path)
                if not url.startswith(("http://", "https://")):
                    continue
                ref_key = (
                    f"{int(fest_id)}:{str(fest_name or '').strip()}"
                    if fest_id is not None
                    else str(fest_name or "").strip() or None
                )
                targets.append(
                    ProbeTarget(
                        kind="festival",
                        url=url,
                        ref_id=owner_event_id,
                        ref_key=ref_key,
                    )
                )

    # Month/weekend pages in the same window: probe existing stored URLs (best-effort).
    if include_month_pages or include_weekend_pages:
        months: set[str] = set()
        weekends: set[str] = set()
        # Derive month/weekend keys from the full eligible event window to avoid probing years of history
        # while still covering pages outside the current LIMIT rotation slice.
        rows = [(r[5], r[6]) for r in (all_event_rows or []) if len(r) >= 7]

        def _weekend_start(iso: str) -> str | None:
            try:
                d = date.fromisoformat(str(iso))
            except Exception:
                return None
            if d.weekday() < 5:
                return None
            # Saturday start
            sat = d - timedelta(days=(d.weekday() - 5))
            return sat.isoformat()

        for start_d, end_d in rows:
            try:
                d0 = date.fromisoformat(str(start_d))
                d1 = date.fromisoformat(str(end_d))
            except Exception:
                continue
            # Clamp long-running/multi-month events to the selected probe window.
            # Otherwise a single exhibition spanning months can explode the set of derived
            # month/weekend pages (including past weekends) even when `days_back` is small.
            if d0 < today:
                d0 = today
            if d1 > end:
                d1 = end
            if d0 > d1:
                continue
            cur_day = d0
            while cur_day <= d1:
                months.add(cur_day.strftime("%Y-%m"))
                w = _weekend_start(cur_day.isoformat())
                if w:
                    weekends.add(w)
                cur_day += timedelta(days=1)

        # Keep only current/future month pages.
        months = {m for m in months if str(m) >= current_month}

        # Keep only current/future weekend pages (do not probe past weekends).
        filtered_weekends: set[str] = set()
        for w in weekends:
            try:
                ws = date.fromisoformat(str(w))
            except Exception:
                continue
            # Weekend is considered past after Sunday ends.
            if (ws + timedelta(days=1)) < today:
                continue
            filtered_weekends.add(ws.isoformat())
        weekends = filtered_weekends

    if include_month_pages and months:
        async with db.raw_conn() as conn:
            placeholders = ",".join("?" for _ in sorted(months))
            try:
                cur = await conn.execute(
                    f"SELECT month, part_number, url FROM monthpagepart WHERE month IN ({placeholders})",
                    tuple(sorted(months)),
                )
                for month_key, part_number, url in await cur.fetchall():
                    u = _norm_url(url)
                    if not u.startswith(("http://", "https://")):
                        continue
                    ref_key = f"{month_key}"
                    targets.append(
                        ProbeTarget(kind="month", url=u, ref_key=ref_key, ref_id=None)
                    )
            except Exception:
                # monthpagepart may be missing in older snapshots; fall back to legacy monthpage.
                cur = await conn.execute(
                    f"SELECT month, url, url2 FROM monthpage WHERE month IN ({placeholders})",
                    tuple(sorted(months)),
                )
                for month_key, url1, url2 in await cur.fetchall():
                    for u0 in [url1, url2]:
                        u = _norm_url(u0)
                        if not u.startswith(("http://", "https://")):
                            continue
                        targets.append(ProbeTarget(kind="month", url=u, ref_key=str(month_key)))

    if include_weekend_pages and weekends:
        async with db.raw_conn() as conn:
            placeholders = ",".join("?" for _ in sorted(weekends))
            cur = await conn.execute(
                f"SELECT start, url FROM weekendpage WHERE start IN ({placeholders})",
                tuple(sorted(weekends)),
            )
            for start_key, url in await cur.fetchall():
                u = _norm_url(url)
                if not u.startswith(("http://", "https://")):
                    continue
                targets.append(ProbeTarget(kind="weekend", url=u, ref_key=str(start_key)))

    # Ensure deterministic order + uniqueness by URL.
    seen: set[str] = set()
    out: list[ProbeTarget] = []
    for t in targets:
        key = str(t.url).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t)
    meta["selected_total_targets"] = int(len(out))
    return out, meta


def _build_config_payload(
    *,
    run_id: str,
    targets: list[ProbeTarget],
    repeats: int,
    attach_wait_sec: int,
    per_url_timeout_sec: float,
    delete_messages: bool,
    delay_min_sec: float,
    delay_max_sec: float,
    repeat_pause_min_sec: float,
    repeat_pause_max_sec: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "targets": [t.as_dict() for t in (targets or [])],
        "settings": {
            "repeats": int(max(1, repeats)),
            "attach_wait_sec": int(max(1, attach_wait_sec)),
            "per_url_timeout_sec": float(max(3.0, per_url_timeout_sec)),
            "delete_messages": bool(delete_messages),
            "delay_min_sec": float(max(0.0, delay_min_sec)),
            "delay_max_sec": float(max(0.0, delay_max_sec)),
            "repeat_pause_min_sec": float(max(0.0, repeat_pause_min_sec)),
            "repeat_pause_max_sec": float(max(0.0, repeat_pause_max_sec)),
        },
    }
    return payload


async def _prepare_kaggle_datasets(
    client: KaggleClient,
    *,
    config_payload: dict[str, Any],
    secrets_payload: str,
    run_id: str,
) -> tuple[str, str]:
    encrypted, fernet_key = encrypt_secret(secrets_payload)
    username = _require_kaggle_username()
    slug_suffix = _slugify(run_id, max_len=16)

    def write_cipher(path: Path) -> None:
        (path / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (path / "secrets.enc").write_bytes(encrypted)

    def write_key(path: Path) -> None:
        (path / "fernet.key").write_bytes(fernet_key)

    slug_cipher = _create_dataset(
        client,
        username,
        _build_dataset_slug(CONFIG_DATASET_CIPHER, run_id),
        f"Telegraph Cache Probe Cipher {slug_suffix}",
        write_cipher,
    )
    slug_key = _create_dataset(
        client,
        username,
        _build_dataset_slug(CONFIG_DATASET_KEY, run_id),
        f"Telegraph Cache Probe Key {slug_suffix}",
        write_key,
    )
    return slug_cipher, slug_key


async def _cleanup_datasets(dataset_slugs: list[str]) -> None:
    if KEEP_DATASETS:
        logger.info("telegraph_cache.datasets_kept slugs=%s", dataset_slugs)
        return
    client = KaggleClient()
    for slug in dataset_slugs:
        if not slug:
            continue
        try:
            logger.info("telegraph_cache.dataset_delete slug=%s", slug)
            await asyncio.to_thread(client.delete_dataset, slug, no_confirm=True)
        except Exception:
            logger.exception("telegraph_cache.dataset_delete_failed slug=%s", slug)


async def import_probe_results(
    db: Database,
    *,
    report: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Upsert Kaggle probe results into SQLite and return summary metrics."""
    now = now or datetime.now(timezone.utc)
    checked_at_sql = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    results = list(report.get("results") or [])
    total = len(results)
    ok = sum(1 for r in results if r and r.get("ok"))
    fail = total - ok
    cached_ok = sum(1 for r in results if r and r.get("has_cached_page"))
    photo_ok = sum(1 for r in results if r and r.get("has_photo"))
    both_ok = sum(1 for r in results if r and r.get("has_cached_page") and r.get("has_photo"))
    warn_no_photo = sum(1 for r in results if r and r.get("has_cached_page") and not r.get("has_photo"))

    async with db.raw_conn() as conn:
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            kind = str(item.get("kind") or "").strip() or "unknown"
            ref_id = _safe_int(item.get("ref_id"))
            ref_key = str(item.get("ref_key") or "").strip() or None
            last_ok = 1 if bool(item.get("ok")) else 0
            last_cached = 1 if bool(item.get("has_cached_page")) else 0
            last_photo = 1 if bool(item.get("has_photo")) else 0
            title = str(item.get("title") or "").strip() or None
            site_name = str(item.get("site_name") or "").strip() or None
            err = str(item.get("error") or "").strip() or None

            await conn.execute(
                """
                INSERT INTO telegraph_preview_probe(
                  url,
                  kind,
                  ref_id,
                  ref_key,
                  last_checked_at,
                  last_ok,
                  last_has_cached_page,
                  last_has_photo,
                  last_title,
                  last_site_name,
                  last_error,
                  total_checks,
                  total_ok,
                  consecutive_failures,
                  last_ok_at,
                  last_fail_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                  kind=excluded.kind,
                  ref_id=excluded.ref_id,
                  ref_key=excluded.ref_key,
                  last_checked_at=excluded.last_checked_at,
                  last_ok=excluded.last_ok,
                  last_has_cached_page=excluded.last_has_cached_page,
                  last_has_photo=excluded.last_has_photo,
                  last_title=excluded.last_title,
                  last_site_name=excluded.last_site_name,
                  last_error=excluded.last_error,
                  total_checks=telegraph_preview_probe.total_checks + 1,
                  total_ok=telegraph_preview_probe.total_ok + excluded.last_ok,
                  consecutive_failures=CASE
                    WHEN excluded.last_ok = 1 THEN 0
                    ELSE telegraph_preview_probe.consecutive_failures + 1
                  END,
                  last_ok_at=CASE
                    WHEN excluded.last_ok = 1 THEN excluded.last_checked_at
                    ELSE telegraph_preview_probe.last_ok_at
                  END,
                  last_fail_at=CASE
                    WHEN excluded.last_ok = 1 THEN telegraph_preview_probe.last_fail_at
                    ELSE excluded.last_checked_at
                  END
                """,
                (
                    url,
                    kind,
                    ref_id,
                    ref_key,
                    checked_at_sql,
                    last_ok,
                    last_cached,
                    last_photo,
                    title,
                    site_name,
                    err,
                    last_ok,
                    0 if last_ok else 1,
                    checked_at_sql if last_ok else None,
                    checked_at_sql if not last_ok else None,
                ),
            )
        await conn.commit()

    return {
        "total": total,
        "ok": ok,
        "fail": fail,
        "cached_ok": cached_ok,
        "photo_ok": photo_ok,
        "both_ok": both_ok,
        "warn_no_photo": warn_no_photo,
        "checked_at": checked_at_sql,
    }


async def _load_enqueued_regen_candidates(
    db: Database,
    *,
    min_consecutive_failures: int,
    min_last_checked_at: str | None = None,
    kinds: Iterable[TargetKind] | None = None,
) -> list[tuple[str, str, int | None, str | None, int]]:
    """Return failing probe rows eligible for regeneration enqueue.

    Returns list of (url, kind, ref_id, ref_key, consecutive_failures).
    """
    min_consecutive_failures = max(1, int(min_consecutive_failures))
    kinds_set = {str(k) for k in (kinds or []) if str(k)}
    where_kind = ""
    where_checked = ""
    params: list[Any] = [min_consecutive_failures]
    if min_last_checked_at:
        where_checked = " AND last_checked_at >= ?"
        params.append(str(min_last_checked_at))
    if kinds_set:
        placeholders = ",".join("?" for _ in sorted(kinds_set))
        where_kind = f" AND kind IN ({placeholders})"
        params.extend(sorted(kinds_set))

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            f"""
            SELECT url, kind, ref_id, ref_key, consecutive_failures
            FROM telegraph_preview_probe
            WHERE last_ok = 0
              AND consecutive_failures >= ?
              {where_checked}
              {where_kind}
            ORDER BY consecutive_failures DESC, last_checked_at DESC
            """,
            tuple(params),
        )
        return [
            (str(url), str(kind), _safe_int(ref_id), str(ref_key) if ref_key else None, int(streak or 0))
            for url, kind, ref_id, ref_key, streak in await cur.fetchall()
            if url and kind
        ]


async def enqueue_regeneration_for_failures(
    db: Database,
    *,
    min_consecutive_failures: int,
    min_last_checked_at: str | None = None,
    include_festivals_index: bool,
) -> dict[str, int]:
    """Enqueue JobOutbox rebuild tasks for persistently failing pages."""
    from main import enqueue_job  # local import to avoid import cycles
    from models import JobTask

    rows = await _load_enqueued_regen_candidates(
        db,
        min_consecutive_failures=min_consecutive_failures,
        min_last_checked_at=min_last_checked_at,
        kinds=["event", "festival", "month", "weekend", "festivals_index"],
    )
    enq = {"event": 0, "festival": 0, "month": 0, "weekend": 0, "festivals_index": 0}

    today = date.today()
    today_iso = today.isoformat()
    current_month = today.strftime("%Y-%m")

    # Filter out past events/festivals (ended before today) so sanitizer never rebuilds old pages.
    event_ids: set[int] = {
        int(ref_id)
        for _url, kind, ref_id, _ref_key, _streak in rows
        if kind in {"event", "festival"} and ref_id
    }
    eligible_event_ids: set[int] = set()
    if event_ids:
        async with db.raw_conn() as conn:
            placeholders = ",".join("?" for _ in sorted(event_ids))
            cur = await conn.execute(
                f"""
                SELECT id
                FROM event
                WHERE id IN ({placeholders})
                  AND lifecycle_status = 'active'
                  AND (silent IS NULL OR silent = 0)
                  AND COALESCE(
                    end_date,
                    CASE
                      WHEN instr(date, '..') > 0 THEN substr(date, instr(date, '..') + 2, 10)
                      ELSE substr(date, 1, 10)
                    END
                  ) >= ?
                """,
                tuple(sorted(event_ids)) + (today_iso,),
            )
            eligible_event_ids = {int(r[0]) for r in await cur.fetchall() if r and r[0] is not None}

    for _url, kind, ref_id, ref_key, _streak in rows:
        if kind == "event" and ref_id and int(ref_id) in eligible_event_ids:
            await enqueue_job(db, int(ref_id), JobTask.telegraph_build)
            enq["event"] += 1
        elif kind == "festival" and ref_id and int(ref_id) in eligible_event_ids:
            await enqueue_job(db, int(ref_id), JobTask.festival_pages)
            enq["festival"] += 1
        elif kind == "month" and ref_key and str(ref_key) >= current_month:
            await enqueue_job(
                db,
                0,
                JobTask.month_pages,
                coalesce_key=f"month_pages:{ref_key}",
                depends_on=[],
            )
            enq["month"] += 1
        elif kind == "weekend" and ref_key:
            try:
                ws = date.fromisoformat(str(ref_key))
            except Exception:
                ws = None
            if ws is None or (ws + timedelta(days=1)) < today:
                continue
            await enqueue_job(
                db,
                0,
                JobTask.weekend_pages,
                coalesce_key=f"weekend_pages:{ref_key}",
                depends_on=[],
            )
            enq["weekend"] += 1
        elif kind == "festivals_index" and include_festivals_index:
            # Direct rebuild (no JobOutbox task exists yet).
            try:
                from main import rebuild_festivals_index_if_needed

                await rebuild_festivals_index_if_needed(db, force=True)
                enq["festivals_index"] += 1
            except Exception:
                logger.warning("telegraph_cache: failed to rebuild festivals index", exc_info=True)

    return enq


async def run_telegraph_cache_sanitizer(
    db: Database,
    *,
    bot: Any | None,
    chat_id: int | None,
    operator_id: int | None,
    trigger: str,
    run_id: str | None = None,
    send_progress: bool | None = None,
    status_message_id: int | None = None,
    days_back: int | None = None,
    days_forward: int | None = None,
    limit_events: int | None = None,
    limit_festivals: int | None = None,
    include_month_pages: bool | None = None,
    include_weekend_pages: bool | None = None,
    include_festival_pages: bool | None = None,
    include_festivals_index: bool | None = None,
    repeats: int | None = None,
    attach_wait_sec: int | None = None,
    per_url_timeout_sec: float | None = None,
    delete_messages: bool | None = None,
    delay_min_sec: float | None = None,
    delay_max_sec: float | None = None,
    repeat_pause_min_sec: float | None = None,
    repeat_pause_max_sec: float | None = None,
    kaggle_timeout_minutes: int | None = None,
    regen_min_consecutive_failures: int | None = None,
    enqueue_regen: bool | None = None,
) -> dict[str, Any]:
    """Run Telegraph cache sanitizer via Kaggle/Telethon and persist stats in SQLite."""

    run_id = (str(run_id).strip() if run_id else "") or uuid.uuid4().hex[:12]
    trigger = str(trigger or "").strip() or "manual"
    if send_progress is None:
        send_progress = trigger == "manual"
    send_progress = bool(send_progress)
    kaggle_status_message_id: int | None = int(status_message_id) if status_message_id else None
    kaggle_kernel_ref = _kernel_ref_from_meta(KERNEL_PATH)

    async def _update_kaggle_status(phase: str, kernel_ref: str, status: dict | None) -> None:
        nonlocal kaggle_status_message_id, kaggle_kernel_ref
        if not (send_progress and bot and chat_id):
            return
        if kernel_ref:
            kaggle_kernel_ref = kernel_ref
        text = _format_kaggle_status_message(phase, kaggle_kernel_ref, status, run_id=run_id)
        try:
            if kaggle_status_message_id is None:
                sent = await bot.send_message(chat_id, text, disable_web_page_preview=True)
                kaggle_status_message_id = getattr(sent, "message_id", None)
            else:
                await bot.edit_message_text(
                    text=text,
                    chat_id=chat_id,
                    message_id=kaggle_status_message_id,
                    disable_web_page_preview=True,
                )
        except Exception:
            logger.exception("telegraph_cache: failed to update kaggle status")

    days_back = int(days_back) if days_back is not None else _read_int_env("TELEGRAPH_CACHE_DAYS_BACK", 7, max_value=365)
    days_forward = int(days_forward) if days_forward is not None else _read_int_env("TELEGRAPH_CACHE_DAYS_FORWARD", 120, max_value=3650)
    limit_events = int(limit_events) if limit_events is not None else _read_int_env("TELEGRAPH_CACHE_LIMIT_EVENTS", 180, max_value=2000)
    limit_festivals = int(limit_festivals) if limit_festivals is not None else _read_int_env("TELEGRAPH_CACHE_LIMIT_FESTIVALS", 80, max_value=2000)
    include_month_pages = bool(include_month_pages) if include_month_pages is not None else _read_bool_env("TELEGRAPH_CACHE_INCLUDE_MONTHS", True)
    include_weekend_pages = bool(include_weekend_pages) if include_weekend_pages is not None else _read_bool_env("TELEGRAPH_CACHE_INCLUDE_WEEKENDS", True)
    include_festival_pages = bool(include_festival_pages) if include_festival_pages is not None else _read_bool_env("TELEGRAPH_CACHE_INCLUDE_FESTIVALS", True)
    include_festivals_index = bool(include_festivals_index) if include_festivals_index is not None else _read_bool_env("TELEGRAPH_CACHE_INCLUDE_FESTIVALS_INDEX", True)
    repeats = int(repeats) if repeats is not None else _read_int_env("TELEGRAPH_CACHE_REPEATS", 2, min_value=1, max_value=5)
    attach_wait_sec = int(attach_wait_sec) if attach_wait_sec is not None else _read_int_env("TELEGRAPH_CACHE_ATTACH_WAIT_SEC", 20, min_value=3, max_value=120)
    per_url_timeout_sec = float(per_url_timeout_sec) if per_url_timeout_sec is not None else float(os.getenv("TELEGRAPH_CACHE_PER_URL_TIMEOUT_SEC") or "35")
    delete_messages = bool(delete_messages) if delete_messages is not None else _read_bool_env("TELEGRAPH_CACHE_DELETE_MESSAGES", True)
    delay_min_sec = float(delay_min_sec) if delay_min_sec is not None else float(os.getenv("TELEGRAPH_CACHE_DELAY_MIN_SEC") or "1.0")
    delay_max_sec = float(delay_max_sec) if delay_max_sec is not None else float(os.getenv("TELEGRAPH_CACHE_DELAY_MAX_SEC") or "2.2")
    repeat_pause_min_sec = float(repeat_pause_min_sec) if repeat_pause_min_sec is not None else float(os.getenv("TELEGRAPH_CACHE_REPEAT_PAUSE_MIN_SEC") or "1.5")
    repeat_pause_max_sec = float(repeat_pause_max_sec) if repeat_pause_max_sec is not None else float(os.getenv("TELEGRAPH_CACHE_REPEAT_PAUSE_MAX_SEC") or "3.5")
    kaggle_timeout_minutes = int(kaggle_timeout_minutes) if kaggle_timeout_minutes is not None else _read_int_env("TELEGRAPH_CACHE_KAGGLE_TIMEOUT_MIN", 35, min_value=5, max_value=360)
    regen_min_consecutive_failures = int(regen_min_consecutive_failures) if regen_min_consecutive_failures is not None else _read_int_env("TELEGRAPH_CACHE_REGEN_AFTER_RUNS", 2, min_value=1, max_value=10)
    enqueue_regen = bool(enqueue_regen) if enqueue_regen is not None else _read_bool_env("TELEGRAPH_CACHE_ENQUEUE_REGEN", True)

    ops_id = await start_ops_run(
        db,
        kind="telegraph_cache_sanitize",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={
            "run_id": run_id,
            "days_back": days_back,
            "days_forward": days_forward,
            "limit_events": limit_events,
            "include_month_pages": include_month_pages,
            "include_weekend_pages": include_weekend_pages,
            "include_festival_pages": include_festival_pages,
            "include_festivals_index": include_festivals_index,
        },
    )
    ops_status = "success"
    ops_err: str | None = None
    report_path: str | None = None
    kernel_ref: str | None = None
    dataset_slugs: list[str] = []
    imported_summary: dict[str, Any] = {}
    regen_summary: dict[str, Any] = {}
    try:
        async with heavy_operation(kind="telegraph_cache_sanitize", trigger=trigger, mode="wait", run_id=run_id, operator_id=operator_id, chat_id=chat_id):
            targets, targets_meta = await collect_probe_targets(
                db,
                days_back=days_back,
                days_forward=days_forward,
                limit_events=limit_events,
                limit_festivals=limit_festivals,
                include_month_pages=include_month_pages,
                include_weekend_pages=include_weekend_pages,
                include_festival_pages=include_festival_pages,
                include_festivals_index=include_festivals_index,
            )
            if not targets:
                imported_summary = {"total": 0, "ok": 0, "fail": 0}
                return {
                    "run_id": run_id,
                    "targets": 0,
                    "imported": imported_summary,
                    "regen": regen_summary,
                    "targets_meta": targets_meta,
                }

            await _update_kaggle_status("prepare", kaggle_kernel_ref, None)
            config = _build_config_payload(
                run_id=run_id,
                targets=targets,
                repeats=repeats,
                attach_wait_sec=attach_wait_sec,
                per_url_timeout_sec=per_url_timeout_sec,
                delete_messages=delete_messages,
                delay_min_sec=delay_min_sec,
                delay_max_sec=delay_max_sec,
                repeat_pause_min_sec=repeat_pause_min_sec,
                repeat_pause_max_sec=repeat_pause_max_sec,
            )
            secrets_payload = _build_secrets_payload()

            client = KaggleClient()
            slug_cipher, slug_key = await _prepare_kaggle_datasets(
                client,
                config_payload=config,
                secrets_payload=secrets_payload,
                run_id=run_id,
            )
            dataset_slugs = [slug_cipher, slug_key]
            await asyncio.sleep(KAGGLE_DATASET_WAIT_SECONDS)

            kernel_ref = await _push_kernel(client, dataset_sources=[slug_cipher, slug_key])
            await _update_kaggle_status("pushed", kernel_ref, None)
            try:
                await register_job(
                    "telegraph_cache_probe",
                    kernel_ref,
                    meta={"run_id": run_id, "chat_id": chat_id, "pid": os.getpid()},
                )
            except Exception:
                logger.debug("telegraph_cache: register_job failed (non-fatal)", exc_info=True)

            final_status, status_data, duration = await _poll_kaggle_kernel(
                client,
                kernel_ref,
                run_id=run_id,
                timeout_minutes=kaggle_timeout_minutes,
                status_callback=_update_kaggle_status,
            )
            if final_status != "complete":
                failure = ""
                if status_data:
                    failure = str(status_data.get("failureMessage") or status_data.get("failure_message") or "")
                raise RuntimeError(f"Kaggle kernel failed ({final_status}) {failure}".strip())

            report_file = await _download_results(client, kernel_ref, run_id)
            report_path = str(report_file)
            report = json.loads(report_file.read_text(encoding="utf-8"))
            imported_summary = await import_probe_results(db, report=report)

            if enqueue_regen:
                regen_summary = await enqueue_regeneration_for_failures(
                    db,
                    min_consecutive_failures=regen_min_consecutive_failures,
                    min_last_checked_at=str(imported_summary.get("checked_at") or "").strip() or None,
                    include_festivals_index=include_festivals_index,
                )
    except Exception as exc:
        ops_status = "error"
        ops_err = str(exc)
        try:
            await _update_kaggle_status(
                "failed",
                kernel_ref or kaggle_kernel_ref,
                {"status": "FAILED", "failureMessage": ops_err[:500] if ops_err else ""},
            )
        except Exception:
            logger.debug("telegraph_cache: failed to send error status update", exc_info=True)
        raise
    finally:
        try:
            if kernel_ref:
                await remove_job("telegraph_cache_probe", kernel_ref)
        except Exception:
            pass
        try:
            await _cleanup_datasets(dataset_slugs)
        except Exception:
            logger.debug("telegraph_cache: dataset cleanup failed", exc_info=True)
        await finish_ops_run(
            db,
            run_id=ops_id,
            status=ops_status,
            metrics={
                "targets": int(imported_summary.get("total") or 0),
                "ok": int(imported_summary.get("ok") or 0),
                "fail": int(imported_summary.get("fail") or 0),
                "cached_ok": int(imported_summary.get("cached_ok") or 0),
                "photo_ok": int(imported_summary.get("photo_ok") or 0),
                "both_ok": int(imported_summary.get("both_ok") or 0),
                "warn_no_photo": int(imported_summary.get("warn_no_photo") or 0),
                **{f"regen_{k}": int(v) for k, v in (regen_summary or {}).items()},
            },
            details={
                "run_id": run_id,
                "kernel_ref": kernel_ref,
                "report_path": report_path,
                "error": ops_err,
            },
        )

    return {
        "run_id": run_id,
        "kernel_ref": kernel_ref,
        "report_path": report_path,
        "targets": int(imported_summary.get("total") or 0),
        "imported": imported_summary,
        "regen": regen_summary,
        "targets_meta": targets_meta,
    }


async def format_probe_stats_text(
    db: Database,
    *,
    kind: str | None = None,
    limit_examples: int = 12,
) -> str:
    """Render current probe stats from SQLite (no Kaggle calls)."""
    where = ""
    params: list[Any] = []
    if kind:
        where = "WHERE kind = ?"
        params.append(str(kind))

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            f"""
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN last_ok = 1 THEN 1 ELSE 0 END) AS cached_ok,
              SUM(CASE WHEN last_ok = 0 THEN 1 ELSE 0 END) AS no_cached,
              SUM(CASE WHEN last_ok = 1 AND last_has_photo = 0 THEN 1 ELSE 0 END) AS warn_no_photo,
              SUM(CASE WHEN last_ok = 1 AND last_has_photo = 1 THEN 1 ELSE 0 END) AS ok_with_photo
            FROM telegraph_preview_probe
            {where}
            """,
            tuple(params),
        )
        row = await cur.fetchone()
        total = int(row[0] or 0) if row else 0
        cached_ok = int(row[1] or 0) if row else 0
        no_cached = int(row[2] or 0) if row else 0
        warn_no_photo = int(row[3] or 0) if row else 0
        ok_with_photo = int(row[4] or 0) if row else 0

        cur2 = await conn.execute(
            f"""
            SELECT url, kind, consecutive_failures, last_error, last_title, last_checked_at
            FROM telegraph_preview_probe
            WHERE last_ok = 0
            {('AND kind = ?' if kind else '')}
            ORDER BY consecutive_failures DESC, last_checked_at DESC
            LIMIT ?
            """,
            tuple(([str(kind)] if kind else []) + [max(1, int(limit_examples))]),
        )
        examples = await cur2.fetchall()

        cur3 = await conn.execute(
            f"""
            SELECT url, kind, last_error, last_title, last_checked_at
            FROM telegraph_preview_probe
            WHERE last_ok = 1
              AND last_has_photo = 0
            {('AND kind = ?' if kind else '')}
            ORDER BY last_checked_at DESC
            LIMIT ?
            """,
            tuple(([str(kind)] if kind else []) + [max(1, int(limit_examples))]),
        )
        warn_examples = await cur3.fetchall()

    head = (
        f"Telegraph cache probe: total={total} "
        f"cached_page_ok={cached_ok} no_cached_page={no_cached} "
        f"warn_no_photo={warn_no_photo}"
    )
    if kind:
        head += f" kind={kind}"
    lines = [head]
    if not examples:
        if not warn_examples:
            return "\n".join(lines)
        lines.append("")
        lines.append("⚠️ Примеры без фото (Instant View есть):")
        for url, k, err, title, checked_at in warn_examples:
            label = (str(title or "").strip() or str(err or "").strip() or "—")[:80]
            lines.append(f"- [{k}] {label} — {url}")
        return "\n".join(lines)
    lines.append("")
    lines.append("❌ Примеры без Instant View (cached_page отсутствует):")
    for url, k, streak, err, title, checked_at in examples:
        label = (str(title or "").strip() or str(err or "").strip() or "—")[:80]
        lines.append(f"- [{k}] streak={int(streak or 0)} {label} — {url}")
    if warn_examples:
        lines.append("")
        lines.append("⚠️ Примеры без фото (Instant View есть):")
        for url, k, err, title, checked_at in warn_examples:
            label = (str(title or "").strip() or str(err or "").strip() or "—")[:80]
            lines.append(f"- [{k}] {label} — {url}")
    return "\n".join(lines)
