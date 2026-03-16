from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    EVENT_JOB_SUBMITTED,
)
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from admin_chat import resolve_superadmin_chat_id
from db import optimize, wal_checkpoint_truncate, vacuum
from heavy_ops import current_heavy_meta, describe_heavy_meta, heavy_operation
from ops_run import finish_ops_run, start_ops_run
from runtime import get_running_main


@dataclass
class Job:
    key: str
    func: Callable[[Any], Awaitable[None]]
    payload: List[Any] = field(default_factory=list)
    depends_on: Set[str] = field(default_factory=set)
    dirty: bool = False
    track: bool = True


MONTHS_NOM = [
    "",
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
]

MONTHS_GEN = [
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


async def _run_scheduled_guide_excursions(
    db,
    bot,
    *,
    mode: str,
) -> None:
    from guide_excursions.service import publish_guide_digest, run_guide_monitor

    target_chat_id = await resolve_superadmin_chat_id(db)
    result = await run_guide_monitor(
        db,
        bot,
        chat_id=target_chat_id,
        operator_id=None,
        trigger="scheduled",
        mode=mode,
        send_progress=bool(target_chat_id),
    )
    auto_publish = _env_enabled("ENABLE_GUIDE_DIGEST_SCHEDULED", default=False)
    if not auto_publish or mode != "full" or result.errors or bot is None:
        return
    publish_result = await publish_guide_digest(
        db,
        bot,
        family="new_occurrences",
        chat_id=target_chat_id,
    )
    if target_chat_id and publish_result.get("published"):
        try:
            await bot.send_message(
                int(target_chat_id),
                (
                    "📣 Scheduled guide digest published\n"
                    f"issue_id={publish_result.get('issue_id')}\n"
                    f"target={publish_result.get('target_chat') or '—'}"
                ),
                disable_web_page_preview=True,
            )
        except Exception:
            logging.exception("SCHED failed to notify admin about scheduled guide digest publish")


async def _run_scheduled_video_tomorrow_test(
    db,
    bot,
    *,
    profile_key: str,
) -> None:
    from video_announce.scenario import (
        TOMORROW_TEST_MIN_POSTERS,
        VideoAnnounceScenario,
    )

    target_chat_id = await resolve_superadmin_chat_id(db)
    if not target_chat_id or bot is None:
        logging.warning(
            "SCHED skipping video_tomorrow_test: missing target_chat_id=%s or bot=%s",
            target_chat_id,
            bot is not None,
        )
        return

    scenario = VideoAnnounceScenario(
        db,
        bot,
        chat_id=int(target_chat_id),
        user_id=int(target_chat_id),
    )
    await scenario.run_tomorrow_pipeline(
        profile_key=(profile_key or "default").strip() or "default",
        selected_max=TOMORROW_TEST_MIN_POSTERS,
        test_mode=True,
    )


def _cron_from_local(
    time_raw: str,
    tz_name: str,
    *,
    default_hour: str,
    default_minute: str,
    label: str,
) -> tuple[str, str]:
    hour = default_hour
    minute = default_minute
    try:
        if time_raw:
            hh, mm = map(int, time_raw.split(":"))
            tz = ZoneInfo(tz_name)
            local_dt = datetime.now(tz).replace(hour=hh, minute=mm, second=0, microsecond=0)
            utc_dt = local_dt.astimezone(timezone.utc)
            hour = str(utc_dt.hour)
            minute = str(utc_dt.minute)
    except Exception:
        logging.warning(
            "invalid %s time=%s tz=%s; using %s:%s UTC",
            label,
            time_raw,
            tz_name,
            default_hour,
            default_minute,
        )
    return hour, minute


def _safe_zoneinfo(tz_name: str, *, label: str) -> timezone | ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logging.warning("invalid %s timezone=%s; using UTC", label, tz_name)
        return timezone.utc


def _env_enabled(key: str, *, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class BatchProgress:
    """Track progress for a batch of event tasks."""

    def __init__(self, total_events: int) -> None:
        self.total_events = total_events
        self.events_done = 0
        self.status: Dict[str, str] = {}

    def register_job(self, key: str) -> None:
        self.status.setdefault(key, "pending")

    def finish_job(self, key: str, status: str = "done") -> None:
        if key in self.status:
            self.status[key] = status

    def event_completed(self) -> None:
        self.events_done += 1

    # Formatting -----------------------------------------------------------------

    def _format_range(self, start: datetime, end: datetime) -> str:
        if start.month == end.month and start.year == end.year:
            name = MONTHS_GEN[start.month]
            return f"{start.day}\u2013{end.day} {name} {start.year}"
        if start.year == end.year:
            s = f"{start.day} {MONTHS_GEN[start.month]}"
            e = f"{end.day} {MONTHS_GEN[end.month]} {start.year}"
        else:
            s = f"{start.day} {MONTHS_GEN[start.month]} {start.year}"
            e = f"{end.day} {MONTHS_GEN[end.month]} {end.year}"
        return f"{s}\u2013{e}"

    def _label(self, key: str) -> str:
        kind, _, ident = key.partition(":")
        if kind == "festival_pages":
            return "Festival"
        if kind == "month_pages":
            _, month = ident.split("-")
            name = MONTHS_NOM[int(month)].capitalize()
            return f"Month: {name}"
        if kind == "week_pages":
            year, week = ident.split("-")
            start = datetime.fromisocalendar(int(year), int(week), 1)
            end = start + timedelta(days=6)
            return f"Week: {self._format_range(start, end)}"
        if kind == "weekend_pages":
            start = datetime.strptime(ident, "%Y-%m-%d")
            end = start + timedelta(days=1)
            return f"Weekend: {self._format_range(start, end)}"
        if kind == "vk_week_post":
            year, week = ident.split("-")
            start = datetime.fromisocalendar(int(year), int(week), 1)
            end = start + timedelta(days=6)
            return f"VK week: {self._format_range(start, end)}"
        if kind == "vk_weekend_post":
            start = datetime.strptime(ident, "%Y-%m-%d")
            end = start + timedelta(days=1)
            return f"VK weekend: {self._format_range(start, end)}"
        return key

    def snapshot_text(self) -> str:
        icon = {
            "pending": "⏳",
            "running": "🔄",
            "deferred": "⏸",
            "captcha": "🧩⏸",
            "captcha_expired": "⚠️",
            "done": "✅",
            "error": "❌",
            "skipped_nochange": "⏭",
        }
        lines = [
            f"Events (Telegraph): {self.events_done}/{self.total_events}"
        ]
        order = {
            "festival_pages": 0,
            "month_pages": 1,
            "week_pages": 2,
            "weekend_pages": 3,
            "vk_week_post": 4,
            "vk_weekend_post": 5,
        }
        for key in sorted(
            self.status.keys(), key=lambda k: (order.get(k.split(":")[0], 99), k)
        ):
            lines.append(f"{icon[self.status[key]]} {self._label(key)}")
        return "\n".join(lines)

    def report(self) -> Dict[str, Any]:
        return {"events": (self.events_done, self.total_events), **self.status}


class CoalescingScheduler:
    def __init__(
        self,
        progress: Optional[BatchProgress] = None,
        debounce_seconds: float = 0.0,
        on_captcha: Optional[Callable[["CoalescingScheduler", str], None]] = None,
    ) -> None:
        self.jobs: Dict[str, Job] = {}
        self.progress = progress
        self.order: List[str] = []
        self.debounce_seconds = debounce_seconds
        self._remaining: Set[str] | None = None
        self.on_captcha = on_captcha

    def add_job(
        self,
        key: str,
        func: Callable[[Any], Awaitable[None]],
        payload: Optional[Any] = None,
        depends_on: Optional[List[str]] = None,
        track: bool = True,
        coalesce: bool = True,
    ) -> None:
        if key in self.jobs:
            job = self.jobs[key]
            if payload is not None and coalesce:
                if isinstance(job.payload, list):
                    if isinstance(payload, list):
                        job.payload.extend(payload)
                    else:
                        job.payload.append(payload)
                else:
                    job.payload = [job.payload, payload]
            job.dirty = True
            if depends_on:
                job.depends_on.update(depends_on)
            return
        job = Job(
            key=key,
            func=func,
            payload=
            []
            if payload is None
            else (
                [payload]
                if coalesce and not isinstance(payload, list)
                else payload
            ),
            depends_on=set(depends_on or []),
            track=track,
        )
        self.jobs[key] = job
        if track and self.progress:
            self.progress.register_job(key)

    async def run(self) -> None:
        if self.debounce_seconds > 0 and self._remaining is None:
            await asyncio.sleep(self.debounce_seconds)
        remaining = self._remaining if self._remaining is not None else set(self.jobs.keys())
        self._remaining = remaining
        completed: Set[str] = set(self.jobs.keys()) - remaining
        while remaining:
            progress_made = False
            for key in list(remaining):
                job = self.jobs[key]
                if job.depends_on - completed:
                    continue
                progress_made = True
                try:
                    if self.progress:
                        self.progress.finish_job(key, "running")
                    await job.func(job.payload)
                except Exception as e:
                    if getattr(e, "code", None) == 14:
                        if self.progress:
                            self.progress.finish_job(key, "captcha")
                        self._remaining = remaining
                        if self.on_captcha:
                            self.on_captcha(self, key)
                        return
                    if self.progress:
                        self.progress.finish_job(key, "error")
                    self._remaining = remaining
                    raise
                if self.progress:
                    self.progress.finish_job(key, "done")
                if job.track:
                    self.order.append(key)
                completed.add(key)
                remaining.remove(key)
            if not progress_made:
                raise RuntimeError("Circular dependency detected")
        self._remaining = None

    @property
    def remaining_jobs(self) -> Set[str]:
        return self._remaining or set()

# Utilities for tests -------------------------------------------------------------------

async def _dummy_job(payload: Any, progress: Optional[BatchProgress] = None) -> None:
    await asyncio.sleep(0)
    if progress and isinstance(payload, dict) and payload.get("event"):
        progress.event_completed()


def schedule_event_batch(
    scheduler: CoalescingScheduler,
    festival_id: int,
    dates: List[str],
) -> None:
    """Schedule tasks for a batch of events belonging to one festival."""

    festival_key = f"festival_pages:{festival_id}"
    scheduler.add_job(festival_key, _dummy_job)

    for idx, date_str in enumerate(dates, 1):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        week = dt.isocalendar().week
        month_key = f"month_pages:{dt:%Y-%m}"
        week_key = f"week_pages:{dt.year}-{week:02d}"
        vk_week_key = f"vk_week_post:{dt.year}-{week:02d}"
        weekend_key = None
        vk_weekend_key = None
        if dt.weekday() >= 5:
            wstart = dt - timedelta(days=dt.weekday() - 5)
            weekend_key = f"weekend_pages:{wstart:%Y-%m-%d}"
            vk_weekend_key = f"vk_weekend_post:{wstart:%Y-%m-%d}"

        scheduler.add_job(
            f"telegraph:{idx}",
            lambda payload, p=scheduler.progress: _dummy_job(payload, p),
            payload={"event": idx},
            track=False,
            coalesce=False,
        )
        scheduler.add_job(
            month_key,
            _dummy_job,
            payload=idx,
            depends_on=[festival_key],
        )
        scheduler.add_job(
            week_key,
            _dummy_job,
            payload=idx,
            depends_on=[festival_key],
        )
        if weekend_key:
            scheduler.add_job(
                weekend_key,
                _dummy_job,
                payload=idx,
                depends_on=[festival_key],
            )
            wk_dep = [month_key, week_key, weekend_key]
        else:
            wk_dep = [month_key, week_key]
        scheduler.add_job(
            vk_week_key,
            _dummy_job,
            payload=idx,
            depends_on=wk_dep,
        )
        if vk_weekend_key and weekend_key:
            scheduler.add_job(
                vk_weekend_key,
                _dummy_job,
                payload=idx,
                depends_on=wk_dep,
            )


# ---------------------------------------------------------------------------
# APScheduler wrapper used by the main application

_scheduler: AsyncIOScheduler | None = None
_run_meta: dict[str, tuple[str, float]] = {}
_heavy_job_lock = asyncio.Lock()

# Jobs that can take minutes/hours (Kaggle/LLM/rendering) and should not overlap in prod.
_HEAVY_JOB_IDS: set[str] = {
    "tg_monitoring",
    "vk_auto_import",
    "source_parsing",
    "source_parsing_day",
    "festival_queue",
    "3di_scheduler",
    "nightly_page_sync",
    "telegraph_cache_sanitize",
}

_OPS_RUN_KIND_BY_JOB_ID: dict[str, str] = {
    "3di_scheduler": "3di",
    "source_parsing": "parse",
    "source_parsing_day": "parse",
}


def _ops_run_kind_for_job(job_id: str) -> str:
    return _OPS_RUN_KIND_BY_JOB_ID.get(job_id, job_id)


async def _record_scheduler_skip(
    db_obj: Any,
    *,
    job_id: str,
    run_id: str | None,
    reason: str,
    blocked_by: Any | None = None,
) -> None:
    if db_obj is None or not hasattr(db_obj, "raw_conn"):
        return
    details: dict[str, Any] = {
        "run_id": run_id,
        "skip_reason": str(reason or "").strip() or "unknown",
        "scheduler_job_id": str(job_id or "").strip() or "scheduler_job",
    }
    blocked_kind = str(getattr(blocked_by, "kind", "") or "").strip()
    blocked_trigger = str(getattr(blocked_by, "trigger", "") or "").strip()
    if blocked_kind:
        details["blocked_by_kind"] = blocked_kind
    if blocked_trigger:
        details["blocked_by_trigger"] = blocked_trigger
    try:
        ops_run_id = await start_ops_run(
            db_obj,
            kind=_ops_run_kind_for_job(job_id),
            trigger="scheduled",
            operator_id=0,
            details=details,
        )
        await finish_ops_run(
            db_obj,
            run_id=ops_run_id,
            status="skipped",
            details=details,
        )
    except Exception:
        logging.warning("SCHED failed to record skipped ops_run job_id=%s", job_id, exc_info=True)


def _job_next_run(job):
    return getattr(job, "next_run_time", None) or getattr(job, "next_run_at", None)


def _job_wrapper(job_id: str, func, *, notify_skip: Callable[[str, str], None] | None = None):
    async def _run(*args, **kwargs):
        serialize_heavy = (os.getenv("SCHED_SERIALIZE_HEAVY_JOBS") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        is_heavy = job_id in _HEAVY_JOB_IDS
        guard_mode_raw = (os.getenv("SCHED_HEAVY_GUARD_MODE") or "").strip().lower()
        if guard_mode_raw in {"0", "off", "false", "no", "disable", "disabled"}:
            guard_mode = "off"
        elif guard_mode_raw in {"wait", "block", "serialize"}:
            guard_mode = "wait"
        elif guard_mode_raw in {"skip", "try", "nonblocking", "non-blocking"}:
            guard_mode = "skip"
        else:
            # Backwards-compatible default: old "serialize" mode implies waiting.
            guard_mode = "wait" if serialize_heavy else "skip"
        timeout_raw = (os.getenv("SCHED_HEAVY_TRY_TIMEOUT_SEC") or "0.2").strip()
        try:
            guard_timeout = max(0.0, float(timeout_raw))
        except ValueError:
            guard_timeout = 0.2
        run_id, start = _run_meta.get(job_id, (uuid4().hex, _time.perf_counter()))
        done = asyncio.Event()

        async def heartbeat():
            while not done.is_set():
                await asyncio.sleep(10)
                took_ms = (_time.perf_counter() - start) * 1000
                logging.info(
                    "job_heartbeat job_id=%s run_id=%s took_ms=%.0f",
                    job_id,
                    run_id,
                    took_ms,
                )

        async def _execute():
            hb_task = asyncio.create_task(heartbeat())
            try:
                return await func(*args, run_id=run_id, **kwargs)
            finally:
                done.set()
                hb_task.cancel()

        async def _run_guarded():
            if not is_heavy or guard_mode == "off":
                return await _execute()

            if guard_mode == "skip":
                async with heavy_operation(
                    kind=job_id,
                    trigger="scheduled",
                    mode="try",
                    timeout_sec=guard_timeout,
                    run_id=run_id,
                    operator_id=0,
                ) as acquired:
                    if not acquired:
                        meta = current_heavy_meta()
                        meta_txt = describe_heavy_meta(meta)
                        await _record_scheduler_skip(
                            args[0] if args else None,
                            job_id=job_id,
                            run_id=run_id,
                            reason="heavy_busy",
                            blocked_by=meta,
                        )
                        logging.info(
                            "job_skip_heavy_busy job_id=%s run_id=%s current=%s",
                            job_id,
                            run_id,
                            meta_txt,
                        )
                        if notify_skip:
                            notify_skip(job_id, f"идёт другая тяжёлая операция: {meta_txt}")
                        return None
                    return await _execute()

            async with heavy_operation(
                kind=job_id,
                trigger="scheduled",
                mode="wait",
                run_id=run_id,
                operator_id=0,
            ):
                return await _execute()

        if serialize_heavy and is_heavy:
            if _heavy_job_lock.locked():
                logging.info("job_wait_heavy_lock job_id=%s run_id=%s", job_id, run_id)
            async with _heavy_job_lock:
                return await _run_guarded()
        return await _run_guarded()

    return _run


def _on_event(event):
    if not hasattr(event, "job_id"):
        logging.debug(
            "scheduler event %s (no job_id), ignored", getattr(event, "code", None)
        )
        return
    job_id = event.job_id
    name_map = {
        EVENT_JOB_SUBMITTED: "JOB_SUBMITTED",
        EVENT_JOB_EXECUTED: "JOB_EXECUTED",
        EVENT_JOB_ERROR: "JOB_ERROR",
        EVENT_JOB_MISSED: "JOB_MISSED",
    }
    event_name = name_map.get(event.code, str(event.code))
    run_id = None
    start = None
    if event.code == EVENT_JOB_SUBMITTED:
        run_id = uuid4().hex
        start = _time.perf_counter()
        _run_meta[job_id] = (run_id, start)
    else:
        run_id, start = _run_meta.get(job_id, (uuid4().hex, None))
    took_ms = None
    if event.code in (EVENT_JOB_EXECUTED, EVENT_JOB_ERROR) and start is not None:
        took_ms = (_time.perf_counter() - start) * 1000
        _run_meta.pop(job_id, None)
    if event.code == EVENT_JOB_MISSED:
        _run_meta.pop(job_id, None)
        run_id = uuid4().hex
    next_run = None
    if _scheduler:
        job = _scheduler.get_job(job_id)
        next_run = _job_next_run(job) if job else None
    tb_excerpt = None
    tb = getattr(event, "traceback", None)
    if tb:
        tb_excerpt = " | ".join(tb.strip().splitlines()[-3:])
    logging.info(
        "%s job_id=%s run_id=%s next_run=%s took_ms=%s traceback_excerpt=%s",
        event_name,
        job_id,
        run_id,
        next_run,
        f"{took_ms:.0f}" if took_ms is not None else "0",
        tb_excerpt,
    )


def startup(
    db,
    bot,
    *,
    vk_scheduler=None,
    vk_poll_scheduler=None,
    vk_crawl_cron=None,
    cleanup_scheduler=None,
    partner_notification_scheduler=None,
    nightly_page_sync=None,
    rebuild_fest_nav_if_changed=None,
) -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        executor = AsyncIOExecutor()
        _scheduler = AsyncIOScheduler(executors={"default": executor}, timezone="UTC")
        _scheduler.configure(
            job_defaults={
                "max_instances": 1,
                "coalesce": True,
                "misfire_grace_time": 30,
            }
        )

    is_prod = os.getenv("DEV_MODE") != "1" and os.getenv("PYTEST_CURRENT_TEST") is None

    main_module = None

    async def _notify_admin_skip_async(job_name: str, reason: str) -> None:
        chat_id = await resolve_superadmin_chat_id(db)
        if not chat_id:
            return
        if bot is None or not hasattr(bot, "send_message"):
            return
        text = f"⚠️ SCHED: пропуск {job_name}. Причина: {reason}"
        try:
            await bot.send_message(chat_id, text)
        except Exception:
            logging.exception("SCHED failed to notify admin chat")

    def _notify_admin_skip(job_name: str, reason: str) -> None:
        try:
            asyncio.create_task(_notify_admin_skip_async(job_name, reason))
        except RuntimeError:
            logging.warning("SCHED failed to notify admin: no running event loop")
        except Exception:
            logging.exception("SCHED failed to notify admin chat")

    def resolve(name: str, value):
        nonlocal main_module
        if value is not None:
            return value
        if main_module is None:
            main_module = get_running_main()
        if main_module is None:
            raise RuntimeError(
                f"{name} not provided and main module is not loaded"
            )
        try:
            return getattr(main_module, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"running main module does not define {name!r}"
            ) from exc

    vk_scheduler = resolve("vk_scheduler", vk_scheduler)
    vk_poll_scheduler = resolve("vk_poll_scheduler", vk_poll_scheduler)
    vk_crawl_cron = resolve("vk_crawl_cron", vk_crawl_cron)
    cleanup_scheduler = resolve("cleanup_scheduler", cleanup_scheduler)
    partner_notification_scheduler = resolve(
        "partner_notification_scheduler", partner_notification_scheduler
    )
    rebuild_fest_nav_if_changed = resolve(
        "rebuild_fest_nav_if_changed", rebuild_fest_nav_if_changed
    )
    nightly_page_sync = (
        resolve("nightly_page_sync", nightly_page_sync)
        if os.getenv("ENABLE_NIGHTLY_PAGE_SYNC") == "1"
        else nightly_page_sync
    )

    def _register_job(job_id: str, *args, **kwargs):
        try:
            job = _scheduler.add_job(*args, **kwargs)
        except Exception:
            logging.exception("SCHED failed to register job id=%s", job_id)
            return None
        logging.info(
            "SCHED registered job id=%s next_run=%s", job.id, _job_next_run(job)
        )
        return job

    enable_core_schedulers = _env_enabled("ENABLE_CORE_SCHEDULERS", default=True)
    if enable_core_schedulers:
        _register_job(
            "vk_scheduler",
            _job_wrapper("vk_scheduler", vk_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="vk_scheduler",
            minute="1,16,31,46",
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _register_job(
            "vk_poll_scheduler",
            _job_wrapper(
                "vk_poll_scheduler", vk_poll_scheduler, notify_skip=_notify_admin_skip
            ),
            "cron",
            id="vk_poll_scheduler",
            minute="2,17,32,47",
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _register_job(
            "cleanup_scheduler",
            _job_wrapper("cleanup_scheduler", cleanup_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="cleanup_scheduler",
            hour="2",
            minute="7",
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _register_job(
            "partner_notification_scheduler",
            _job_wrapper(
                "partner_notification_scheduler",
                partner_notification_scheduler,
                notify_skip=_notify_admin_skip,
            ),
            "cron",
            id="partner_notification_scheduler",
            minute="5",
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _register_job(
            "fest_nav_rebuild",
            _job_wrapper(
                "fest_nav_rebuild",
                rebuild_fest_nav_if_changed,
                notify_skip=_notify_admin_skip,
            ),
            "cron",
            id="fest_nav_rebuild",
            hour="3",
            minute="0",
            args=[db],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )

        times_raw = os.getenv(
            "VK_CRAWL_TIMES_LOCAL", "05:15,09:15,13:15,17:15,21:15,22:45"
        )
        tz_name = os.getenv("VK_CRAWL_TZ", "Europe/Kaliningrad")
        tz = _safe_zoneinfo(tz_name, label="VK_CRAWL_TZ")
        for idx, t in enumerate(times_raw.split(",")):
            t = t.strip()
            if not t:
                continue
            try:
                hh, mm = map(int, t.split(":"))
            except ValueError:
                logging.warning("invalid VK_CRAWL_TIMES_LOCAL entry: %s", t)
                continue
            now_local = datetime.now(tz).replace(hour=hh, minute=mm, second=0, microsecond=0)
            now_utc = now_local.astimezone(timezone.utc)
            _register_job(
                f"vk_crawl_cron_{idx}",
                _job_wrapper(
                    "vk_crawl_cron", vk_crawl_cron, notify_skip=_notify_admin_skip
                ),
                "cron",
                id=f"vk_crawl_cron_{idx}",
                hour=str(now_utc.hour),
                minute=str(now_utc.minute),
                args=[db, bot],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )
    else:
        logging.info("SCHED skipping core schedulers (ENABLE_CORE_SCHEDULERS!=1)")

    # Source parsing from theatres (before daily announcement at 08:00)
    enable_source_parsing = _env_enabled("ENABLE_SOURCE_PARSING", default=is_prod)
    if enable_source_parsing:
        from source_parsing.commands import source_parsing_scheduler
        parsing_time_raw = os.getenv("SOURCE_PARSING_TIME_LOCAL", "04:30").strip()
        parsing_tz_name = os.getenv("SOURCE_PARSING_TZ", "Europe/Kaliningrad")
        parsing_hour, parsing_minute = _cron_from_local(
            parsing_time_raw,
            parsing_tz_name,
            default_hour="4",
            default_minute="30",
            label="SOURCE_PARSING_TIME_LOCAL",
        )
        _register_job(
            "source_parsing",
            _job_wrapper("source_parsing", source_parsing_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="source_parsing",
            hour=parsing_hour,
            minute=parsing_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping source_parsing (ENABLE_SOURCE_PARSING!=1)")
        _notify_admin_skip("source_parsing", "ENABLE_SOURCE_PARSING!=1")

    enable_source_parsing_day = _env_enabled("ENABLE_SOURCE_PARSING_DAY", default=is_prod)
    if enable_source_parsing_day:
        from source_parsing.commands import source_parsing_scheduler_if_changed
        day_time_raw = os.getenv("SOURCE_PARSING_DAY_TIME_LOCAL", "14:15").strip()
        day_tz_name = os.getenv("SOURCE_PARSING_DAY_TZ", "Europe/Kaliningrad")
        day_hour, day_minute = _cron_from_local(
            day_time_raw,
            day_tz_name,
            default_hour="12",
            default_minute="15",
            label="SOURCE_PARSING_DAY_TIME_LOCAL",
        )
        _register_job(
            "source_parsing_day",
            _job_wrapper("source_parsing_day", source_parsing_scheduler_if_changed, notify_skip=_notify_admin_skip),
            "cron",
            id="source_parsing_day",
            hour=day_hour,
            minute=day_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping source_parsing_day (ENABLE_SOURCE_PARSING_DAY!=1)")
        _notify_admin_skip("source_parsing_day", "ENABLE_SOURCE_PARSING_DAY!=1")

    enable_tg_monitoring = _env_enabled("ENABLE_TG_MONITORING", default=is_prod)
    if enable_tg_monitoring:
        from source_parsing.telegram.service import telegram_monitor_scheduler
        tg_time_raw = os.getenv("TG_MONITORING_TIME_LOCAL", "23:40").strip()
        tg_tz_name = os.getenv("TG_MONITORING_TZ", "Europe/Kaliningrad")
        tg_hour, tg_minute = _cron_from_local(
            tg_time_raw,
            tg_tz_name,
            default_hour="23",
            default_minute="40",
            label="TG_MONITORING_TIME_LOCAL",
        )
        _register_job(
            "tg_monitoring",
            _job_wrapper("tg_monitoring", telegram_monitor_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="tg_monitoring",
            hour=tg_hour,
            minute=tg_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping tg_monitoring (ENABLE_TG_MONITORING!=1)")
        _notify_admin_skip("tg_monitoring", "ENABLE_TG_MONITORING!=1")

    enable_vk_auto_import = _env_enabled("ENABLE_VK_AUTO_IMPORT", default=False)
    if enable_vk_auto_import:
        from vk_auto_queue import vk_auto_import_scheduler

        vk_auto_times = os.getenv(
            "VK_AUTO_IMPORT_TIMES_LOCAL", "06:15,10:15,12:00,18:30"
        ).strip()
        vk_auto_tz = os.getenv("VK_AUTO_IMPORT_TZ", "Europe/Kaliningrad").strip()
        for idx, t in enumerate(vk_auto_times.split(",")):
            t = t.strip()
            if not t:
                continue
            hour, minute = _cron_from_local(
                t,
                vk_auto_tz,
                default_hour="6",
                default_minute="30",
                label="VK_AUTO_IMPORT_TIMES_LOCAL",
            )
            _register_job(
                f"vk_auto_import_{idx}",
                _job_wrapper("vk_auto_import", vk_auto_import_scheduler, notify_skip=_notify_admin_skip),
                "cron",
                id=f"vk_auto_import_{idx}",
                hour=hour,
                minute=minute,
                args=[db, bot],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )
    else:
        logging.info("SCHED skipping vk_auto_import (ENABLE_VK_AUTO_IMPORT!=1)")
        _notify_admin_skip("vk_auto_import", "ENABLE_VK_AUTO_IMPORT!=1")

    enable_festival_queue = _env_enabled("ENABLE_FESTIVAL_QUEUE", default=False)
    if enable_festival_queue:
        from festival_queue import process_festival_queue

        async def festival_queue_scheduler(
            db_obj,
            bot_obj,
            *,
            run_id: str | None = None,
        ) -> None:
            limit_raw = (os.getenv("FESTIVAL_QUEUE_LIMIT") or "").strip()
            limit: int | None = None
            if limit_raw:
                try:
                    parsed_limit = int(limit_raw)
                    if parsed_limit > 0:
                        limit = parsed_limit
                except ValueError:
                    logging.warning("invalid FESTIVAL_QUEUE_LIMIT=%r; using no limit", limit_raw)
            admin_chat_id = await resolve_superadmin_chat_id(db_obj)
            report = await process_festival_queue(
                db_obj,
                bot=bot_obj,
                chat_id=admin_chat_id,
                limit=limit,
                trigger="scheduled",
                operator_id=0,
                run_id=run_id,
            )
            logging.info(
                "festival_queue_scheduler processed=%s success=%s failed=%s skipped=%s",
                report.processed,
                report.success,
                report.failed,
                report.skipped,
            )

        fest_queue_times = os.getenv("FESTIVAL_QUEUE_TIMES_LOCAL", "03:30,16:30").strip()
        fest_queue_tz = os.getenv("FESTIVAL_QUEUE_TZ", "Europe/Kaliningrad").strip()
        for idx, t in enumerate(fest_queue_times.split(",")):
            t = t.strip()
            if not t:
                continue
            hour, minute = _cron_from_local(
                t,
                fest_queue_tz,
                default_hour="3",
                default_minute="30",
                label="FESTIVAL_QUEUE_TIMES_LOCAL",
            )
            _register_job(
                f"festival_queue_{idx}",
                _job_wrapper("festival_queue", festival_queue_scheduler, notify_skip=_notify_admin_skip),
                "cron",
                id=f"festival_queue_{idx}",
                hour=hour,
                minute=minute,
                args=[db, bot],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )
    else:
        logging.info("SCHED skipping festival_queue (ENABLE_FESTIVAL_QUEUE!=1)")
        _notify_admin_skip("festival_queue", "ENABLE_FESTIVAL_QUEUE!=1")

    enable_ticket_sites_queue = _env_enabled("ENABLE_TICKET_SITES_QUEUE", default=False)
    if enable_ticket_sites_queue:
        from ticket_sites_queue import process_ticket_sites_queue

        async def ticket_sites_queue_scheduler(
            db_obj,
            bot_obj,
            *,
            run_id: str | None = None,
        ) -> None:
            limit_raw = (os.getenv("TICKET_SITES_QUEUE_LIMIT") or "").strip()
            limit: int | None = None
            if limit_raw:
                try:
                    parsed_limit = int(limit_raw)
                    if parsed_limit > 0:
                        limit = parsed_limit
                except ValueError:
                    logging.warning("invalid TICKET_SITES_QUEUE_LIMIT=%r; using default", limit_raw)
            admin_chat_id = await resolve_superadmin_chat_id(db_obj)
            report = await process_ticket_sites_queue(
                db_obj,
                bot=bot_obj,
                chat_id=admin_chat_id,
                limit=limit,
                trigger="scheduled",
                operator_id=0,
                run_id=run_id,
            )
            logging.info(
                "ticket_sites_queue_scheduler processed=%s success=%s failed=%s skipped=%s",
                report.processed,
                report.success,
                report.failed,
                report.skipped,
            )

        t_time_raw = os.getenv("TICKET_SITES_QUEUE_TIME_LOCAL", "11:20").strip()
        t_tz_name = os.getenv("TICKET_SITES_QUEUE_TZ", "Europe/Kaliningrad").strip()
        t_hour, t_minute = _cron_from_local(
            t_time_raw,
            t_tz_name,
            default_hour="9",
            default_minute="20",
            label="TICKET_SITES_QUEUE_TIME_LOCAL",
        )
        _register_job(
            "ticket_sites_queue",
            _job_wrapper("ticket_sites_queue", ticket_sites_queue_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="ticket_sites_queue",
            hour=t_hour,
            minute=t_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping ticket_sites_queue (ENABLE_TICKET_SITES_QUEUE!=1)")
        _notify_admin_skip("ticket_sites_queue", "ENABLE_TICKET_SITES_QUEUE!=1")

    enable_3di = _env_enabled("ENABLE_3DI_SCHEDULED", default=is_prod)
    if enable_3di:
        from preview_3d.handlers import run_3di_new_only_scheduler

        async def preview_3di_scheduler(
            db_obj,
            bot_obj,
            *,
            run_id: str | None = None,
        ) -> None:
            run_chat_id = await resolve_superadmin_chat_id(db_obj)
            await run_3di_new_only_scheduler(
                db_obj,
                bot_obj,
                chat_id=run_chat_id,
                run_id=run_id,
            )

        three_di_times = os.getenv("THREEDI_TIMES_LOCAL", "05:30,15:15,17:15")
        three_di_tz = os.getenv("THREEDI_TZ", "Europe/Kaliningrad")
        for idx, t in enumerate(three_di_times.split(",")):
            t = t.strip()
            if not t:
                continue
            hour, minute = _cron_from_local(
                t,
                three_di_tz,
                default_hour="5",
                default_minute="30",
                label="THREEDI_TIMES_LOCAL",
            )
            _register_job(
                f"3di_scheduler_{idx}",
                _job_wrapper("3di_scheduler", preview_3di_scheduler, notify_skip=_notify_admin_skip),
                "cron",
                id=f"3di_scheduler_{idx}",
                hour=hour,
                minute=minute,
                args=[db, bot],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )
    else:
        logging.info("SCHED skipping 3di_scheduler (ENABLE_3DI_SCHEDULED!=1)")
        _notify_admin_skip("3di_scheduler", "ENABLE_3DI_SCHEDULED!=1")

    enable_guide_excursions = _env_enabled("ENABLE_GUIDE_EXCURSIONS_SCHEDULED", default=False)
    if enable_guide_excursions:
        async def guide_excursions_scheduler(
            db_obj,
            bot_obj,
            *,
            mode: str,
            run_id: str | None = None,
        ) -> None:
            await _run_scheduled_guide_excursions(
                db_obj,
                bot_obj,
                mode=mode,
            )

        guide_tz_name = os.getenv("GUIDE_EXCURSIONS_TZ", "Europe/Kaliningrad").strip()
        light_times = (os.getenv("GUIDE_EXCURSIONS_LIGHT_TIMES_LOCAL", "09:05,13:20") or "").split(",")
        for idx, value in enumerate(light_times):
            raw_time = value.strip()
            if not raw_time:
                continue
            hour, minute = _cron_from_local(
                raw_time,
                guide_tz_name,
                default_hour="9",
                default_minute="5",
                label="GUIDE_EXCURSIONS_LIGHT_TIMES_LOCAL",
            )
            _register_job(
                f"guide_excursions_light_{idx}",
                _job_wrapper("guide_excursions_light", guide_excursions_scheduler, notify_skip=_notify_admin_skip),
                "cron",
                id=f"guide_excursions_light_{idx}",
                hour=hour,
                minute=minute,
                args=[db, bot],
                kwargs={"mode": "light"},
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )

        full_time_raw = os.getenv("GUIDE_EXCURSIONS_FULL_TIME_LOCAL", "20:10").strip()
        full_hour, full_minute = _cron_from_local(
            full_time_raw,
            guide_tz_name,
            default_hour="20",
            default_minute="10",
            label="GUIDE_EXCURSIONS_FULL_TIME_LOCAL",
        )
        _register_job(
            "guide_excursions_full",
            _job_wrapper("guide_excursions_full", guide_excursions_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="guide_excursions_full",
            hour=full_hour,
            minute=full_minute,
            args=[db, bot],
            kwargs={"mode": "full"},
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping guide_excursions (ENABLE_GUIDE_EXCURSIONS_SCHEDULED!=1)")
        _notify_admin_skip("guide_excursions", "ENABLE_GUIDE_EXCURSIONS_SCHEDULED!=1")

    enable_video_tomorrow_test = _env_enabled(
        "ENABLE_V_TEST_TOMORROW_SCHEDULED", default=False
    )
    if enable_video_tomorrow_test:
        async def video_tomorrow_test_scheduler(
            db_obj,
            bot_obj,
            *,
            profile_key: str,
            run_id: str | None = None,
        ) -> None:
            await _run_scheduled_video_tomorrow_test(
                db_obj,
                bot_obj,
                profile_key=profile_key,
            )

        video_tz_name = os.getenv("V_TEST_TOMORROW_TZ", "Europe/Kaliningrad").strip()
        video_time_raw = os.getenv("V_TEST_TOMORROW_TIME_LOCAL", "21:20").strip()
        video_profile_key = (os.getenv("V_TEST_TOMORROW_PROFILE", "default") or "").strip() or "default"
        video_hour, video_minute = _cron_from_local(
            video_time_raw,
            video_tz_name,
            default_hour="21",
            default_minute="20",
            label="V_TEST_TOMORROW_TIME_LOCAL",
        )
        _register_job(
            "video_tomorrow_test",
            _job_wrapper("video_tomorrow_test", video_tomorrow_test_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="video_tomorrow_test",
            hour=video_hour,
            minute=video_minute,
            args=[db, bot],
            kwargs={"profile_key": video_profile_key},
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping video_tomorrow_test (ENABLE_V_TEST_TOMORROW_SCHEDULED!=1)")
        _notify_admin_skip("video_tomorrow_test", "ENABLE_V_TEST_TOMORROW_SCHEDULED!=1")

    enable_general_stats = _env_enabled("ENABLE_GENERAL_STATS", default=False)
    if enable_general_stats:
        from general_stats import general_stats_scheduler

        general_stats_time_raw = os.getenv("GENERAL_STATS_TIME_LOCAL", "07:30").strip()
        general_stats_tz_name = os.getenv("GENERAL_STATS_TZ", "Europe/Kaliningrad").strip()
        general_stats_hour, general_stats_minute = _cron_from_local(
            general_stats_time_raw,
            general_stats_tz_name,
            default_hour="5",
            default_minute="30",
            label="GENERAL_STATS_TIME_LOCAL",
        )
        _register_job(
            "general_stats",
            _job_wrapper("general_stats", general_stats_scheduler, notify_skip=_notify_admin_skip),
            "cron",
            id="general_stats",
            hour=general_stats_hour,
            minute=general_stats_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping general_stats (ENABLE_GENERAL_STATS!=1)")
        _notify_admin_skip("general_stats", "ENABLE_GENERAL_STATS!=1")

    enable_telegraph_cache = _env_enabled("ENABLE_TELEGRAPH_CACHE_SANITIZER", default=False)
    if enable_telegraph_cache:
        from telegraph_cache_sanitizer import run_telegraph_cache_sanitizer

        async def telegraph_cache_sanitize_scheduler(
            db_obj,
            bot_obj,
            *,
            run_id: str | None = None,
        ) -> None:
            admin_chat_id = await resolve_superadmin_chat_id(db_obj)
            res = await run_telegraph_cache_sanitizer(
                db_obj,
                bot=bot_obj,
                chat_id=admin_chat_id,
                operator_id=0,
                trigger="scheduled",
                run_id=run_id,
            )
            imported = res.get("imported") or {}
            regen = res.get("regen") or {}
            logging.info(
                "telegraph_cache_sanitize_scheduler total=%s ok=%s fail=%s regen=%s",
                imported.get("total"),
                imported.get("ok"),
                imported.get("fail"),
                regen,
            )
            if admin_chat_id and bot_obj is not None:
                try:
                    text = (
                        "🧼 Telegraph cache sanitizer (scheduled): готово\n"
                        f"ok={imported.get('ok', 0)} fail={imported.get('fail', 0)} total={imported.get('total', 0)}\n"
                        + (
                            (
                                "regen: "
                                + ", ".join(
                                    f"{k}={int(v)}"
                                    for k, v in regen.items()
                                    if int(v or 0) > 0
                                )
                            )
                            if regen
                            else ""
                        )
                    ).strip()
                    await bot_obj.send_message(admin_chat_id, text, disable_web_page_preview=True)
                except Exception:
                    logging.warning("telegraph_cache_sanitize_scheduler notify failed", exc_info=True)

        cache_time_raw = os.getenv("TELEGRAPH_CACHE_TIME_LOCAL", "01:10").strip()
        cache_tz_name = os.getenv("TELEGRAPH_CACHE_TZ", "Europe/Kaliningrad").strip() or "Europe/Kaliningrad"
        cache_hour, cache_minute = _cron_from_local(
            cache_time_raw,
            cache_tz_name,
            default_hour="23",
            default_minute="10",
            label="TELEGRAPH_CACHE_TIME_LOCAL",
        )
        _register_job(
            "telegraph_cache_sanitize",
            _job_wrapper(
                "telegraph_cache_sanitize",
                telegraph_cache_sanitize_scheduler,
                notify_skip=_notify_admin_skip,
            ),
            "cron",
            id="telegraph_cache_sanitize",
            hour=cache_hour,
            minute=cache_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping telegraph_cache_sanitize (ENABLE_TELEGRAPH_CACHE_SANITIZER!=1)")
        _notify_admin_skip("telegraph_cache_sanitize", "ENABLE_TELEGRAPH_CACHE_SANITIZER!=1")

    enable_kaggle_recovery = _env_enabled("ENABLE_KAGGLE_RECOVERY", default=is_prod)
    if enable_kaggle_recovery:
        from kaggle_recovery import kaggle_recovery_scheduler
        interval_raw = os.getenv("KAGGLE_RECOVERY_INTERVAL_MINUTES", "5").strip()
        try:
            interval_min = max(1, int(interval_raw))
        except ValueError:
            interval_min = 5
        _register_job(
            "kaggle_recovery",
            _job_wrapper("kaggle_recovery", kaggle_recovery_scheduler, notify_skip=_notify_admin_skip),
            "interval",
            id="kaggle_recovery",
            minutes=interval_min,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping kaggle_recovery (ENABLE_KAGGLE_RECOVERY!=1)")
        _notify_admin_skip("kaggle_recovery", "ENABLE_KAGGLE_RECOVERY!=1")

    if os.getenv("ENABLE_NIGHTLY_PAGE_SYNC") == "1":
        _register_job(
            "nightly_page_sync",
            _job_wrapper("nightly_page_sync", nightly_page_sync, notify_skip=_notify_admin_skip),
            "cron",
            id="nightly_page_sync",
            hour="2",
            minute="30",
            args=[db],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
    else:
        logging.info("SCHED skipping nightly_page_sync (ENABLE_NIGHTLY_PAGE_SYNC!=1)")

    # Pinned button update at 18:00 Kaliningrad time (UTC+2 = 16:00 UTC)
    from handlers.pinned_button import pinned_button_scheduler
    
    pinned_tz = _safe_zoneinfo("Europe/Kaliningrad", label="PINNED_BUTTON_TZ")
    pinned_local = datetime.now(pinned_tz).replace(hour=18, minute=0, second=0, microsecond=0)
    pinned_utc = pinned_local.astimezone(timezone.utc)
    _register_job(
        "pinned_button_scheduler",
        _job_wrapper("pinned_button_scheduler", pinned_button_scheduler, notify_skip=_notify_admin_skip),
        "cron",
        id="pinned_button_scheduler",
        hour=str(pinned_utc.hour),
        minute=str(pinned_utc.minute),
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    async def _run_maintenance(job, name: str, timeout: float, run_id: str | None = None) -> None:
        start = _time.perf_counter()
        try:
            await asyncio.wait_for(job(), timeout=timeout)
            dur = (_time.perf_counter() - start) * 1000
            logging.info("db_maintenance %s done in %.0f ms", name, dur)
        except asyncio.TimeoutError:
            logging.warning(
                "db_maintenance %s timed out after %.1f s", name, timeout
            )
        except Exception:
            logging.warning("db_maintenance %s failed", name, exc_info=True)

    if db is not None:
        try:
            from source_parsing.post_metrics import cleanup_post_metrics
        except Exception:
            cleanup_post_metrics = None  # type: ignore[assignment]

        _register_job(
            "db_optimize",
            _job_wrapper("db_optimize", _run_maintenance, notify_skip=_notify_admin_skip),
            "interval",
            id="db_optimize",
            hours=1,
            args=[partial(optimize, db.engine), "PRAGMA optimize", 10.0],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _register_job(
            "db_wal_checkpoint",
            _job_wrapper("db_wal_checkpoint", _run_maintenance, notify_skip=_notify_admin_skip),
            "interval",
            id="db_wal_checkpoint",
            hours=1,
            args=[partial(wal_checkpoint_truncate, db.engine), "PRAGMA wal_checkpoint(TRUNCATE)", 30.0],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        _register_job(
            "db_vacuum",
            _job_wrapper("db_vacuum", _run_maintenance, notify_skip=_notify_admin_skip),
            "interval",
            id="db_vacuum",
            hours=12,
            args=[partial(vacuum, db.engine), "VACUUM", 120.0],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )
        if cleanup_post_metrics is not None:
            _register_job(
                "post_metrics_cleanup",
                _job_wrapper("post_metrics_cleanup", _run_maintenance, notify_skip=_notify_admin_skip),
                "interval",
                id="post_metrics_cleanup",
                hours=24,
                args=[partial(cleanup_post_metrics, db), "post_metrics_cleanup", 20.0],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )

    _scheduler.add_listener(
        _on_event,
        EVENT_JOB_SUBMITTED | EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
    )
    _scheduler.start()
    return _scheduler


def cleanup() -> None:
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            logging.exception("scheduler shutdown failed")
