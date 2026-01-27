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

from db import optimize, wal_checkpoint_truncate, vacuum
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


def _job_next_run(job):
    return getattr(job, "next_run_time", None) or getattr(job, "next_run_at", None)


def _job_wrapper(job_id: str, func):
    async def _run(*args, **kwargs):
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

        hb_task = asyncio.create_task(heartbeat())
        try:
            return await func(*args, run_id=run_id, **kwargs)
        finally:
            done.set()
            hb_task.cancel()

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

    def _notify_admin_skip(job_name: str, reason: str) -> None:
        admin_chat_id = os.getenv("ADMIN_CHAT_ID")
        if not admin_chat_id:
            return
        try:
            chat_id = int(admin_chat_id)
        except (TypeError, ValueError):
            logging.warning("SCHED invalid ADMIN_CHAT_ID=%r", admin_chat_id)
            return
        if bot is None or not hasattr(bot, "send_message"):
            return
        text = f"⚠️ SCHED: пропуск {job_name}. Причина: {reason}"
        try:
            asyncio.create_task(bot.send_message(chat_id, text))
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

    _register_job(
        "vk_scheduler",
        _job_wrapper("vk_scheduler", vk_scheduler),
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
        _job_wrapper("vk_poll_scheduler", vk_poll_scheduler),
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
        _job_wrapper("cleanup_scheduler", cleanup_scheduler),
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
        _job_wrapper("partner_notification_scheduler", partner_notification_scheduler),
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
        _job_wrapper("fest_nav_rebuild", rebuild_fest_nav_if_changed),
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
            _job_wrapper("vk_crawl_cron", vk_crawl_cron),
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

    # Source parsing from theatres (before daily announcement at 08:00)
    enable_source_parsing = _env_enabled("ENABLE_SOURCE_PARSING", default=is_prod)
    if enable_source_parsing:
        from source_parsing.commands import source_parsing_scheduler
        parsing_time_raw = os.getenv("SOURCE_PARSING_TIME_LOCAL", "02:15").strip()
        parsing_tz_name = os.getenv("SOURCE_PARSING_TZ", "Europe/Kaliningrad")
        parsing_hour, parsing_minute = _cron_from_local(
            parsing_time_raw,
            parsing_tz_name,
            default_hour="2",
            default_minute="0",
            label="SOURCE_PARSING_TIME_LOCAL",
        )
        _register_job(
            "source_parsing",
            _job_wrapper("source_parsing", source_parsing_scheduler),
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
            _job_wrapper("source_parsing_day", source_parsing_scheduler_if_changed),
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
            _job_wrapper("tg_monitoring", telegram_monitor_scheduler),
            "cron",
            id="tg_monitoring",
            hour=tg_hour,
            minute=tg_minute,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
    else:
        logging.info("SCHED skipping tg_monitoring (ENABLE_TG_MONITORING!=1)")
        _notify_admin_skip("tg_monitoring", "ENABLE_TG_MONITORING!=1")

    enable_3di = _env_enabled("ENABLE_3DI_SCHEDULED", default=is_prod)
    if enable_3di:
        from preview_3d.handlers import run_3di_new_only_scheduler
        admin_chat_id = os.getenv("ADMIN_CHAT_ID")
        run_chat_id = int(admin_chat_id) if admin_chat_id else None
        three_di_times = os.getenv("THREEDI_TIMES_LOCAL", "03:15,15:15,17:15")
        three_di_tz = os.getenv("THREEDI_TZ", "Europe/Kaliningrad")
        for idx, t in enumerate(three_di_times.split(",")):
            t = t.strip()
            if not t:
                continue
            hour, minute = _cron_from_local(
                t,
                three_di_tz,
                default_hour="3",
                default_minute="15",
                label="THREEDI_TIMES_LOCAL",
            )
            _register_job(
                f"3di_scheduler_{idx}",
                _job_wrapper("3di_scheduler", run_3di_new_only_scheduler),
                "cron",
                id=f"3di_scheduler_{idx}",
                hour=hour,
                minute=minute,
                args=[db, bot],
                kwargs={"chat_id": run_chat_id},
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30,
            )
    else:
        logging.info("SCHED skipping 3di_scheduler (ENABLE_3DI_SCHEDULED!=1)")
        _notify_admin_skip("3di_scheduler", "ENABLE_3DI_SCHEDULED!=1")

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
            _job_wrapper("kaggle_recovery", kaggle_recovery_scheduler),
            "interval",
            id="kaggle_recovery",
            minutes=interval_min,
            args=[db, bot],
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
    else:
        logging.info("SCHED skipping kaggle_recovery (ENABLE_KAGGLE_RECOVERY!=1)")
        _notify_admin_skip("kaggle_recovery", "ENABLE_KAGGLE_RECOVERY!=1")

    if os.getenv("ENABLE_NIGHTLY_PAGE_SYNC") == "1":
        _register_job(
            "nightly_page_sync",
            _job_wrapper("nightly_page_sync", nightly_page_sync),
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
        _job_wrapper("pinned_button_scheduler", pinned_button_scheduler),
        "cron",
        id="pinned_button_scheduler",
        hour=str(pinned_utc.hour),
        minute=str(pinned_utc.minute),
        args=[db, bot],
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
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
        _register_job(
            "db_optimize",
            _job_wrapper("db_optimize", _run_maintenance),
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
            _job_wrapper("db_wal_checkpoint", _run_maintenance),
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
            _job_wrapper("db_vacuum", _run_maintenance),
            "interval",
            id="db_vacuum",
            hours=12,
            args=[partial(vacuum, db.engine), "VACUUM", 120.0],
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
