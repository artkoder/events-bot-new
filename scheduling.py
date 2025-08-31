from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from uuid import uuid4

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    EVENT_JOB_SUBMITTED,
)
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from db import optimize, wal_checkpoint_truncate, vacuum


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


def _job_wrapper(job_id: str, func):
    async def _run(*args):
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
            return await func(*args, run_id=run_id)
        finally:
            done.set()
            hb_task.cancel()

    return _run


def _on_event(event):
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
        next_run = job.next_run_time if job else None
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


def startup(db, bot) -> AsyncIOScheduler:
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

    from main import (
        vk_scheduler,
        vk_poll_scheduler,
        cleanup_scheduler,
        partner_notification_scheduler,
        nightly_page_sync,
        rebuild_fest_nav_if_changed,
    )

    _scheduler.add_job(
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
    _scheduler.add_job(
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
    _scheduler.add_job(
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
    _scheduler.add_job(
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

    _scheduler.add_job(
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

    if os.getenv("ENABLE_NIGHTLY_PAGE_SYNC") == "1":
        _scheduler.add_job(
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
        _scheduler.add_job(
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
        _scheduler.add_job(
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
        _scheduler.add_job(
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

    _scheduler.add_listener(_on_event)
    _scheduler.start()
    return _scheduler


def cleanup() -> None:
    if _scheduler:
        _scheduler.shutdown(wait=False)
