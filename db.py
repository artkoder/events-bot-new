import asyncio
import logging
import os
import time as _time
from contextlib import asynccontextmanager

import aiosqlite
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.pool import NullPool

from models import create_all


DEBUG_SQL_PLAN = os.getenv("DEBUG_SQL_PLAN") == "1"


async def pragma(conn, sql: str) -> None:
    await conn.exec_driver_sql(sql)


async def wal_checkpoint_truncate(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await pragma(conn, "PRAGMA wal_checkpoint(TRUNCATE)")


async def optimize(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await pragma(conn, "PRAGMA optimize")


async def vacuum(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await pragma(conn, "VACUUM")


async def explain_sql(
    engine: AsyncEngine, sql: str, *args, timeout: float = 0.5, **kwargs
) -> str:
    async def _run() -> str:
        async with engine.connect() as conn:
            result = await conn.exec_driver_sql(
                f"EXPLAIN QUERY PLAN {sql}", *args, **kwargs
            )
            try:
                rows = result.fetchall()
            finally:
                result.close()
            return "\n".join(" ".join(map(str, r)) for r in rows)

    try:
        return await asyncio.wait_for(_run(), timeout)
    except Exception as e:  # pragma: no cover - logging only
        return f"<explain failed: {e}>"


class LoggingAsyncSession(AsyncSession):
    async def _log(self, sql: str, params, duration: float) -> None:
        if duration <= 2000:
            return
        logging.warning("SLOW SQL %.0f ms: %s", duration, sql)
        if DEBUG_SQL_PLAN:
            args = (params,) if params is not None else ()
            plan = await explain_sql(self.bind, sql, *args)
            logging.warning("PLAN: %s", plan)

    async def execute(self, statement, params=None, *args, **kwargs):
        compile_fn = getattr(statement, "compile", None)
        if compile_fn:
            sql = str(
                compile_fn(
                    self.bind.sync_engine if hasattr(self.bind, "sync_engine") else self.bind,
                    compile_kwargs={
                        "literal_binds": False,
                        "render_postcompile": True,
                    },
                )
            )
        else:
            sql = str(statement)
        start = _time.perf_counter() * 1000
        result = await super().execute(statement, params=params, *args, **kwargs)
        dur = _time.perf_counter() * 1000 - start
        await self._log(sql, params, dur)
        return result

    async def scalars(self, statement, params=None, *args, **kwargs):
        compile_fn = getattr(statement, "compile", None)
        if compile_fn:
            sql = str(
                compile_fn(
                    self.bind.sync_engine if hasattr(self.bind, "sync_engine") else self.bind,
                    compile_kwargs={
                        "literal_binds": False,
                        "render_postcompile": True,
                    },
                )
            )
        else:
            sql = str(statement)
        start = _time.perf_counter() * 1000
        result = await super().scalars(statement, params=params, *args, **kwargs)
        dur = _time.perf_counter() * 1000 - start
        await self._log(sql, params, dur)
        return result

class Database:
    def __init__(self, path: str):
        """Initialize async engine and reusable aiosqlite connection."""
        self.path = path
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{path}",
            poolclass=NullPool,
            connect_args={"timeout": 15, "check_same_thread": False},
        )
        self._session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=LoggingAsyncSession,
        )
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def raw_conn(self) -> aiosqlite.Connection:
        """Return a shared aiosqlite connection."""
        if self._conn is None:
            async with self._lock:
                if self._conn is None:
                    self._conn = await aiosqlite.connect(self.path, timeout=15)
                    await self._conn.execute("PRAGMA journal_mode=WAL")
                    await self._conn.execute("PRAGMA synchronous=NORMAL")
                    await self._conn.execute("PRAGMA cache_size=-40000")
                    await self._conn.execute("PRAGMA temp_store=MEMORY")
                    await self._conn.execute("PRAGMA busy_timeout=5000")
                    await self._conn.execute("PRAGMA read_uncommitted = 1")
                    await self._conn.execute("PRAGMA mmap_size=134217728")
        return self._conn

    @asynccontextmanager
    async def read_only(self):
        """Context manager for read-only transactions."""
        conn = await self.raw_conn()
        await conn.execute("BEGIN")
        try:
            yield conn
        finally:
            await conn.rollback()

    async def exec_driver_sql(self, sql: str, *args, **kwargs):
        start = _time.perf_counter()
        async with self.engine.begin() as conn:
            result = await conn.exec_driver_sql(sql, *args, **kwargs)
            try:
                rows = result.fetchall() if result.returns_rows else None
            finally:
                result.close()
            dur = (_time.perf_counter() - start) * 1000
            if dur > 2000:
                logging.warning("SLOW SQL %.0f ms: %s", dur, sql)
                if DEBUG_SQL_PLAN:
                    plan = await explain_sql(self.engine, sql, *args, **kwargs)
                    logging.warning("PLAN: %s", plan)
        return rows

    async def init(self):
        async with self.engine.begin() as conn:
            if self.engine.url.get_backend_name() == "sqlite":
                await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                await conn.exec_driver_sql("PRAGMA temp_store=MEMORY;")
                await conn.exec_driver_sql("PRAGMA cache_size=-40000;")
                await conn.exec_driver_sql("PRAGMA busy_timeout=5000;")
                await conn.exec_driver_sql("PRAGMA mmap_size=134217728;")
            await conn.run_sync(create_all)
            result = await conn.exec_driver_sql("PRAGMA table_info(event)")
            event_cols = [r[1] for r in result.fetchall()]
            cols = event_cols
            if "telegraph_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN telegraph_url VARCHAR"
                )
            if "ticket_price_min" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ticket_price_min INTEGER"
                )
            if "ticket_price_max" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ticket_price_max INTEGER"
                )
            if "ticket_link" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ticket_link VARCHAR"
                )
            if "source_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_post_url VARCHAR"
                )
            if "source_vk_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_vk_post_url VARCHAR"
                )
            if "is_free" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN is_free BOOLEAN DEFAULT 0"
                )
            if "silent" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN silent BOOLEAN DEFAULT 0"
                )
            if "telegraph_path" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN telegraph_path VARCHAR"
                )
            if "event_type" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN event_type VARCHAR"
                )
            if "emoji" not in cols:
                await conn.exec_driver_sql("ALTER TABLE event ADD COLUMN emoji VARCHAR")
            if "end_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN end_date VARCHAR"
                )
            if "added_at" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN added_at VARCHAR"
                )
            if "photo_count" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN photo_count INTEGER DEFAULT 0"
                )
            if "pushkin_card" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN pushkin_card BOOLEAN DEFAULT 0"
                )
            if "ics_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ics_url VARCHAR"
                )
            if "ics_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ics_post_url VARCHAR"
                )
            if "ics_post_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ics_post_id INTEGER"
                )
            if "source_chat_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_chat_id INTEGER"
                )
            if "source_message_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_message_id INTEGER"
                )
            if "creator_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN creator_id INTEGER"
                )
            if "content_hash" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN content_hash VARCHAR"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(user)")
            cols = [r[1] for r in result.fetchall()]
            if "is_partner" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN is_partner BOOLEAN DEFAULT 0"
                )
            if "organization" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN organization VARCHAR"
                )
            if "location" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN location VARCHAR"
                )
            if "blocked" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN blocked BOOLEAN DEFAULT 0"
                )
            if "last_partner_reminder" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN last_partner_reminder VARCHAR"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(channel)")
            cols = [r[1] for r in result.fetchall()]
            if "daily_time" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE channel ADD COLUMN daily_time VARCHAR"
                )
            if "last_daily" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE channel ADD COLUMN last_daily VARCHAR"
                )
            if "is_asset" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE channel ADD COLUMN is_asset BOOLEAN DEFAULT 0"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(monthpage)")
            cols = [r[1] for r in result.fetchall()]
            if "url2" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE monthpage ADD COLUMN url2 VARCHAR"
                )
            if "path2" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE monthpage ADD COLUMN path2 VARCHAR"
                )
            if "content_hash" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE monthpage ADD COLUMN content_hash TEXT"
                )
            if "content_hash2" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE monthpage ADD COLUMN content_hash2 TEXT"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(weekendpage)")
            cols = [r[1] for r in result.fetchall()]
            if "vk_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE weekendpage ADD COLUMN vk_post_url VARCHAR"
                )
            if "content_hash" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE weekendpage ADD COLUMN content_hash TEXT"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(festival)")
            cols = [r[1] for r in result.fetchall()]
            if "full_name" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN full_name VARCHAR"
                )
                await conn.exec_driver_sql(
                    "UPDATE festival SET full_name = name"
                )
            if "photo_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN photo_url VARCHAR"
                )
            if "website_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN website_url VARCHAR"
                )
            if "vk_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN vk_url VARCHAR"
                )
            if "tg_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN tg_url VARCHAR"
                )
            if "start_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN start_date VARCHAR"
                )
            if "end_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN end_date VARCHAR"
                )
            if "vk_poll_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN vk_poll_url VARCHAR"
                )
            if "location_name" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN location_name VARCHAR"
                )
            if "location_address" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN location_address VARCHAR"
                )
            if "city" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN city VARCHAR"
                )
            if "source_text" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN source_text TEXT"
                )

            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_festival_name ON festival(name)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_date_time ON event(date, time)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_festival_date_time ON event(festival, date, time)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_type_dates ON event(event_type, date, end_date)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_added_at ON event(added_at)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_creator_added ON event(creator_id, added_at DESC)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_city_date_time ON event(city, date, time)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_date ON event(date)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_end_date ON event(end_date)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_city ON event(city)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON event(event_type)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_event_is_free ON event(is_free)"
            )
            if "month" in event_cols:
                await conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_event_month ON event(month)"
                )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_event_date_city ON event(date, city)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_event_date_festival ON event(date, festival)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_event_content_hash ON event(content_hash)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_week_page_start ON weekpage(start)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_event_telegraph_not_null ON event(date) WHERE telegraph_url IS NOT NULL"
            )

        # ensure shared connection is ready
        await self.raw_conn()

    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        async with self._session_factory() as session:
            yield session

