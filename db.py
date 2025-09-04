from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import aiosqlite
from sqlalchemy.ext.asyncio import AsyncConnection


async def _add_column(conn, table: str, col_def: str) -> None:
    try:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except Exception as e:
        if "duplicate column name" not in str(e).lower():
            raise


class Database:
    def __init__(self, path: str):
        self.path = path
        self._conn: aiosqlite.Connection | None = None
        self._orm_engine = None
        self._sessionmaker = None

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA cache_size=-40000")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("PRAGMA mmap_size=134217728")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user(
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    is_superadmin BOOLEAN DEFAULT 0,
                    is_partner BOOLEAN DEFAULT 0,
                    organization TEXT,
                    location TEXT,
                    blocked BOOLEAN DEFAULT 0,
                    last_partner_reminder TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pendinguser(
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    requested_at TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rejecteduser(
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    rejected_at TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel(
                    channel_id INTEGER PRIMARY KEY,
                    title TEXT,
                    username TEXT,
                    is_admin BOOLEAN DEFAULT 0,
                    is_registered BOOLEAN DEFAULT 0,
                    is_asset BOOLEAN DEFAULT 0,
                    daily_time TEXT,
                    last_daily TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS setting(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    festival TEXT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    location_name TEXT NOT NULL,
                    location_address TEXT,
                    city TEXT,
                    ticket_price_min INTEGER,
                    ticket_price_max INTEGER,
                    ticket_link TEXT,
                    event_type TEXT,
                    emoji TEXT,
                    end_date TEXT,
                    is_free BOOLEAN DEFAULT 0,
                    pushkin_card BOOLEAN DEFAULT 0,
                    silent BOOLEAN DEFAULT 0,
                    telegraph_path TEXT,
                    source_text TEXT NOT NULL,
                    telegraph_url TEXT,
                    ics_url TEXT,
                    source_post_url TEXT,
                    source_vk_post_url TEXT,
                    ics_hash TEXT,
                    ics_file_id TEXT,
                    ics_updated_at TIMESTAMP,
                    ics_post_url TEXT,
                    ics_post_id INTEGER,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    creator_id INTEGER,
                    photo_urls JSON,
                    photo_count INTEGER DEFAULT 0,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT
                )
                """
            )
            await _add_column(conn, "event", "photo_urls JSON")
            await _add_column(conn, "event", "ics_hash TEXT")
            await _add_column(conn, "event", "ics_file_id TEXT")
            await _add_column(conn, "event", "ics_updated_at TIMESTAMP")
            await _add_column(conn, "event", "ics_post_url TEXT")
            await _add_column(conn, "event", "ics_post_id INTEGER")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monthpage(
                    month TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    path TEXT NOT NULL,
                    url2 TEXT,
                    path2 TEXT,
                    content_hash TEXT,
                    content_hash2 TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weekendpage(
                    start TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    path TEXT NOT NULL,
                    vk_post_url TEXT,
                    content_hash TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weekpage(
                    start TEXT PRIMARY KEY,
                    vk_post_url TEXT,
                    content_hash TEXT
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS festival(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    full_name TEXT,
                    description TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    telegraph_url TEXT,
                    telegraph_path TEXT,
                    vk_post_url TEXT,
                    vk_poll_url TEXT,
                    photo_url TEXT,
                    photo_urls JSON,
                    website_url TEXT,
                    program_url TEXT,
                    vk_url TEXT,
                    tg_url TEXT,
                    ticket_url TEXT,
                    location_name TEXT,
                    location_address TEXT,
                    city TEXT,
                    source_text TEXT
                )
                """
            )
            await _add_column(conn, "festival", "location_name TEXT")
            await _add_column(conn, "festival", "location_address TEXT")
            await _add_column(conn, "festival", "city TEXT")
            await _add_column(conn, "festival", "program_url TEXT")
            await _add_column(conn, "festival", "ticket_url TEXT")
            await _add_column(conn, "festival", "nav_hash TEXT")
            await _add_column(conn, "festival", "photo_urls JSON")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS joboutbox(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    task TEXT NOT NULL,
                    payload TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_error TEXT,
                    last_result TEXT,
                    coalesce_key TEXT,
                    depends_on TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await _add_column(conn, "joboutbox", "last_result TEXT")
            await _add_column(conn, "joboutbox", "coalesce_key TEXT")
            await _add_column(conn, "joboutbox", "depends_on TEXT")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS page_section_cache(
                    page_key TEXT NOT NULL,
                    section_key TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(page_key, section_key)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_psc_page ON page_section_cache(page_key)"
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_festival_name ON festival(name)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_date ON event(date)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_end_date ON event(end_date)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_city ON event(city)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON event(event_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_is_free ON event(is_free)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_date_city ON event(date, city)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_date_festival ON event(date, festival)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_event_content_hash ON event(content_hash)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_date_time ON event(date, time)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_festival_date_time ON event(festival, date, time)"
            )

            await conn.commit()

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.path, timeout=15)
        return self._conn

    @asynccontextmanager
    async def raw_conn(self):
        conn = await self._ensure_conn()
        yield conn

    @asynccontextmanager
    async def get_session(self):
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        if self._orm_engine is None:
            self._orm_engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.path}", future=True
            )
        if self._sessionmaker is None:
            self._sessionmaker = sessionmaker(
                self._orm_engine, expire_on_commit=False, class_=AsyncSession
            )
        async with self._sessionmaker() as session:
            yield session

    @property
    def engine(self):
        from sqlalchemy.ext.asyncio import create_async_engine

        if self._orm_engine is None:
            self._orm_engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.path}", future=True
            )
        return self._orm_engine

    async def exec_driver_sql(
        self, sql: str, params: tuple | dict | None = None
    ):
        async with self.engine.begin() as conn:  # type: AsyncConnection
            result = await conn.exec_driver_sql(sql, params or ())
            try:
                return result.fetchall()
            except Exception:
                return []


async def wal_checkpoint_truncate(engine):
    async with engine.begin() as conn:
        result = await conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        rows = result.fetchall()
    logging.info("db_checkpoint result=%s", rows)
    return rows


async def optimize(engine):
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA optimize")


async def vacuum(engine):
    async with engine.begin() as conn:
        await conn.exec_driver_sql("VACUUM")


