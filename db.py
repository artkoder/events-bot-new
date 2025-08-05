import asyncio
from contextlib import asynccontextmanager

import aiosqlite
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from models import create_all


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
                    await self._conn.execute("PRAGMA read_uncommitted = 1")
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

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            await conn.run_sync(create_all)
            result = await conn.exec_driver_sql("PRAGMA table_info(event)")
            cols = [r[1] for r in result.fetchall()]
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

            result = await conn.exec_driver_sql("PRAGMA table_info(weekendpage)")
            cols = [r[1] for r in result.fetchall()]
            if "vk_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE weekendpage ADD COLUMN vk_post_url VARCHAR"
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

            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_events_date_city ON event(date, city)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_events_added_at ON event(added_at)"
            )

        # ensure shared connection is ready
        await self.raw_conn()

    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        async with self._session_factory() as session:
            yield session

