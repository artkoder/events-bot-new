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

            pragma_cursor = await conn.execute("PRAGMA table_info('posterocrcache')")
            poster_ocr_columns = await pragma_cursor.fetchall()
            await pragma_cursor.close()

            detail_exists = any(col[1] == "detail" for col in poster_ocr_columns)
            model_exists = any(col[1] == "model" for col in poster_ocr_columns)
            created_at_exists = any(col[1] == "created_at" for col in poster_ocr_columns)
            pk_columns: list[str] = []
            if poster_ocr_columns:
                pk_info = sorted(
                    ((col[5], col[1]) for col in poster_ocr_columns if col[5]),
                    key=lambda item: item[0],
                )
                pk_columns = [name for _, name in pk_info]

            expected_pk = ["hash", "detail", "model"]
            needs_posterocr_migration = False
            if poster_ocr_columns:
                if not detail_exists or not model_exists:
                    needs_posterocr_migration = True
                elif pk_columns != expected_pk:
                    needs_posterocr_migration = True

            if needs_posterocr_migration:
                await conn.execute("DROP TABLE IF EXISTS posterocrcache_new")
                await conn.execute(
                    """
                    CREATE TABLE posterocrcache_new(
                        hash TEXT NOT NULL,
                        detail TEXT NOT NULL,
                        model TEXT NOT NULL,
                        text TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (hash, detail, model)
                    )
                    """
                )

                detail_default = "auto"
                model_default = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")

                detail_expr = "COALESCE(detail, ?)" if detail_exists else "?"
                model_expr = "COALESCE(model, ?)" if model_exists else "?"
                created_at_expr = "created_at" if created_at_exists else "CURRENT_TIMESTAMP"

                insert_sql = f"""
                    INSERT INTO posterocrcache_new (
                        hash, detail, model, text,
                        prompt_tokens, completion_tokens, total_tokens, created_at
                    )
                    SELECT
                        hash,
                        {detail_expr},
                        {model_expr},
                        text,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        {created_at_expr}
                    FROM posterocrcache
                """

                params: list[str] = []
                params.append(detail_default)
                params.append(model_default)

                await conn.execute(insert_sql, params)
                await conn.execute("DROP TABLE posterocrcache")
                await conn.execute("ALTER TABLE posterocrcache_new RENAME TO posterocrcache")

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
                    last_partner_reminder TIMESTAMP WITH TIME ZONE
                        -- Existing deployments should backfill naive values as UTC.
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
                    vk_repost_url TEXT,
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
                    topics TEXT DEFAULT '[]',
                    topics_manual BOOLEAN DEFAULT 0,
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
            await _add_column(conn, "event", "vk_repost_url TEXT")
            await _add_column(conn, "event", "vk_ticket_short_url TEXT")
            await _add_column(conn, "event", "vk_ticket_short_key TEXT")
            await _add_column(conn, "event", "vk_ics_short_url TEXT")
            await _add_column(conn, "event", "vk_ics_short_key TEXT")
            await _add_column(conn, "event", "topics TEXT DEFAULT '[]'")
            await _add_column(conn, "event", "topics_manual BOOLEAN DEFAULT 0")
            await _add_column(conn, "event", "tourist_label SMALLINT")
            await _add_column(conn, "event", "tourist_factors TEXT")
            await _add_column(conn, "event", "tourist_note TEXT")
            await _add_column(conn, "event", "tourist_label_by INTEGER")
            await _add_column(conn, "event", "tourist_label_at TIMESTAMP")
            await _add_column(conn, "event", "tourist_label_source TEXT")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_tourist_label ON event(tourist_label)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eventposter(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    catbox_url TEXT,
                    poster_hash TEXT NOT NULL,
                    ocr_text TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(event_id) REFERENCES event(id) ON DELETE CASCADE,
                    UNIQUE(event_id, poster_hash)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_eventposter_event ON eventposter(event_id)"
            )

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
                    aliases JSON,
                    website_url TEXT,
                    program_url TEXT,
                    vk_url TEXT,
                    tg_url TEXT,
                    ticket_url TEXT,
                    location_name TEXT,
                    location_address TEXT,
                    city TEXT,
                    source_text TEXT,
                    source_post_url TEXT,
                    source_chat_id INTEGER,
                    source_message_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            await _add_column(conn, "festival", "aliases JSON")
            await _add_column(conn, "festival", "source_post_url TEXT")
            await _add_column(conn, "festival", "source_chat_id INTEGER")
            await _add_column(conn, "festival", "source_message_id INTEGER")

            festival_cursor = await conn.execute("PRAGMA table_info('festival')")
            festival_columns = await festival_cursor.fetchall()
            await festival_cursor.close()
            festival_column_names = {column[1] for column in festival_columns}
            if "created_at" not in festival_column_names:
                await conn.execute("ALTER TABLE festival ADD COLUMN created_at TIMESTAMP")
                await conn.execute(
                    "UPDATE festival SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                )

            await conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS festival_set_created_at
                AFTER INSERT ON festival
                FOR EACH ROW
                WHEN NEW.created_at IS NULL
                BEGIN
                    UPDATE festival
                    SET created_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END;
                """
            )

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
                CREATE TABLE IF NOT EXISTS vk_source(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    screen_name TEXT,
                    name TEXT,
                    location TEXT,
                    default_time TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_vk_source_group ON vk_source(group_id)"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_tmp_post(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    group_id INTEGER NOT NULL,
                    post_id INTEGER NOT NULL,
                    date INTEGER NOT NULL,
                    text TEXT,
                    photos JSON,
                    url TEXT
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_vk_tmp_post_batch ON vk_tmp_post(batch, id)"
            )

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
                """
                CREATE TABLE IF NOT EXISTS vk_crawl_cursor (
                    group_id     INTEGER PRIMARY KEY,
                    last_seen_ts INTEGER DEFAULT 0,
                    last_post_id INTEGER DEFAULT 0,
                    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_inbox (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id     INTEGER NOT NULL,
                    post_id      INTEGER NOT NULL,
                    date         INTEGER NOT NULL,
                    text         TEXT NOT NULL,
                    matched_kw   TEXT,
                    has_date     INTEGER NOT NULL,
                    event_ts_hint INTEGER,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    locked_by    INTEGER,
                    locked_at    TIMESTAMP,
                    imported_event_id INTEGER,
                    review_batch TEXT,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_vk_inbox_unique ON vk_inbox(group_id, post_id)"
            )

            await _add_column(conn, "vk_inbox", "event_ts_hint INTEGER")

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vk_review_batch (
                    batch_id     TEXT PRIMARY KEY,
                    operator_id  INTEGER NOT NULL,
                    months_csv   TEXT NOT NULL,
                    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at  TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posterocrcache(
                    hash TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (hash, detail, model)
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocrusage(
                    date TEXT PRIMARY KEY,
                    spent_tokens INTEGER NOT NULL DEFAULT 0
                )
                """
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


